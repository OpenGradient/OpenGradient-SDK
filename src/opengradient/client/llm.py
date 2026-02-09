"""LLM chat and completion via TEE-verified execution with x402 payments."""

import asyncio
import json
from typing import Dict, List, Optional, Union

import httpx
from eth_account.account import LocalAccount
from x402.clients.base import x402Client
from x402.clients.httpx import x402HttpxClient

from ..defaults import (
    DEFAULT_NETWORK_FILTER,
)
from ..types import (
    TEE_LLM,
    StreamChunk,
    TextGenerationOutput,
    TextGenerationStream,
    x402SettlementMode,
)
from .exceptions import OpenGradientError
from .x402_auth import X402Auth

X402_PROCESSING_HASH_HEADER = "x-processing-hash"
X402_PLACEHOLDER_API_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

TIMEOUT = httpx.Timeout(
    timeout=90.0,
    connect=15.0,
    read=15.0,
    write=30.0,
    pool=10.0,
)
LIMITS = httpx.Limits(
    max_keepalive_connections=100,
    max_connections=500,
    keepalive_expiry=60 * 20,  # 20 minutes
)


class LLM:
    """
    LLM inference namespace.

    Provides access to large language model completions and chat via TEE
    (Trusted Execution Environment) with x402 payment protocol support.
    Supports both streaming and non-streaming responses.

    Usage:
        client = og.Client(...)
        result = client.llm.chat(model=TEE_LLM.CLAUDE_3_5_HAIKU, messages=[...])
        result = client.llm.completion(model=TEE_LLM.CLAUDE_3_5_HAIKU, prompt="Hello")
    """

    def __init__(
        self,
        wallet_account: LocalAccount,
        og_llm_server_url: str,
        og_llm_streaming_server_url: str,
        network_filter: str = DEFAULT_NETWORK_FILTER,
    ):
        self._wallet_account = wallet_account
        self._og_llm_server_url = og_llm_server_url
        self._og_llm_streaming_server_url = og_llm_streaming_server_url
        self._network_filter = network_filter

    def _og_payment_selector(self, accepts, network_filter=None, scheme_filter=None, max_value=None):
        """Custom payment selector for OpenGradient network."""
        return x402Client.default_payment_requirements_selector(
            accepts,
            network_filter=network_filter or self._network_filter,
            scheme_filter=scheme_filter,
            max_value=max_value,
        )

    def completion(
        self,
        model: TEE_LLM,
        prompt: str,
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.SETTLE_BATCH,
    ) -> TextGenerationOutput:
        """
        Perform inference on an LLM model using completions via TEE.

        Args:
            model (TEE_LLM): The model to use (e.g., TEE_LLM.CLAUDE_3_5_HAIKU).
            prompt (str): The input prompt for the LLM.
            max_tokens (int): Maximum number of tokens for LLM output. Default is 100.
            stop_sequence (List[str], optional): List of stop sequences for LLM. Default is None.
            temperature (float): Temperature for LLM inference, between 0 and 1. Default is 0.0.
            x402_settlement_mode (x402SettlementMode, optional): Settlement mode for x402 payments.
                - SETTLE: Records input/output hashes only (most privacy-preserving).
                - SETTLE_BATCH: Aggregates multiple inferences into batch hashes (most cost-efficient).
                - SETTLE_METADATA: Records full model info, complete input/output data, and all metadata.
                Defaults to SETTLE_BATCH.

        Returns:
            TextGenerationOutput: Generated text results including:
                - Transaction hash ("external" for TEE providers)
                - String of completion output
                - Payment hash for x402 transactions

        Raises:
            OpenGradientError: If the inference fails.
        """
        return self._tee_llm_completion(
            model=model.split("/")[1],
            prompt=prompt,
            max_tokens=max_tokens,
            stop_sequence=stop_sequence,
            temperature=temperature,
            x402_settlement_mode=x402_settlement_mode,
        )

    def _tee_llm_completion(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.SETTLE_BATCH,
    ) -> TextGenerationOutput:
        """
        Route completion request to OpenGradient TEE LLM server with x402 payments.
        """

        async def make_request():
            # Security Fix: verify=True enabled
            async with x402HttpxClient(
                account=self._wallet_account,
                base_url=self._og_llm_server_url,
                payment_requirements_selector=self._og_payment_selector,
                verify=True,
            ) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
                    "X-SETTLEMENT-TYPE": x402_settlement_mode,
                }

                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                if stop_sequence:
                    payload["stop"] = stop_sequence

                try:
                    response = await client.post("/v1/completions", json=payload, headers=headers, timeout=60)

                    # Read the response content
                    content = await response.aread()
                    result = json.loads(content.decode())
                    payment_hash = ""

                    if X402_PROCESSING_HASH_HEADER in response.headers:
                        payment_hash = response.headers[X402_PROCESSING_HASH_HEADER]

                    return TextGenerationOutput(
                        transaction_hash="external", completion_output=result.get("completion"), payment_hash=payment_hash
                    )

                except Exception as e:
                    raise OpenGradientError(f"TEE LLM completion request failed: {str(e)}")

        try:
            return asyncio.run(make_request())
        except OpenGradientError:
            raise
        except Exception as e:
            raise OpenGradientError(f"TEE LLM completion failed: {str(e)}")

    def chat(
        self,
        model: TEE_LLM,
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = [],
        tool_choice: Optional[str] = None,
        x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.SETTLE_BATCH,
        stream: bool = False,
    ) -> Union[TextGenerationOutput, TextGenerationStream]:
        """
        Perform inference on an LLM model using chat via TEE.

        Args:
            model (TEE_LLM): The model to use (e.g., TEE_LLM.CLAUDE_3_5_HAIKU).
            messages (List[Dict]): The messages that will be passed into the chat.
            max_tokens (int): Maximum number of tokens for LLM output. Default is 100.
            stop_sequence (List[str], optional): List of stop sequences for LLM.
            temperature (float): Temperature for LLM inference, between 0 and 1.
            tools (List[dict], optional): Set of tools for function calling.
            tool_choice (str, optional): Sets a specific tool to choose.
            x402_settlement_mode (x402SettlementMode, optional): Settlement mode for x402 payments.
                - SETTLE: Records input/output hashes only (most privacy-preserving).
                - SETTLE_BATCH: Aggregates multiple inferences into batch hashes (most cost-efficient).
                - SETTLE_METADATA: Records full model info, complete input/output data, and all metadata.
                Defaults to SETTLE_BATCH.
            stream (bool, optional): Whether to stream the response. Default is False.

        Returns:
            Union[TextGenerationOutput, TextGenerationStream]:
                - If stream=False: TextGenerationOutput with chat_output, transaction_hash, finish_reason, and payment_hash
                - If stream=True: TextGenerationStream yielding StreamChunk objects with typed deltas (true streaming via threading)

        Raises:
            OpenGradientError: If the inference fails.
        """
        if stream:
            # Use threading bridge for true sync streaming
            return self._tee_llm_chat_stream_sync(
                model=model.split("/")[1],
                messages=messages,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                x402_settlement_mode=x402_settlement_mode,
            )
        else:
            # Non-streaming
            return self._tee_llm_chat(
                model=model.split("/")[1],
                messages=messages,
                max_tokens=max_tokens,
                stop_sequence=stop_sequence,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                x402_settlement_mode=x402_settlement_mode,
            )

    def _tee_llm_chat(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.SETTLE_BATCH,
    ) -> TextGenerationOutput:
        """
        Route chat request to OpenGradient TEE LLM server with x402 payments.
        """

        async def make_request():
            # Security Fix: verify=True enabled
            async with x402HttpxClient(
                account=self._wallet_account,
                base_url=self._og_llm_server_url,
                payment_requirements_selector=self._og_payment_selector,
                verify=True,
            ) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
                    "X-SETTLEMENT-TYPE": x402_settlement_mode,
                }

                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                if stop_sequence:
                    payload["stop"] = stop_sequence

                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = tool_choice or "auto"

                try:
                    # Non-streaming with x402
                    endpoint = "/v1/chat/completions"
                    response = await client.post(endpoint, json=payload, headers=headers, timeout=60)

                    # Read the response content
                    content = await response.aread()
                    result = json.loads(content.decode())

                    payment_hash = ""
                    if X402_PROCESSING_HASH_HEADER in response.headers:
                        payment_hash = response.headers[X402_PROCESSING_HASH_HEADER]

                    choices = result.get("choices")
                    if not choices:
                        raise OpenGradientError(f"Invalid response: 'choices' missing or empty in {result}")

                    return TextGenerationOutput(
                        transaction_hash="external",
                        finish_reason=choices[0].get("finish_reason"),
                        chat_output=choices[0].get("message"),
                        payment_hash=payment_hash,
                    )

                except Exception as e:
                    raise OpenGradientError(f"TEE LLM chat request failed: {str(e)}")

        try:
            return asyncio.run(make_request())
        except OpenGradientError:
            raise
        except Exception as e:
            raise OpenGradientError(f"TEE LLM chat failed: {str(e)}")

    def _tee_llm_chat_stream_sync(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.SETTLE_BATCH,
    ):
        """
        Sync streaming using threading bridge - TRUE real-time streaming.

        Yields StreamChunk objects as they arrive from the background thread.
        NO buffering, NO conversion, just direct pass-through.
        """
        import threading
        from queue import Queue

        queue = Queue()
        exception_holder = []

        def _run_async():
            """Run async streaming in background thread"""
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def _stream():
                    try:
                        async for chunk in self._tee_llm_chat_stream_async(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            stop_sequence=stop_sequence,
                            temperature=temperature,
                            tools=tools,
                            tool_choice=tool_choice,
                            x402_settlement_mode=x402_settlement_mode,
                        ):
                            queue.put(chunk)  # Put chunk immediately
                    except Exception as e:
                        exception_holder.append(e)
                    finally:
                        queue.put(None)  # Signal completion

                loop.run_until_complete(_stream())
            except Exception as e:
                exception_holder.append(e)
                queue.put(None)
            finally:
                if loop:
                    try:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        # Properly close async generators to avoid RuntimeWarning
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    finally:
                        loop.close()

        # Start background thread
        thread = threading.Thread(target=_run_async, daemon=True)
        thread.start()

        # Yield chunks DIRECTLY as they arrive - NO buffering
        try:
            while True:
                chunk = queue.get()  # Blocks until chunk available
                if chunk is None:
                    break
                yield chunk  # Yield immediately!

            thread.join(timeout=5)

            if exception_holder:
                raise exception_holder[0]
        except Exception:
            thread.join(timeout=1)
            raise

    async def _tee_llm_chat_stream_async(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.SETTLE_BATCH,
    ):
        """
        Internal async streaming implementation for TEE LLM with x402 payments.

        Yields StreamChunk objects as they arrive from the server.
        """
        async with httpx.AsyncClient(
            base_url=self._og_llm_streaming_server_url,
            headers={"Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}"},
            timeout=TIMEOUT,
            limits=LIMITS,
            http2=False,
            follow_redirects=False,
            auth=X402Auth(account=self._wallet_account, network_filter=self._network_filter),  # type: ignore
            verify=True,
        ) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
                "X-SETTLEMENT-TYPE": x402_settlement_mode,
            }

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }

            if stop_sequence:
                payload["stop"] = stop_sequence
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice or "auto"

            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                buffer = b""
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue

                    buffer += chunk

                    # Process complete lines from buffer
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)

                        if not line_bytes.strip():
                            continue

                        try:
                            line = line_bytes.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            continue

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return

                        try:
                            data = json.loads(data_str)
                            yield StreamChunk.from_sse_data(data)
                        except json.JSONDecodeError:
                            continue
