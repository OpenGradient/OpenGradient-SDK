"""LLM chat and completion via TEE-verified execution with x402 payments."""

import asyncio
import json
import threading
from queue import Queue
from typing import AsyncGenerator, Dict, List, Optional, Union
import ssl
import socket
import tempfile
from urllib.parse import urlparse

import httpx
from eth_account.account import LocalAccount
from x402v2 import x402Client as x402Clientv2
from x402v2.http.clients import x402HttpxClient as x402HttpxClientv2
from x402v2.mechanisms.evm import EthAccountSigner as EthAccountSignerv2
from x402v2.mechanisms.evm.exact.register import register_exact_evm_client as register_exact_evm_clientv2
from x402v2.mechanisms.evm.upto.register import register_upto_evm_client as register_upto_evm_clientv2

from ..types import TEE_LLM, StreamChunk, TextGenerationOutput, TextGenerationStream, x402SettlementMode
from .exceptions import OpenGradientError
from .opg_token import Permit2ApprovalResult, ensure_opg_approval

X402_PROCESSING_HASH_HEADER = "x-processing-hash"
X402_PLACEHOLDER_API_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
BASE_TESTNET_NETWORK = "eip155:84532"

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


def _fetch_tls_cert_as_ssl_context(server_url: str) -> Optional[ssl.SSLContext]:
    """
    Connect to a server, retrieve its TLS certificate (TOFU),
    and return an ssl.SSLContext that trusts ONLY that certificate.

    Hostname verification is disabled because the TEE server's cert
    is typically issued for a hostname but we may connect via IP address.
    The pinned certificate itself provides the trust anchor.

    Returns None if the server is not HTTPS or unreachable.
    """
    parsed = urlparse(server_url)
    if parsed.scheme != "https":
        return None

    hostname = parsed.hostname
    port = parsed.port or 443

    # Connect without verification to retrieve the server's certificate
    fetch_ctx = ssl.create_default_context()
    fetch_ctx.check_hostname = False
    fetch_ctx.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with fetch_ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                der_cert = ssock.getpeercert(binary_form=True)
                pem_cert = ssl.DER_cert_to_PEM_cert(der_cert)
    except Exception:
        return None

    # Write PEM to a temp file so we can load it into the SSLContext
    cert_file = tempfile.NamedTemporaryFile(
        prefix="og_tee_tls_", suffix=".pem", delete=False, mode="w"
    )
    cert_file.write(pem_cert)
    cert_file.flush()
    cert_file.close()

    # Build an SSLContext that trusts ONLY this cert, with hostname check disabled
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cert_file.name)
    ctx.check_hostname = False  # Cert is for a hostname, but we connect via IP
    ctx.verify_mode = ssl.CERT_REQUIRED  # Still verify the cert itself
    return ctx


class LLM:
    """
    LLM inference namespace.

    Provides access to large language model completions and chat via TEE
    (Trusted Execution Environment) with x402 payment protocol support.
    Supports both streaming and non-streaming responses.

    Before making LLM requests, ensure your wallet has approved sufficient
    OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.
    This only sends an on-chain transaction when the current allowance is
    below the requested amount.

    Usage:
        client = og.Client(...)

        # One-time approval (idempotent — skips if allowance is already sufficient)
        client.llm.ensure_opg_approval(opg_amount=5)

        result = client.llm.chat(model=TEE_LLM.CLAUDE_3_5_HAIKU, messages=[...])
        result = client.llm.completion(model=TEE_LLM.CLAUDE_3_5_HAIKU, prompt="Hello")
    """

    def __init__(self, wallet_account: LocalAccount, og_llm_server_url: str, og_llm_streaming_server_url: str):
        self._wallet_account = wallet_account
        self._og_llm_server_url = og_llm_server_url
        self._og_llm_streaming_server_url = og_llm_streaming_server_url

        self._tls_verify: Union[ssl.SSLContext, bool] = (
            _fetch_tls_cert_as_ssl_context(self._og_llm_server_url) or True
        )
        self._streaming_tls_verify: Union[ssl.SSLContext, bool] = (
            _fetch_tls_cert_as_ssl_context(self._og_llm_streaming_server_url) or True
        )

        signer = EthAccountSignerv2(self._wallet_account)
        self._x402_client = x402Clientv2()
        register_exact_evm_clientv2(self._x402_client, signer, networks=[BASE_TESTNET_NETWORK])
        register_upto_evm_clientv2(self._x402_client, signer, networks=[BASE_TESTNET_NETWORK])

        self._request_client_ctx = None
        self._request_client = None
        self._stream_client_ctx = None
        self._stream_client = None
        self._closed = False

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        self._run_coroutine(self._initialize_http_clients())

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coroutine):
        if self._closed:
            raise OpenGradientError("LLM client is closed.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    async def _initialize_http_clients(self) -> None:
        if self._request_client is None:
            self._request_client_ctx = x402HttpxClientv2(self._x402_client, verify=self._tls_verify)
            self._request_client = await self._request_client_ctx.__aenter__()
        if self._stream_client is None:
            self._stream_client_ctx = x402HttpxClientv2(self._x402_client, verify=self._streaming_tls_verify)
            self._stream_client = await self._stream_client_ctx.__aenter__()

    async def _close_http_clients(self) -> None:
        if self._request_client_ctx is not None:
            await self._request_client_ctx.__aexit__(None, None, None)
            self._request_client_ctx = None
            self._request_client = None
        if self._stream_client_ctx is not None:
            await self._stream_client_ctx.__aexit__(None, None, None)
            self._stream_client_ctx = None
            self._stream_client = None

    def close(self) -> None:
        if self._closed:
            return
        self._run_coroutine(self._close_http_clients())
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    def ensure_opg_approval(self, opg_amount: float) -> Permit2ApprovalResult:
        """Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

        Checks the current Permit2 allowance for the wallet. If the allowance
        is already >= the requested amount, returns immediately without sending
        a transaction. Otherwise, sends an ERC-20 approve transaction.

        Args:
            opg_amount: Minimum number of OPG tokens required (e.g. ``0.05``
                for 0.05 OPG). Must be at least 0.05 OPG.

        Returns:
            Permit2ApprovalResult: Contains ``allowance_before``,
                ``allowance_after``, and ``tx_hash`` (None when no approval
                was needed).

        Raises:
            ValueError: If the OPG amount is less than 0.05.
            OpenGradientError: If the approval transaction fails.
        """
        if opg_amount < 0.05:
            raise ValueError("OPG amount must be at least 0.05.")
        return ensure_opg_approval(self._wallet_account, opg_amount)

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

        async def make_request_v2():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
                "X-SETTLEMENT-TYPE": x402_settlement_mode.value,
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
                response = await self._request_client.post(
                    self._og_llm_server_url + "/v1/completions", json=payload, headers=headers, timeout=60
                )

                content = await response.aread()
                result = json.loads(content.decode())

                return TextGenerationOutput(
                    transaction_hash="external",
                    completion_output=result.get("completion"),
                    tee_signature=result.get("tee_signature"),
                    tee_timestamp=result.get("tee_timestamp"),
                )

            except Exception as e:
                raise OpenGradientError(f"TEE LLM completion request failed: {str(e)}")

        try:
            return self._run_coroutine(make_request_v2())
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
        tools: Optional[List[Dict]] = None,
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

        async def make_request_v2():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
                "X-SETTLEMENT-TYPE": x402_settlement_mode.value,
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
                endpoint = "/v1/chat/completions"
                response = await self._request_client.post(
                    self._og_llm_server_url + endpoint, json=payload, headers=headers, timeout=60
                )

                response.raise_for_status()
                content = await response.aread()
                result = json.loads(content.decode())

                choices = result.get("choices")
                if not choices:
                    raise OpenGradientError(f"Invalid response: 'choices' missing or empty in {result}")

                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    message["content"] = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ).strip()

                return TextGenerationOutput(
                    transaction_hash="external",
                    finish_reason=choices[0].get("finish_reason"),
                    chat_output=message,
                    tee_signature=result.get("tee_signature"),
                    tee_timestamp=result.get("tee_timestamp"),
                )

            except Exception as e:
                raise OpenGradientError(f"TEE LLM chat request failed: {str(e)}")

        try:
            return self._run_coroutine(make_request_v2())
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
        queue = Queue()
        sentinel = object()

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
                    queue.put(chunk)
            except Exception as e:
                queue.put(e)
            finally:
                queue.put(sentinel)

        future = asyncio.run_coroutine_threadsafe(_stream(), self._loop)

        try:
            while True:
                chunk = queue.get()
                if chunk is sentinel:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        except Exception:
            if not future.done():
                future.cancel()
            raise
        finally:
            if not future.done():
                future.cancel()

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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
            "X-SETTLEMENT-TYPE": x402_settlement_mode.value,
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

        async def _parse_sse_response(response) -> AsyncGenerator[StreamChunk, None]:
            status_code = getattr(response, "status_code", None)
            if status_code is not None and status_code >= 400:
                body = await response.aread()
                body_text = body.decode("utf-8", errors="replace")
                raise OpenGradientError(f"TEE LLM streaming request failed with status {status_code}: {body_text}")

            buffer = b""
            async for chunk in response.aiter_raw():
                if not chunk:
                    continue

                buffer += chunk

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

        endpoint = "/v1/chat/completions"
        async with self._stream_client.stream(
            "POST",
            self._og_llm_streaming_server_url + endpoint,
            json=payload,
            headers=headers,
            timeout=60,
        ) as response:
            async for parsed_chunk in _parse_sse_response(response):
                yield parsed_chunk
