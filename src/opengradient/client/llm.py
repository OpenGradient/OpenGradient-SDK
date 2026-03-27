"""LLM chat and completion via TEE-verified execution with x402 payments."""

import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
import httpx
import asyncio

from eth_account import Account
from eth_account.account import LocalAccount
from x402 import x402Client
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client
from x402.mechanisms.evm.upto.register import register_upto_evm_client

from ..types import TEE_LLM, StreamChoice, StreamChunk, StreamDelta, TextGenerationOutput, x402SettlementMode
from .opg_token import Permit2ApprovalResult, ensure_opg_allowance
from .tee_connection import RegistryTEEConnection, StaticTEEConnection, TEEConnectionInterface
from .tee_registry import TEERegistry

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_RPC_URL = "https://ogevmdevnet.opengradient.ai"
DEFAULT_TEE_REGISTRY_ADDRESS = "0x4e72238852f3c918f4E4e57AeC9280dDB0c80248"

X402_PROCESSING_HASH_HEADER = "x-processing-hash"
X402_PLACEHOLDER_API_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
BASE_TESTNET_NETWORK = "eip155:84532"

_CHAT_ENDPOINT = "/v1/chat/completions"
_COMPLETION_ENDPOINT = "/v1/completions"
_REQUEST_TIMEOUT = 60


@dataclass(frozen=True)
class _ChatParams:
    """Bundles the common parameters for chat/completion requests."""

    model: str
    max_tokens: int
    temperature: float
    stop_sequence: Optional[List[str]]
    tools: Optional[List[Dict]]
    tool_choice: Optional[str]
    x402_settlement_mode: x402SettlementMode


class LLM:
    """
    LLM inference namespace.

    Provides access to large language model completions and chat via TEE
    (Trusted Execution Environment) with x402 payment protocol support.
    Supports both streaming and non-streaming responses.

    All request methods (``chat``, ``completion``) are async.

    Before making LLM requests, ensure your wallet has approved sufficient
    OPG tokens for Permit2 spending by calling ``ensure_opg_allowance``.

    Usage:
        # Via on-chain registry (default)
        llm = og.LLM(private_key="0x...")

        # Via hardcoded URL (development / self-hosted)
        llm = og.LLM.from_url(private_key="0x...", llm_server_url="https://1.2.3.4")

        # Ensure sufficient OPG allowance (only sends tx when below threshold)
        llm.ensure_opg_allowance(min_allowance=5)

        result = await llm.chat(model=TEE_LLM.CLAUDE_HAIKU_4_5, messages=[...])
        result = await llm.completion(model=TEE_LLM.CLAUDE_HAIKU_4_5, prompt="Hello")
    """

    def __init__(
        self,
        private_key: str,
        rpc_url: str = DEFAULT_RPC_URL,
        tee_registry_address: str = DEFAULT_TEE_REGISTRY_ADDRESS,
    ):
        if not private_key:
            raise ValueError("A private key is required to use the LLM client. Pass a valid private_key to the constructor.")
        self._wallet_account: LocalAccount = Account.from_key(private_key)

        x402_client = LLM._build_x402_client(private_key)
        onchain_registry = TEERegistry(rpc_url=rpc_url, registry_address=tee_registry_address)
        self._tee: TEEConnectionInterface = RegistryTEEConnection(x402_client=x402_client, registry=onchain_registry)

    @classmethod
    def from_url(
        cls,
        private_key: str,
        llm_server_url: str,
    ) -> "LLM":
        """**[Dev]** Create an LLM client with a hardcoded TEE endpoint URL.

        Intended for development and self-hosted TEE servers. TLS certificate
        verification is disabled because these servers typically use self-signed
        certificates. For production use, prefer the default constructor which
        resolves TEEs from the on-chain registry.

        Args:
            private_key: Ethereum private key for signing x402 payments.
            llm_server_url: The TEE endpoint URL (e.g. ``"https://1.2.3.4"``).
        """
        instance = cls.__new__(cls)
        if not private_key:
            raise ValueError("A private key is required to use the LLM client. Pass a valid private_key to the constructor.")
        instance._wallet_account = Account.from_key(private_key)
        x402_client = cls._build_x402_client(private_key)
        instance._tee = StaticTEEConnection(x402_client=x402_client, endpoint=llm_server_url)
        return instance

    @staticmethod
    def _build_x402_client(private_key: str) -> x402Client:
        """Build the x402 payment stack from a private key."""
        account = Account.from_key(private_key)
        signer = EthAccountSigner(account)
        client = x402Client()
        register_exact_evm_client(client, signer, networks=[BASE_TESTNET_NETWORK])
        register_upto_evm_client(client, signer, networks=[BASE_TESTNET_NETWORK])
        return client

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel the background refresh loop and close the HTTP client."""
        await self._tee.close()

    # ── Request helpers ─────────────────────────────────────────────────

    def _headers(self, settlement_mode: x402SettlementMode) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
            "X-SETTLEMENT-TYPE": settlement_mode.value,
        }

    def _chat_payload(self, params: _ChatParams, messages: List[Dict], stream: bool = False) -> Dict:
        payload: Dict = {
            "model": params.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }
        if stream:
            payload["stream"] = True
        if params.stop_sequence:
            payload["stop"] = params.stop_sequence
        if params.tools:
            payload["tools"] = params.tools
            payload["tool_choice"] = params.tool_choice or "auto"
        return payload

    async def _call_with_tee_retry(
        self,
        operation_name: str,
        call: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute *call*; on connection failure, pick a new TEE and retry once.

        Only retries when the request never reached the server (no HTTP response).
        Server-side errors (4xx/5xx) are not retried.
        """
        self._tee.ensure_refresh_loop()
        try:
            return await call()
        except httpx.HTTPStatusError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Connection failure during %s; refreshing TEE and retrying once: %s",
                operation_name,
                exc,
            )
            await self._tee.reconnect()
            return await call()

    # ── Public API ──────────────────────────────────────────────────────

    def ensure_opg_allowance(
        self,
        min_allowance: float,
        approve_amount: Optional[float] = None,
    ) -> Permit2ApprovalResult:
        """Ensure the Permit2 allowance stays above a minimum threshold.

        Only sends a transaction when the current allowance drops below
        ``min_allowance``. When approval is needed, approves ``approve_amount``
        (defaults to ``10 * min_allowance``) to create a buffer that survives
        multiple service restarts without re-approving.

        Best for backend servers that call this on startup::

            llm.ensure_opg_allowance(min_allowance=5.0, approve_amount=100.0)

        Args:
            min_allowance: The minimum acceptable allowance in OPG. Must be
                at least 0.1 OPG.
            approve_amount: The amount of OPG to approve when a transaction
                is needed. Defaults to ``10 * min_allowance``. Must be
                >= ``min_allowance``.

        Returns:
            Permit2ApprovalResult: Contains ``allowance_before``,
                ``allowance_after``, and ``tx_hash`` (None when no approval
                was needed).

        Raises:
            ValueError: If ``min_allowance`` is less than 0.1 or
                ``approve_amount`` is less than ``min_allowance``.
            RuntimeError: If the approval transaction fails.
        """
        if min_allowance < 0.1:
            raise ValueError("min_allowance must be at least 0.1.")
        return ensure_opg_allowance(self._wallet_account, min_allowance, approve_amount)

    async def completion(
        self,
        model: TEE_LLM,
        prompt: str,
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
    ) -> TextGenerationOutput:
        """
        Perform inference on an LLM model using completions via TEE.

        Args:
            model (TEE_LLM): The model to use (e.g., TEE_LLM.CLAUDE_HAIKU_4_5).
            prompt (str): The input prompt for the LLM.
            max_tokens (int): Maximum number of tokens for LLM output. Default is 100.
            stop_sequence (List[str], optional): List of stop sequences for LLM. Default is None.
            temperature (float): Temperature for LLM inference, between 0 and 1. Default is 0.0.
            x402_settlement_mode (x402SettlementMode, optional): Settlement mode for x402 payments.
                - PRIVATE: Payment only, no input/output data on-chain (most privacy-preserving).
                - BATCH_HASHED: Aggregates inferences into a Merkle tree with input/output hashes and signatures (default, most cost-efficient).
                - INDIVIDUAL_FULL: Records input, output, timestamp, and verification on-chain (maximum auditability).
                Defaults to BATCH_HASHED.

        Returns:
            TextGenerationOutput: Generated text results including:
                - Transaction hash ("external" for TEE providers)
                - String of completion output
                - Payment hash for x402 transactions

        Raises:
            RuntimeError: If the inference fails.
        """
        model_id = model.split("/")[1]
        payload: Dict = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop_sequence:
            payload["stop"] = stop_sequence

        async def _request() -> TextGenerationOutput:
            tee = self._tee.get()
            response = await tee.http_client.post(
                tee.endpoint + _COMPLETION_ENDPOINT,
                json=payload,
                headers=self._headers(x402_settlement_mode),
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw_body = await response.aread()
            result = json.loads(raw_body.decode())
            return TextGenerationOutput(
                transaction_hash="external",
                completion_output=result.get("completion"),
                tee_signature=result.get("tee_signature"),
                tee_timestamp=result.get("tee_timestamp"),
                **tee.metadata(),
            )

        try:
            return await self._call_with_tee_retry("completion", _request)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"TEE LLM completion failed: {e}") from e

    async def chat(
        self,
        model: TEE_LLM,
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
        stream: bool = False,
    ) -> Union[TextGenerationOutput, AsyncGenerator[StreamChunk, None]]:
        """
        Perform inference on an LLM model using chat via TEE.

        Args:
            model (TEE_LLM): The model to use (e.g., TEE_LLM.CLAUDE_HAIKU_4_5).
            messages (List[Dict]): The messages that will be passed into the chat.
            max_tokens (int): Maximum number of tokens for LLM output. Default is 100.
            stop_sequence (List[str], optional): List of stop sequences for LLM.
            temperature (float): Temperature for LLM inference, between 0 and 1.
            tools (List[dict], optional): Set of tools for function calling.
            tool_choice (str, optional): Sets a specific tool to choose.
            x402_settlement_mode (x402SettlementMode, optional): Settlement mode for x402 payments.
                - PRIVATE: Payment only, no input/output data on-chain (most privacy-preserving).
                - BATCH_HASHED: Aggregates inferences into a Merkle tree with input/output hashes and signatures (default, most cost-efficient).
                - INDIVIDUAL_FULL: Records input, output, timestamp, and verification on-chain (maximum auditability).
                Defaults to BATCH_HASHED.
            stream (bool, optional): Whether to stream the response. Default is False.

        Returns:
            Union[TextGenerationOutput, AsyncGenerator[StreamChunk, None]]:
                - If stream=False: TextGenerationOutput with chat_output, transaction_hash, finish_reason, and payment_hash
                - If stream=True: Async generator yielding StreamChunk objects

        Raises:
            RuntimeError: If the inference fails.
        """
        params = _ChatParams(
            model=model.split("/")[1],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequence=stop_sequence,
            tools=tools,
            tool_choice=tool_choice,
            x402_settlement_mode=x402_settlement_mode,
        )

        if not stream:
            return await self._chat_request(params, messages)

        # The TEE streaming endpoint omits tool call content from SSE events.
        # Fall back to non-streaming and emit a single final StreamChunk.
        if tools:
            return self._chat_tools_as_stream(params, messages)

        return self._chat_stream(params, messages)

    # ── Chat internals ──────────────────────────────────────────────────

    async def _chat_request(self, params: _ChatParams, messages: List[Dict]) -> TextGenerationOutput:
        """Non-streaming chat request."""
        payload = self._chat_payload(params, messages)

        async def _request() -> TextGenerationOutput:
            tee = self._tee.get()
            response = await tee.http_client.post(
                tee.endpoint + _CHAT_ENDPOINT,
                json=payload,
                headers=self._headers(params.x402_settlement_mode),
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw_body = await response.aread()
            result = json.loads(raw_body.decode())

            choices = result.get("choices")
            if not choices:
                raise RuntimeError(f"Invalid response: 'choices' missing or empty in {result}")

            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
                ).strip()

            return TextGenerationOutput(
                transaction_hash="external",
                finish_reason=choices[0].get("finish_reason"),
                chat_output=message,
                tee_signature=result.get("tee_signature"),
                tee_timestamp=result.get("tee_timestamp"),
                **tee.metadata(),
            )

        try:
            return await self._call_with_tee_retry("chat", _request)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"TEE LLM chat failed: {e}") from e

    async def _chat_tools_as_stream(self, params: _ChatParams, messages: List[Dict]) -> AsyncGenerator[StreamChunk, None]:
        """Non-streaming fallback for tool-call requests wrapped as a single StreamChunk."""
        result = await self._chat_request(params, messages)
        chat_output = result.chat_output or {}
        yield StreamChunk(
            choices=[
                StreamChoice(
                    delta=StreamDelta(
                        role=chat_output.get("role"),
                        content=chat_output.get("content"),
                        tool_calls=chat_output.get("tool_calls"),
                    ),
                    index=0,
                    finish_reason=result.finish_reason,
                )
            ],
            model=params.model,
            is_final=True,
            tee_signature=result.tee_signature,
            tee_timestamp=result.tee_timestamp,
            tee_id=result.tee_id,
            tee_endpoint=result.tee_endpoint,
            tee_payment_address=result.tee_payment_address,
        )

    async def _chat_stream(self, params: _ChatParams, messages: List[Dict]) -> AsyncGenerator[StreamChunk, None]:
        """Async SSE streaming implementation."""
        self._tee.ensure_refresh_loop()
        headers = self._headers(params.x402_settlement_mode)
        payload = self._chat_payload(params, messages, stream=True)

        chunks_yielded = False
        try:
            tee = self._tee.get()
            async with tee.http_client.stream(
                "POST",
                tee.endpoint + _CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            ) as response:
                async for chunk in self._parse_sse_response(response, tee):
                    chunks_yielded = True
                    yield chunk
            return
        except httpx.HTTPStatusError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if chunks_yielded:
                raise
            logger.warning(
                "Connection failure during stream setup; refreshing TEE and retrying once: %s",
                exc,
            )

        # Only reached if the first attempt failed before yielding any chunks.
        # Re-resolve the TEE endpoint from the registry and retry once.
        await self._tee.reconnect()
        tee = self._tee.get()

        headers = self._headers(params.x402_settlement_mode)
        async with tee.http_client.stream(
            "POST",
            tee.endpoint + _CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        ) as response:
            async for chunk in self._parse_sse_response(response, tee):
                yield chunk

    async def _parse_sse_response(self, response, tee) -> AsyncGenerator[StreamChunk, None]:
        """Parse an SSE response stream into StreamChunk objects."""
        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code >= 400:
            body = await response.aread()
            raise RuntimeError(f"TEE LLM streaming request failed with status {status_code}: {body.decode('utf-8', errors='replace')}")

        buffer = b""
        async for raw_chunk in response.aiter_raw():
            if not raw_chunk:
                continue

            buffer += raw_chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.strip()
                if not line:
                    continue

                try:
                    decoded = line.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunk = StreamChunk.from_sse_data(data)
                if chunk.is_final:
                    chunk.tee_id = tee.tee_id
                    chunk.tee_endpoint = tee.endpoint
                    chunk.tee_payment_address = tee.payment_address
                yield chunk
