"""LLM chat and completion via TEE-verified execution with x402 payments."""

import base64
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
import httpx
import asyncio
import os

from eth_account import Account
from eth_account.account import LocalAccount
from x402 import x402Client
from x402.http import decode_payment_response_header
from x402.mechanisms.evm import EthAccountSignerWithRPC
from x402.mechanisms.evm.batch_settlement.client import (
    BatchSettlementEvmScheme,
    BatchSettlementEvmSchemeOptions,
    FileChannelStorageOptions,
    FileClientChannelStorage,
    process_settle_response,
)
from x402.mechanisms.evm.exact.register import register_exact_evm_client

from ..types import TEE_LLM, ResponseFormat, StreamChoice, StreamChunk, StreamDelta, TextGenerationOutput, x402SettlementMode
from .opg_token import Permit2ApprovalResult, ensure_opg_approval
from .tee_connection import (
    ActiveTEE,
    RegistryTEEConnection,
    StaticTEEConnection,
    TEEConnectionInterface,
)
from .tee_registry import TEERegistry

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_RPC_URL = "https://ogevmdevnet.opengradient.ai"
DEFAULT_TEE_REGISTRY_ADDRESS = "0x703cB174AEadB35D611858369B4b1111dC9Abda6"

X402_PROCESSING_HASH_HEADER = "x-processing-hash"
X402_DATA_SETTLEMENT_TX_HASH_HEADER = "x-settlement-tx-hash"
X402_DATA_SETTLEMENT_BLOB_ID_HEADER = "x-settlement-walrus-blob-id"
X402_PLACEHOLDER_API_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
BASE_MAINNET_NETWORK = "eip155:8453"
BASE_MAINNET_RPC = os.getenv("BASE_MAINNET_RPC", "https://base-rpc.publicnode.com")
DEFAULT_BATCH_CHANNEL_STORAGE_DIR = Path(
    os.getenv("X402_BATCH_CHANNEL_STORAGE_DIR", "~/.opengradient/x402-batch-channels")
).expanduser()

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
    response_format: Optional[ResponseFormat]
    x402_settlement_mode: x402SettlementMode
    web_search: bool = False


class LLM:
    """
    LLM inference namespace.

    Provides access to large language model completions and chat via TEE
    (Trusted Execution Environment) with x402 payment protocol support.
    Supports both streaming and non-streaming responses.

    All request methods (``chat``, ``completion``) are async.

    Before making LLM requests, ensure your wallet has approved sufficient
    OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.

    Usage:
        # Via on-chain registry (default)
        llm = og.LLM(private_key="0x...")

        # Via hardcoded URL (development / self-hosted)
        llm = og.LLM.from_url(private_key="0x...", llm_server_url="https://1.2.3.4")

        # Ensure sufficient OPG allowance (only sends tx when below threshold)
        llm.ensure_opg_approval(min_allowance=5)

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

        x402_client, self._batch_channel_storage = LLM._build_x402_client(
            private_key,
            rpc_url=BASE_MAINNET_RPC,
        )
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
        x402_client, instance._batch_channel_storage = cls._build_x402_client(private_key)
        instance._tee = StaticTEEConnection(x402_client=x402_client, endpoint=llm_server_url)
        return instance

    @staticmethod
    def _build_x402_client(
        private_key: str,
        rpc_url: str = BASE_MAINNET_RPC,
    ) -> tuple[x402Client, FileClientChannelStorage]:
        """Build the x402 payment stack and persistent batch channel state."""
        account = Account.from_key(private_key)
        signer = EthAccountSignerWithRPC(account, rpc_url=rpc_url)
        channel_storage = FileClientChannelStorage(
            FileChannelStorageOptions(
                directory=DEFAULT_BATCH_CHANNEL_STORAGE_DIR / account.address.lower(),
            )
        )
        client = x402Client()
        register_exact_evm_client(client, signer, networks=[BASE_MAINNET_NETWORK])
        client.register(
            BASE_MAINNET_NETWORK,
            BatchSettlementEvmScheme(
                signer,
                BatchSettlementEvmSchemeOptions(
                    storage=channel_storage,
                    rpc_url=rpc_url,
                ),
            ),
        )
        return client, channel_storage

    def process_batch_payment_response(self, payment_response: str) -> None:
        """Commit a batch voucher settlement response delivered outside HTTP headers.

        Normal x402 HTTP requests are handled by ``x402HttpxClient``. Direct
        SSE and OHTTP streaming cannot expose ``PAYMENT-RESPONSE`` as a normal
        response header, so their trusted relay path calls this method with the
        encoded header value instead.
        """
        settle_response = decode_payment_response_header(payment_response)
        if not settle_response.success:
            raise RuntimeError(
                "Batch payment settlement failed: "
                f"{settle_response.error_reason or 'unknown error'}"
            )
        process_settle_response(self._batch_channel_storage, settle_response)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel the background refresh loop and close the HTTP client."""
        await self._tee.close()

    def resolve_tee_connection(self, tee_id: Optional[str] = None) -> ActiveTEE:
        """Resolve the current TEE or a specific active registry TEE.

        This is primarily for backend relays that need SDK-managed TEE routing,
        TLS pinning, and x402 clients without using the chat/completion helpers
        directly, for example when forwarding OHTTP ciphertext.

        Warning:
            Resolving a TEE id other than the active one scans the on-chain
            registry with a blocking web3 call. Async servers resolving a TEE
            per request should use ``aresolve_tee_connection`` instead.
        """
        return self._tee.resolve(tee_id)

    async def aresolve_tee_connection(self, tee_id: Optional[str] = None) -> ActiveTEE:
        """Async, event-loop-safe variant of ``resolve_tee_connection``.

        Built for backend relays that resolve a TEE on every request: registry
        scans run in a worker thread and each pinned id's outcome (found or
        not-active) is cached briefly, so this never blocks the event loop and
        doesn't hit the chain RPC per request. Also starts the background TEE
        refresh loop, so the active TEE fails over when it is retired from the
        registry.
        """
        return await self._tee.aresolve(tee_id)

    def ensure_tee_refresh_loop(self) -> None:
        """Start the background TEE health-check/failover loop if not running.

        The loop starts lazily from the SDK's own request helpers and from
        ``aresolve_tee_connection``. Call this explicitly from server startup
        when neither is used on every code path and you still want the active
        TEE to fail over once it is retired from the registry. Requires a
        running event loop. No-op for static/dev connections.
        """
        self._tee.ensure_refresh_loop()

    # ── Image helpers ────────────────────────────────────────────────────

    @staticmethod
    async def _resolve_images(images: Optional[List[str]]) -> Optional[List[str]]:
        """Fetch any HTTP/HTTPS image URLs and convert them to data: URIs.

        Providers like ByteDance Seedance return pre-signed CDN URLs instead of
        inline base64. This normalises the list so callers always receive data: URIs.
        """
        if not images:
            return images
        resolved: List[str] = []
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            for img in images:
                if img.startswith("http://") or img.startswith("https://"):
                    resp = await client.get(img)
                    resp.raise_for_status()
                    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    b64 = base64.b64encode(resp.content).decode("ascii")
                    resolved.append(f"data:{mime};base64,{b64}")
                else:
                    resolved.append(img)
        return resolved

    # ── Request helpers ─────────────────────────────────────────────────

    def _headers(self, settlement_mode: x402SettlementMode) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {X402_PLACEHOLDER_API_KEY}",
            "X-SETTLEMENT-TYPE": settlement_mode.value,
        }

    @staticmethod
    def _data_settlement_transaction_hash(response: httpx.Response) -> Optional[str]:
        headers = getattr(response, "headers", {}) or {}
        value: Optional[str] = headers.get(X402_DATA_SETTLEMENT_TX_HASH_HEADER)
        return value

    @staticmethod
    def _data_settlement_blob_id(response: httpx.Response) -> Optional[str]:
        headers = getattr(response, "headers", {}) or {}
        value: Optional[str] = headers.get(X402_DATA_SETTLEMENT_BLOB_ID_HEADER)
        return value

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
        if params.response_format:
            payload["response_format"] = params.response_format.to_dict()
        if params.web_search:
            payload["web_search"] = True
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

    def ensure_opg_approval(
        self,
        min_allowance: float,
        approve_amount: Optional[float] = None,
    ) -> Permit2ApprovalResult:
        """Ensure the Permit2 allowance stays above a minimum threshold.

        Only sends a transaction when the current allowance drops below
        ``min_allowance``. When approval is needed, approves ``approve_amount``
        (defaults to ``2 * min_allowance``) to create a buffer that survives
        multiple service restarts without re-approving.

        Best for backend servers that call this on startup::

            llm.ensure_opg_approval(min_allowance=5.0, approve_amount=100.0)

        Args:
            min_allowance: The minimum acceptable allowance in OPG. Must be
                at least 0.1 OPG.
            approve_amount: The amount of OPG to approve when a transaction
                is needed. Defaults to ``2 * min_allowance``. Must be
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
        return ensure_opg_approval(self._wallet_account, min_allowance, approve_amount)

    async def completion(
        self,
        model: TEE_LLM,
        prompt: str,
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        web_search: bool = False,
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
            web_search (bool, optional): Enable the provider's native web search. When True,
                the model can search the web to answer the prompt; each search is billed per
                search on top of token usage at the provider's list price. Supported by OpenAI,
                Anthropic, Google, and xAI models; other providers ignore the flag. Default is False.
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
        if web_search:
            payload["web_search"] = True

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
                data_settlement_transaction_hash=self._data_settlement_transaction_hash(response),
                data_settlement_blob_id=self._data_settlement_blob_id(response),
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
        response_format: Optional[ResponseFormat] = None,
        web_search: bool = False,
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
            response_format (ResponseFormat, optional): Enforce a specific output format.
                Use ``ResponseFormat(type="json_object")`` for any valid JSON (not supported
                by Anthropic models). Use ``ResponseFormat(type="json_schema", json_schema={...})``
                to enforce a strict schema (supported by all providers including Anthropic).
                Defaults to None (plain text).
            web_search (bool, optional): Enable the provider's native web search. When True,
                the model can search the web while answering; each search is billed per search
                on top of token usage at the provider's list price. Supported by OpenAI,
                Anthropic, Google, and xAI models; other providers ignore the flag. Default is False.
            x402_settlement_mode (x402SettlementMode, optional): Settlement mode for x402 payments.
                - PRIVATE: Payment only, no input/output data on-chain (most privacy-preserving).
                - BATCH_HASHED: Aggregates inferences into a Merkle tree with input/output hashes and signatures (default, most cost-efficient).
                - INDIVIDUAL_FULL: Records input, output, timestamp, and verification on-chain (maximum auditability).
                Defaults to BATCH_HASHED.
            stream (bool, optional): Whether to stream the response. Default is False.

        Returns:
            Union[TextGenerationOutput, AsyncGenerator[StreamChunk, None]]:
                - If stream=False: TextGenerationOutput with chat_output, data settlement metadata, finish_reason, and payment_hash.
                  Image-output models (e.g. ``TEE_LLM.GEMINI_3_1_FLASH_IMAGE``) populate ``images`` with the generated images as ``data:`` URIs.
                - If stream=True: Async generator yielding StreamChunk objects. The final chunk carries any generated ``images``.

        Raises:
            ValueError: If ``response_format="json_object"`` is used with an Anthropic model.
            RuntimeError: If the inference fails.
        """
        if response_format is not None and response_format.type == "json_object":
            provider = model.split("/")[0]
            if provider == "anthropic":
                raise ValueError(
                    "Anthropic models do not support response_format type 'json_object'. "
                    "Use ResponseFormat(type='json_schema', json_schema={...}) with an explicit schema instead."
                )

        params = _ChatParams(
            model=model.split("/")[1],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequence=stop_sequence,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            x402_settlement_mode=x402_settlement_mode,
            web_search=web_search,
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
            raw_body = await response.aread()
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    _format_http_error(response, raw_body),
                    request=response.request,
                    response=response,
                )
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
                data_settlement_transaction_hash=self._data_settlement_transaction_hash(response),
                data_settlement_blob_id=self._data_settlement_blob_id(response),
                finish_reason=choices[0].get("finish_reason"),
                chat_output=message,
                images=await self._resolve_images(message.get("images")),
                usage=result.get("usage"),
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
            data_settlement_transaction_hash=result.data_settlement_transaction_hash,
            data_settlement_blob_id=result.data_settlement_blob_id,
            images=await self._resolve_images(result.images),
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
            request = getattr(response, "request", None)
            if request is None:
                request = httpx.Request("POST", str(response.url))
            raise httpx.HTTPStatusError(
                _format_http_error(response, body),
                request=request,
                response=response,
            )

        buffer = b""
        pending_final_chunk: Optional[StreamChunk] = None
        event_name: Optional[str] = None
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

                if decoded.startswith("event:"):
                    event_name = decoded[len("event:") :].strip()
                    continue

                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    if pending_final_chunk is not None:
                        yield pending_final_chunk
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    event_name = None
                    continue

                if event_name == "x402-settlement":
                    event_name = None
                    if not isinstance(data, dict) or not data.get("success"):
                        raise RuntimeError(
                            "TEE batch settlement failed: "
                            f"{data.get('error', 'invalid settlement event') if isinstance(data, dict) else 'invalid settlement event'}"
                        )
                    payment_response = data.get("paymentResponse")
                    if not isinstance(payment_response, str) or not payment_response:
                        raise RuntimeError("TEE batch settlement response is missing paymentResponse")
                    self.process_batch_payment_response(payment_response)
                    continue

                event_name = None

                chunk = StreamChunk.from_sse_data(data)
                if chunk.is_final:
                    chunk.data_settlement_transaction_hash = (
                        chunk.data_settlement_transaction_hash or self._data_settlement_transaction_hash(response)
                    )
                    chunk.data_settlement_blob_id = chunk.data_settlement_blob_id or self._data_settlement_blob_id(response)
                    chunk.tee_id = tee.tee_id
                    chunk.tee_endpoint = tee.endpoint
                    chunk.tee_payment_address = tee.payment_address
                    pending_final_chunk = chunk
                    continue
                yield chunk

        if pending_final_chunk is not None:
            yield pending_final_chunk


def _decode_payment_required(header_value: Optional[str]) -> str:
    """Decode the base64-encoded JSON in the `payment-required` response header."""
    if not header_value:
        return "<missing>"
    try:
        decoded = base64.b64decode(header_value).decode("utf-8")
        return json.dumps(json.loads(decoded), indent=2)
    except Exception:
        return header_value


def _format_http_error(response: httpx.Response, body: bytes) -> str:
    """Build an error message that surfaces the x402 payment-required details."""
    return (
        f"HTTP {response.status_code} from {response.url}\n"
        f"Payment-Required: {_decode_payment_required(response.headers.get('payment-required'))}\n"
        f"Body: {body.decode(errors='replace')}"
    )
