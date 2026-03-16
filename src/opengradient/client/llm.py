"""LLM chat and completion via TEE-verified execution with x402 payments."""

import json
import logging
import ssl
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Union

from eth_account import Account
from eth_account.account import LocalAccount
from x402v2 import x402Client as x402Clientv2
from x402v2.http.clients import x402HttpxClient as x402HttpxClientv2
from x402v2.mechanisms.evm import EthAccountSigner as EthAccountSignerv2
from x402v2.mechanisms.evm.exact.register import register_exact_evm_client as register_exact_evm_clientv2
from x402v2.mechanisms.evm.upto.register import register_upto_evm_client as register_upto_evm_clientv2

from ..types import TEE_LLM, StreamChoice, StreamChunk, StreamDelta, TextGenerationOutput, x402SettlementMode
from .opg_token import Permit2ApprovalResult, ensure_opg_approval
from .tee_registry import TEERegistry, build_ssl_context_from_der

logger = logging.getLogger(__name__)

DEFAULT_RPC_URL = "https://ogevmdevnet.opengradient.ai"
DEFAULT_TEE_REGISTRY_ADDRESS = "0x4e72238852f3c918f4E4e57AeC9280dDB0c80248"

X402_PROCESSING_HASH_HEADER = "x-processing-hash"
X402_PLACEHOLDER_API_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
BASE_TESTNET_NETWORK = "eip155:84532"

_CHAT_ENDPOINT = "/v1/chat/completions"
_COMPLETION_ENDPOINT = "/v1/completions"
_REQUEST_TIMEOUT = 60


@dataclass
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
    OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.
    This only sends an on-chain transaction when the current allowance is
    below the requested amount.

    Usage:
        llm = og.LLM(private_key="0x...")

        # One-time approval (idempotent — skips if allowance is already sufficient)
        llm.ensure_opg_approval(opg_amount=5)

        result = await llm.chat(model=TEE_LLM.CLAUDE_HAIKU_4_5, messages=[...])
        result = await llm.completion(model=TEE_LLM.CLAUDE_HAIKU_4_5, prompt="Hello")

    Args:
        private_key (str): Ethereum private key for signing x402 payments.
        rpc_url (str): RPC URL for the OpenGradient network. Used to fetch the
            active TEE endpoint from the on-chain registry when ``llm_server_url``
            is not provided.
        tee_registry_address (str): Address of the on-chain TEE registry contract.
        llm_server_url (str, optional): Bypass the registry and connect directly
            to this TEE endpoint URL (e.g. ``"https://1.2.3.4"``). When set,
            TLS certificate verification is disabled automatically because
            self-hosted TEE servers typically use self-signed certificates.

            .. warning::
                Using ``llm_server_url`` disables TLS certificate verification,
                which removes protection against man-in-the-middle attacks.
                Only connect to servers you trust and over secure network paths.
    """

    def __init__(
        self,
        private_key: str,
        rpc_url: str = DEFAULT_RPC_URL,
        tee_registry_address: str = DEFAULT_TEE_REGISTRY_ADDRESS,
        llm_server_url: Optional[str] = None,
    ):
        self._wallet_account: LocalAccount = Account.from_key(private_key)

        endpoint, tls_cert_der, tee_id, tee_payment_address = self._resolve_tee(
            llm_server_url,
            rpc_url,
            tee_registry_address,
        )

        self._tee_id = tee_id
        self._tee_endpoint = endpoint
        self._tee_payment_address = tee_payment_address

        ssl_ctx = build_ssl_context_from_der(tls_cert_der) if tls_cert_der else None
        # When connecting directly via llm_server_url, skip cert verification —
        # self-hosted TEE servers commonly use self-signed certificates.
        verify_ssl = llm_server_url is None
        self._tls_verify: Union[ssl.SSLContext, bool] = ssl_ctx if ssl_ctx else verify_ssl

        # x402 client and signer
        signer = EthAccountSignerv2(self._wallet_account)
        self._x402_client = x402Clientv2()
        register_exact_evm_clientv2(self._x402_client, signer, networks=[BASE_TESTNET_NETWORK])
        register_upto_evm_clientv2(self._x402_client, signer, networks=[BASE_TESTNET_NETWORK])
        # httpx.AsyncClient subclass - construction is sync, connections open lazily
        self._http_client = x402HttpxClientv2(self._x402_client, verify=self._tls_verify)

    # ── TEE resolution ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_tee(
        tee_endpoint_override: Optional[str],
        og_rpc_url: Optional[str],
        tee_registry_address: Optional[str],
    ) -> tuple:
        """Resolve TEE endpoint and metadata from the on-chain registry or explicit URL.

        Returns:
            (endpoint, tls_cert_der, tee_id, payment_address)
        """
        if tee_endpoint_override is not None:
            return tee_endpoint_override, None, None, None

        if og_rpc_url is None or tee_registry_address is None:
            raise ValueError("Either llm_server_url or both rpc_url and tee_registry_address must be provided.")

        try:
            registry = TEERegistry(rpc_url=og_rpc_url, registry_address=tee_registry_address)
            tee = registry.get_llm_tee()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch LLM TEE endpoint from registry ({tee_registry_address} on {og_rpc_url}): {e}. ") from e

        if tee is None:
            raise ValueError("No active LLM proxy TEE found in the registry. Pass llm_server_url explicitly to override.")

        logger.info("Using TEE endpoint from registry: %s (teeId=%s)", tee.endpoint, tee.tee_id)
        return tee.endpoint, tee.tls_cert_der, tee.tee_id, tee.payment_address

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http_client.aclose()

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

    def _tee_metadata(self) -> Dict:
        return dict(
            tee_id=self._tee_id,
            tee_endpoint=self._tee_endpoint,
            tee_payment_address=self._tee_payment_address,
        )

    # ── Public API ──────────────────────────────────────────────────────

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
            RuntimeError: If the approval transaction fails.
        """
        if opg_amount < 0.05:
            raise ValueError("OPG amount must be at least 0.05.")
        return ensure_opg_approval(self._wallet_account, opg_amount)

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
        headers = self._headers(x402_settlement_mode)
        payload: Dict = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop_sequence:
            payload["stop"] = stop_sequence

        try:
            response = await self._http_client.post(
                self._tee_endpoint + _COMPLETION_ENDPOINT,
                json=payload,
                headers=headers,
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
                **self._tee_metadata(),
            )
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
        headers = self._headers(params.x402_settlement_mode)
        payload = self._chat_payload(params, messages)

        try:
            response = await self._http_client.post(
                self._tee_endpoint + _CHAT_ENDPOINT,
                json=payload,
                headers=headers,
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
                **self._tee_metadata(),
            )
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
        headers = self._headers(params.x402_settlement_mode)
        payload = self._chat_payload(params, messages, stream=True)

        async with self._http_client.stream(
            "POST",
            self._tee_endpoint + _CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        ) as response:
            async for chunk in self._parse_sse_response(response):
                yield chunk

    async def _parse_sse_response(self, response) -> AsyncGenerator[StreamChunk, None]:
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
                    chunk.tee_id = self._tee_id
                    chunk.tee_endpoint = self._tee_endpoint
                    chunk.tee_payment_address = self._tee_payment_address
                yield chunk
