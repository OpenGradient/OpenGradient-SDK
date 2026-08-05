"""High-level confidential (Oblivious HTTP) LLM inference.

This is the one-call entry point for verified, private chat completions over
Oblivious HTTP — the same path the OpenGradient chat app uses in the browser
(``lib/api/ohttp.ts``). It ties together the pieces an integrator would
otherwise have to wire up by hand:

  1. `opengradient.client.tee_registry` — discover an OHTTP-capable TEE
     (endpoint, HPKE key, signing key) from the on-chain registry.
  2. `opengradient.client.tee_ohttp_client` — HPKE-encrypt the request, POST the
     ciphertext to the relay's confidential-inference path
     (``/api/v1/chat/ohttp``), then decrypt and verify the response.

Unlike `opengradient.client.llm.LLM`, this client needs **no wallet on the
caller's side**: the request travels end-to-end encrypted to the enclave, and
the untrusted relay (which holds the x402 account and pays per request) only
ever sees ciphertext. Authentication to the relay is left to the caller via an
``auth_headers`` provider, so the client works against any relay deployment
without baking in a credential scheme.

Every response is signature-verified inside the client before any content is
returned — see `opengradient.client.tee_verify` for the trust chain.

Usage:
    ```python
    import opengradient as og

    # Resolve an OHTTP-capable TEE from the on-chain registry and target the
    # relay's confidential-inference path automatically.
    client = og.ConfidentialLLM(
        relay_url="https://chat-api.opengradient.ai",
        auth_headers=lambda: {"Authorization": "Bearer <token>"},
    )

    result = client.chat(
        model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=200,
    )
    print(result.content)          # verified assistant text
    print(result.proof.tee_id)     # attested enclave identity
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import requests

from .tee_ohttp_client import AuthHeaderProvider, OhttpRelayClient, VerifiedChatResponse
from .tee_registry import TEEEndpoint, TEERegistry

if TYPE_CHECKING:
    from ..types import TEE_LLM

# Registry defaults, kept in sync with `opengradient.client.llm`. Duplicated
# (rather than imported) so the confidential path carries no x402/wallet
# dependency — the relay holds the account, the caller needs nothing on-chain.
DEFAULT_RPC_URL = "https://ogevmdevnet.opengradient.ai"
DEFAULT_TEE_REGISTRY_ADDRESS = "0x703cB174AEadB35D611858369B4b1111dC9Abda6"

# The relay's confidential-inference path. Mirrors the chat app's
# ``OHTTP_ENDPOINT`` constant (``lib/api/ohttp.ts``): encapsulated OHTTP requests
# are POSTed here.
OHTTP_CHAT_ENDPOINT = "/api/v1/chat/ohttp"


class ConfidentialLLM:
    """Verified, private chat completions over Oblivious HTTP through a relay.

    Discovers an OHTTP-capable TEE from the on-chain registry, POSTs
    HPKE-encrypted requests to the relay's confidential-inference path
    (``/api/v1/chat/ohttp``), and verifies the enclave's signature before
    returning any content — matching the browser chat app's OHTTP flow.

    Args:
        relay_url: The relay base URL (e.g. ``"https://chat-api.opengradient.ai"``)
            or the full confidential-inference URL. The
            ``/api/v1/chat/ohttp`` path is appended automatically when the given
            URL does not already end with it.
        rpc_url: RPC endpoint for the chain the TEE registry is deployed on.
            Defaults to the OpenGradient devnet.
        registry_address: Address of the deployed ``TEERegistry`` contract.
        auth_headers: Optional callable returning headers to authenticate to the
            relay (called per request so tokens can be refreshed), e.g.
            ``lambda: {"Authorization": "Bearer <token>"}``.
        session: Optional ``requests.Session`` to reuse connections.
        timeout: Per-request timeout in seconds.

    Raises:
        RuntimeError: If the registry has no active OHTTP-capable LLM TEE.
    """

    def __init__(
        self,
        relay_url: str,
        *,
        rpc_url: str = DEFAULT_RPC_URL,
        registry_address: str = DEFAULT_TEE_REGISTRY_ADDRESS,
        auth_headers: Optional[AuthHeaderProvider] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 120.0,
    ):
        registry = TEERegistry(rpc_url=rpc_url, registry_address=registry_address)
        tee = registry.get_llm_tee_ohttp_config()
        if tee is None:
            raise RuntimeError(
                f"No active OHTTP-capable LLM TEE found in the registry (rpc_url={rpc_url}, registry_address={registry_address})."
            )
        self._init(relay_url, tee, auth_headers=auth_headers, session=session, timeout=timeout)

    @classmethod
    def from_tee(
        cls,
        relay_url: str,
        tee: TEEEndpoint,
        *,
        auth_headers: Optional[AuthHeaderProvider] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 120.0,
    ) -> "ConfidentialLLM":
        """Create a client for a TEE you have already resolved.

        Use this to skip the registry lookup — for a self-hosted TEE, a pinned
        enclave, or when you have already selected a `TEEEndpoint` (which must
        carry an ``ohttp_config`` and ``signing_public_key_der``).

        Args:
            relay_url: The relay base URL or full confidential-inference URL (the
                ``/api/v1/chat/ohttp`` path is appended when missing).
            tee: The `opengradient.client.tee_registry.TEEEndpoint` to encrypt to.
            auth_headers: Optional per-request relay auth header provider.
            session: Optional ``requests.Session`` to reuse connections.
            timeout: Per-request timeout in seconds.
        """
        instance = cls.__new__(cls)
        instance._init(relay_url, tee, auth_headers=auth_headers, session=session, timeout=timeout)
        return instance

    def _init(
        self,
        relay_url: str,
        tee: TEEEndpoint,
        *,
        auth_headers: Optional[AuthHeaderProvider],
        session: Optional[requests.Session],
        timeout: float,
    ) -> None:
        self._tee = tee
        self._relay_url = _confidential_inference_url(relay_url)
        self._relay = OhttpRelayClient(
            self._relay_url,
            tee,
            auth_headers=auth_headers,
            session=session,
            timeout=timeout,
        )

    @property
    def tee(self) -> TEEEndpoint:
        """The resolved TEE this client encrypts requests to."""
        return self._tee

    @property
    def relay_url(self) -> str:
        """The full confidential-inference URL requests are POSTed to."""
        return self._relay_url

    # ── Public API ──────────────────────────────────────────────────────

    def chat(
        self,
        model: Union["TEE_LLM", str],
        messages: List[Dict],
        max_tokens: int = 100,
        stop_sequence: Optional[List[str]] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        web_search: bool = False,
        stream: bool = False,
    ) -> VerifiedChatResponse:
        """Send a verified, private chat completion over Oblivious HTTP.

        Builds an OpenAI ``/v1/chat/completions`` request from the arguments (the
        same shape the chat app builds in ``buildInnerChatRequest``), encrypts
        it to the resolved TEE, and returns the signature-verified response.

        Args:
            model: The model to use, e.g. ``og.TEE_LLM.CLAUDE_HAIKU_4_5``. A
                ``"provider/model"`` value is reduced to the gateway model id
                (the part after the ``/``), mirroring the chat app.
            messages: OpenAI-style chat messages.
            max_tokens: Maximum output tokens. Default 100.
            stop_sequence: Optional list of stop sequences.
            temperature: Sampling temperature (0-1). Default 0.0.
            tools: Optional function-calling tool definitions.
            tool_choice: Optional tool-choice directive (forwarded on the wire but
                not part of the signed request hash).
            response_format: Optional OpenAI ``response_format`` dict.
            web_search: Enable the provider's native web search. Default False.
            stream: When True, the encrypted stream is fully buffered and verified
                before returning; ``result.stream_frames`` holds the decrypted SSE
                frames. Default False.

        Returns:
            A `opengradient.client.tee_ohttp_client.VerifiedChatResponse`.

        Raises:
            opengradient.client.tee_ohttp_client.RelayError: If the relay or inner request errored.
            opengradient.client.tee_verify.VerificationError: If verification failed.
        """
        body = self._build_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stop_sequence=stop_sequence,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            web_search=web_search,
        )
        if stream:
            return self._relay.stream_chat_completion(body)
        return self._relay.chat_completion(body)

    def chat_completion(self, body: Dict) -> VerifiedChatResponse:
        """Send a raw OpenAI chat-completions ``body`` (non-streaming) and verify it.

        Escape hatch for callers that build the request body themselves. See
        `chat` for the argument-driven convenience.
        """
        return self._relay.chat_completion(body)

    def stream_chat_completion(self, body: Dict) -> VerifiedChatResponse:
        """Send a raw OpenAI chat-completions ``body`` as a stream and verify it.

        The encrypted stream is fully buffered and verified before returning, so
        ``result.stream_frames`` is safe to replay to an end user.
        """
        return self._relay.stream_chat_completion(body)

    @staticmethod
    def _build_body(
        *,
        model: Union["TEE_LLM", str],
        messages: List[Dict],
        max_tokens: int,
        stop_sequence: Optional[List[str]],
        temperature: float,
        tools: Optional[List[Dict]],
        tool_choice: Optional[str],
        response_format: Optional[Dict],
        web_search: bool,
    ) -> Dict:
        body: Dict = {
            "model": _gateway_model(str(model)),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop_sequence:
            body["stop"] = stop_sequence
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        if response_format:
            body["response_format"] = response_format
        if web_search:
            body["web_search"] = True
        return body


def _gateway_model(model: str) -> str:
    """Reduce a ``"provider/model"`` id to the gateway model name.

    Mirrors the chat app's ``modelToGatewayModel`` (``lib/api/ohttp.ts``): the
    gateway expects the model id without the provider prefix. Plain ids without a
    ``/`` pass through unchanged.
    """
    return model.split("/", 1)[1] if "/" in model else model


def _confidential_inference_url(relay_url: str) -> str:
    """Join the relay base URL with the confidential-inference path.

    A URL that already targets ``/api/v1/chat/ohttp`` is returned unchanged, so
    callers can pass either the relay base or the full endpoint.
    """
    trimmed = relay_url.rstrip("/")
    if trimmed.endswith(OHTTP_CHAT_ENDPOINT):
        return trimmed
    return trimmed + OHTTP_CHAT_ENDPOINT
