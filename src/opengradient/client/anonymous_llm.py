"""
Anonymous LLM inference using Oblivious HTTP-style encapsulation.

Hides the prompt from any party other than the attested TEE enclave AND
hides the client's IP address from the enclave. The client never opens a
TLS connection to the enclave directly — requests transit an independent
relay that sees only ciphertext + client IP, while the enclave sees only
plaintext + relay IP. Unlinkability holds unless the relay and the enclave
collude.

Pairs with the tee-gateway's ``/v1/ohttp`` endpoint. The HPKE ciphersuite
is fixed at DHKEM(X25519, HKDF-SHA256) / HKDF-SHA256 / ChaCha20-Poly1305 —
matching what the gateway publishes via ``/v1/ohttp/config`` and commits to
inside the Nitro attestation document.

Known gaps in this v1:
  * No streaming (the endpoint refuses ``stream=true``).
  * No payment integration yet — assumes the gateway has anonymous
    inference enabled without a payment gate, or that a blind-token layer
    is added on top (see ``anonymous_inference_privacy`` design notes).
  * The relay must be operated by a different party than the enclave
    operator for unlinkability to be meaningful.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

logger = logging.getLogger(__name__)

_KEM_ID = 0x0020  # DHKEM(X25519, HKDF-SHA256)
_KDF_ID = 0x0001  # HKDF-SHA256
_AEAD_ID = 0x0003  # ChaCha20-Poly1305
_NK = 32
_NN = 12

_LABEL_REQUEST = b"message/bhttp request"
_LABEL_RESPONSE = b"message/bhttp response"

_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)


@dataclass(frozen=True)
class HpkeConfig:
    """HPKE key configuration fetched from the gateway."""

    key_id: int
    public_key: bytes  # raw X25519, 32 bytes

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "HpkeConfig":
        return cls(
            key_id=int(payload["key_id"]),
            public_key=bytes.fromhex(payload["public_key"]),
        )


@dataclass(frozen=True)
class AnonymousResult:
    """Decapsulated response from /v1/ohttp."""

    status: int
    body: Any  # JSON-decoded inner response from the chat handler


class AnonymousLLM:
    """Single-shot anonymous chat completion via an OHTTP relay.

    Usage::

        client = AnonymousLLM(
            relay_url="https://relay.example.com/ohttp",
            gateway_config_url="https://tee.opengradient.ai/v1/ohttp/config",
        )
        result = await client.chat(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
    """

    def __init__(
        self,
        relay_url: str,
        gateway_config_url: str,
        *,
        gateway_target: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Args:
            relay_url: HTTPS URL of the OHTTP relay. Must be operated by a
                different party than the enclave for unlinkability.
            gateway_config_url: URL where the gateway publishes its current
                HPKE key configuration (typically ``<gateway>/v1/ohttp/config``).
                The client fetches this lazily and caches it for the lifetime
                of the instance.
            gateway_target: Optional override for the relay's
                ``target-resource`` hint. Defaults to deriving the path-based
                target from ``gateway_config_url``.
            http_client: Inject an existing ``httpx.AsyncClient`` (useful for
                tests and connection-pool sharing). A fresh client is created
                if not supplied.
        """
        self._relay_url = relay_url.rstrip("/")
        self._config_url = gateway_config_url
        self._gateway_target = gateway_target or _derive_target(gateway_config_url)
        self._http = http_client or httpx.AsyncClient(timeout=60)
        self._owns_http = http_client is None
        self._config: HpkeConfig | None = None

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _ensure_config(self) -> HpkeConfig:
        if self._config is not None:
            return self._config
        resp = await self._http.get(self._config_url)
        resp.raise_for_status()
        self._config = HpkeConfig.from_json(resp.json())
        return self._config

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **extra: Any,
    ) -> AnonymousResult:
        """Send a chat completion request anonymously via the relay.

        ``extra`` is passed straight through to the inner OpenAI-compatible
        payload (``tools``, ``response_format``, etc.). Streaming is rejected
        by the gateway.
        """
        if extra.get("stream"):
            raise ValueError("stream=True is not supported over OHTTP")

        config = await self._ensure_config()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        inner_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        encapsulated, response_secret, enc = _encapsulate_request(config, inner_bytes)

        # Forward through the relay. The relay sees the ciphertext, the
        # target URL inside its own form (set per its own API), and our IP.
        # It MUST strip identifying headers before forwarding upstream — we
        # avoid setting any here beyond the bare minimum.
        relay_response = await self._http.post(
            self._relay_url,
            content=encapsulated,
            headers={
                "Content-Type": "message/ohttp-req",
                "OHTTP-Target": self._gateway_target,
            },
        )
        relay_response.raise_for_status()
        sealed = relay_response.content
        plaintext = _decapsulate_response(response_secret, enc, sealed)

        inner = json.loads(plaintext.decode("utf-8"))
        return AnonymousResult(status=int(inner.get("status", 200)), body=inner.get("body"))


def _derive_target(config_url: str) -> str:
    """Default ``OHTTP-Target`` header from the config URL by swapping the path."""
    # We assume the gateway endpoint is at /v1/ohttp on the same host as the
    # config endpoint. Callers can override via the gateway_target kwarg.
    if config_url.endswith("/v1/ohttp/config"):
        return config_url[: -len("/config")]
    return config_url


def _encapsulate_request(
    config: HpkeConfig, plaintext: bytes
) -> tuple[bytes, bytes, bytes]:
    """Encrypt ``plaintext`` to the gateway's HPKE key. Returns
    ``(wire_bytes, response_secret, enc)`` — the latter two are needed to
    decrypt the matching response.
    """
    hdr = (
        bytes([config.key_id])
        + struct.pack(">HHH", _KEM_ID, _KDF_ID, _AEAD_ID)
    )
    info = _LABEL_REQUEST + b"\x00" + hdr

    pkr = _SUITE.kem.deserialize_public_key(config.public_key)
    enc, sender = _SUITE.create_sender_context(pkr, info=info)
    ct = sender.seal(plaintext, aad=b"")

    response_secret = sender.export(_LABEL_RESPONSE, _NK)
    wire = hdr + enc + ct
    return wire, response_secret, enc


def _decapsulate_response(
    response_secret: bytes, enc: bytes, sealed: bytes
) -> bytes:
    if len(sealed) < max(_NN, _NK):
        raise ValueError("sealed response too short")
    response_nonce = sealed[: max(_NN, _NK)]
    aead_ct = sealed[max(_NN, _NK) :]

    salt = enc + response_nonce
    h = hmac.HMAC(salt, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()

    key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(prk)
    nonce = HKDFExpand(algorithm=hashes.SHA256(), length=_NN, info=b"nonce").derive(prk)
    return ChaCha20Poly1305(key).decrypt(nonce, aead_ct, b"")
