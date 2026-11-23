"""A fake TEE gateway and a fake chat-api relay, wired through httpx.MockTransport.

``FakeRelay`` plays both chat-api's ``/api/v1/developer/ohttp`` route and the
TEE gateway behind it: it records the outer request exactly as chat-api would
see it, decapsulates the sealed request with the gateway's HPKE key, and
answers the way the gateway does (single-shot ``message/ohttp-res``, or a
chunked ``message/ohttp-chunked-res`` SSE stream ending in a signed final
frame and ``[DONE]``), signed with the gateway's RSA key.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from _recipient import ChunkedResponseEncrypter, decapsulate_request, encapsulate_response, generate_keypair
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from eth_hash.auto import keccak

from opengradient_ohttp import TeeTarget
from opengradient_ohttp._verify import build_inner_request

API_KEY = "df_sk_test0000000000000000000000000000000000000000"
BASE_URL = "https://ohttp.invalid/v1"
RELAY_URL = "https://chat-api.memchat.io/api/v1/developer/ohttp"
TIMESTAMP = 1_763_000_000


class FakeGateway:
    """A TEE: an HPKE key pair, an RSA signing key, and the matching TeeTarget."""

    def __init__(self, endpoint: str = "https://gw.test"):
        self.hpke_private, self.hpke_public = generate_keypair()
        self.rsa_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = self.rsa_private.public_key()
        self.signing_key_der = public.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        pem = public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        self.tee_id = "0x" + keccak(self.signing_key_der).hex()
        self.endpoint = endpoint
        self.tee = TeeTarget(
            tee_id=self.tee_id,
            endpoint=endpoint,
            signing_key_pem=pem,
            ohttp_public_key=self.hpke_public,
        )

    def sign(self, canonical: dict, output: str) -> dict[str, Any]:
        """The tee_* fields the gateway puts on a response."""
        request_hash = keccak(json.dumps(canonical, sort_keys=True).encode("utf-8"))
        output_hash = keccak(output.encode("utf-8"))
        msg_hash = keccak(request_hash + output_hash + TIMESTAMP.to_bytes(32, "big"))
        signature = self.rsa_private.sign(
            msg_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        return {
            "tee_signature": base64.b64encode(signature).decode(),
            "tee_request_hash": request_hash.hex(),
            "tee_output_hash": output_hash.hex(),
            "tee_timestamp": TIMESTAMP,
            "tee_id": self.tee_id,
        }

    def registry_tuple(self, *, kem_id: int = 0x0020, hpke_public: Optional[bytes] = None) -> tuple:
        """This TEE as TEERegistry.getActiveTEEs returns it."""
        return (
            "0x" + "11" * 20,
            "0x" + "22" * 20,
            self.endpoint,
            self.signing_key_der,
            b"tls-cert",
            b"\x00" * 32,
            0,
            True,
            1,
            1,
            (0x01, kem_id, 0x0001, 0x0003, self.hpke_public if hpke_public is None else hpke_public, b"key-config", 1),
        )


@dataclass
class FakeRelay:
    """chat-api's developer relay plus the gateway behind it, as an httpx MockTransport handler.

    Knobs:
        reply: the assistant text.
        status / error_body: answer with this relay error instead of relaying.
        tamper: sign ``reply`` but send something else (verification must fail).
        plaintext_split: stream only; re-chunk the SSE plaintext every N bytes
            (splits lines and multi-byte characters across OHTTP chunks).
        wire_split: stream only; deliver the encrypted body N bytes at a time.
        drop_last_chunks: stream only; cut the encrypted stream short.
        in_band_error: stream only; fail after the first delta with ``data: {"error": ...}``.
    """

    gateway: FakeGateway
    reply: str = "Hello from the TEE, verified."
    status: int = 200
    error_body: Any = None
    error_headers: dict = field(default_factory=dict)
    tamper: bool = False
    plaintext_split: Optional[int] = None
    wire_split: Optional[int] = None
    drop_last_chunks: int = 0
    in_band_error: Optional[str] = None
    requests: list[httpx.Request] = field(default_factory=list)
    inner_requests: list[dict] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json=self.error_body, headers=self.error_headers)

        decapsulated = decapsulate_request(self.gateway.hpke_private, request.content)
        inner = json.loads(decapsulated.plaintext)
        self.inner_requests.append(inner)
        # The gateway hashes its own canonical form of what it received.
        _, canonical = build_inner_request({k: v for k, v in inner.items() if k != "stream"})
        sent = self.reply + "!" if self.tamper else self.reply

        if inner.get("stream"):
            return self._stream(decapsulated, inner, canonical, sent)

        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": TIMESTAMP,
            "model": inner["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": sent}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            **self.gateway.sign(canonical, self.reply),
        }
        sealed = encapsulate_response(decapsulated.response_key, decapsulated.enc, json.dumps(body).encode("utf-8"))
        return httpx.Response(200, content=sealed, headers={"content-type": "message/ohttp-res"})

    def _stream(self, decapsulated, inner: dict, canonical: dict, sent: str) -> httpx.Response:
        def chunk(delta: dict, finish_reason: Optional[str] = None, **extra: Any) -> str:
            frame = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": TIMESTAMP,
                "model": inner["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                **extra,
            }
            # ensure_ascii=False so multi-byte characters really hit the wire as UTF-8.
            return f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"

        events = [chunk({"content": piece}) for piece in re.findall(r"\S+\s*", sent)]
        if self.in_band_error is not None:
            events = events[:1] + [f"data: {json.dumps({'error': self.in_band_error})}\n\n"]
        else:
            events.append(chunk({}, "stop", **self.gateway.sign(canonical, self.reply)))
            events.append("data: [DONE]\n\n")

        if self.plaintext_split:
            plaintext = "".join(events).encode("utf-8")
            pieces = [plaintext[i : i + self.plaintext_split] for i in range(0, len(plaintext), self.plaintext_split)]
        else:
            pieces = [event.encode("utf-8") for event in events]

        encrypter = ChunkedResponseEncrypter(decapsulated.response_key_chunked, decapsulated.enc)
        sealed = [encrypter.encrypt_chunk(piece, is_final=(i == len(pieces) - 1)) for i, piece in enumerate(pieces)]
        if self.drop_last_chunks:
            sealed = sealed[: -self.drop_last_chunks]
        body = encrypter.header() + b"".join(sealed)

        headers = {"content-type": "message/ohttp-chunked-res"}
        if self.wire_split:
            parts = [body[i : i + self.wire_split] for i in range(0, len(body), self.wire_split)]
            return httpx.Response(200, content=iter(parts), headers=headers)
        return httpx.Response(200, content=body, headers=headers)
