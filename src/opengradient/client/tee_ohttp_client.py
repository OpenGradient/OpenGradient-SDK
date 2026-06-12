"""High-level Oblivious HTTP relay client for verified, private TEE inference.

This ties together the three lower-level pieces so an integrator doesn't have to:

  1. `opengradient.client.tee_registry` — discover a TEE (endpoint, OHTTP key,
     signing key) from the on-chain registry.
  2. `opengradient.client.tee_ohttp` — HPKE-encrypt the request and decrypt the
     response.
  3. `opengradient.client.tee_verify` — verify the enclave's RSA-PSS signature.

The relay (which holds the x402 wallet / account credentials and pays per
request) only ever sees ciphertext. Authentication to the relay is left to the
caller: pass an ``auth_headers`` provider returning whatever the relay expects
(e.g. ``{"Authorization": "Bearer <token>"}``), so this client works for any
relay deployment without baking in a credential scheme.

Verification happens **before** any content is returned. For streaming requests
the full encrypted stream is buffered, verified, and only then handed back as
decrypted SSE frames — so a caller can guarantee no unverified token ever
reaches the end user, at the cost of streaming latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import requests

from .tee_ohttp import ChunkedResponseDecrypter, encapsulate_request
from .tee_registry import TEEEndpoint
from .tee_verify import (
    TeeProof,
    VerificationError,
    build_inner_request,
    pem_from_der,
    response_content_for_hash,
    verify_response,
)

OHTTP_REQUEST_MEDIA_TYPE = "message/ohttp-req"
OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"
OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE = "message/ohttp-chunked-res"

AuthHeaderProvider = Callable[[], dict]


class RelayError(Exception):
    """The relay returned a non-success status, or the inner response was an error.

    Attributes:
        status_code: The HTTP (or inner) status code.
        message: A human-readable error message extracted from the response.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"relay error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass
class VerifiedChatResponse:
    """A TEE chat response that has passed signature verification.

    Attributes:
        body: The inner response JSON (the single-shot body, or the final SSE
            frame for a stream).
        content: The assistant text (or tool-calls JSON) that was verified.
        proof: The :class:`opengradient.client.tee_verify.TeeProof`.
        stream_frames: For streaming requests, the decrypted inner SSE ``data:``
            event strings (already verified), ready to replay to a client;
            ``None`` for single-shot requests.
    """

    body: dict
    content: str
    proof: TeeProof
    stream_frames: Optional[list[str]] = None


class OhttpRelayClient:
    """Send verified, private chat completions to a TEE through an OHTTP relay.

    Args:
        relay_url: Full URL to POST encapsulated requests to (e.g.
            ``https://chat-api.example.com/api/v1/chat/ohttp``).
        tee: The :class:`opengradient.client.tee_registry.TEEEndpoint` to encrypt
            to (must carry an ``ohttp_config`` and ``signing_public_key_der``).
        auth_headers: Optional callable returning headers to authenticate to the
            relay (called per request so tokens can be refreshed).
        session: Optional ``requests.Session`` to reuse connections.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        relay_url: str,
        tee: TEEEndpoint,
        *,
        auth_headers: Optional[AuthHeaderProvider] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 120.0,
    ):
        if tee.ohttp_config is None or len(tee.ohttp_config.public_key) != 32:
            raise ValueError("TEEEndpoint has no usable OHTTP config")
        if not tee.signing_public_key_der:
            raise ValueError("TEEEndpoint is missing a signing public key")
        self._relay_url = relay_url
        self._tee = tee
        self._ohttp_public_key = tee.ohttp_config.public_key
        # Honor the registry's advertised key/algorithm ids (key rotation) rather
        # than assuming the canonical defaults.
        self._enc_ids = {
            "key_id": tee.ohttp_config.key_id,
            "kem_id": tee.ohttp_config.kem_id,
            "kdf_id": tee.ohttp_config.kdf_id,
            "aead_id": tee.ohttp_config.aead_id,
        }
        self._auth_headers = auth_headers
        self._session = session or requests.Session()
        self._timeout = timeout
        self._signing_key_pem = pem_from_der(tee.signing_public_key_der)

    def chat_completion(self, body: dict) -> VerifiedChatResponse:
        """Send a non-streaming chat completion and return a verified response.

        Args:
            body: An OpenAI ``/v1/chat/completions`` request body.

        Returns:
            A :class:`VerifiedChatResponse`.

        Raises:
            RelayError: If the relay or the inner request errored.
            VerificationError: If the response signature could not be verified.
            opengradient.client.tee_verify.UnsupportedRequestError: If the body is invalid.
        """
        wire, canonical = build_inner_request(body)
        enc = encapsulate_request(self._ohttp_public_key, json.dumps(wire).encode("utf-8"), **self._enc_ids)

        resp = self._session.post(
            self._relay_url,
            data=enc.wire,
            headers=self._headers(stream=False),
            timeout=self._timeout,
        )
        if not resp.ok:
            raise RelayError(resp.status_code, _error_message(resp.content))

        from .tee_ohttp import decrypt_response

        inner_bytes = decrypt_response(enc.response_secret, enc.enc, resp.content)
        status, inner = _normalize_inner(json.loads(inner_bytes.decode("utf-8")))
        if status >= 400:
            raise RelayError(status, str(inner.get("error", "TEE inner error")))

        content = response_content_for_hash(inner)
        proof = verify_response(
            canonical_request=canonical,
            response_body=inner,
            response_content=content,
            signing_key_pem=self._signing_key_pem,
            expected_tee_id=self._tee.tee_id,
            tee_host=self._tee.endpoint,
        )
        return VerifiedChatResponse(body=inner, content=content, proof=proof)

    def stream_chat_completion(self, body: dict) -> VerifiedChatResponse:
        """Send a streaming chat completion, verify it, then return decrypted frames.

        The encrypted stream is fully buffered and verified before returning, so
        the returned ``stream_frames`` are safe to replay to an end user. (This
        trades streaming latency for the "no unverified token leaves the machine"
        guarantee.)

        Args:
            body: An OpenAI ``/v1/chat/completions`` request body (``stream`` is
                forced on for the wire request).

        Returns:
            A :class:`VerifiedChatResponse` with ``stream_frames`` populated.

        Raises:
            RelayError, VerificationError, UnsupportedRequestError: As for
            :meth:`chat_completion`.
        """
        wire, canonical = build_inner_request(body)
        wire = {**wire, "stream": True}
        enc = encapsulate_request(self._ohttp_public_key, json.dumps(wire).encode("utf-8"), **self._enc_ids)

        resp = self._session.post(
            self._relay_url,
            data=enc.wire,
            headers=self._headers(stream=True),
            timeout=self._timeout,
            stream=True,
        )
        if not resp.ok:
            raise RelayError(resp.status_code, _error_message(resp.content))

        decrypter = ChunkedResponseDecrypter(enc.chunked_response_secret, enc.enc)
        frames: list[str] = []
        full_content = ""
        tool_calls: dict[int, dict] = {}
        final_frame: Optional[dict] = None

        chunks = resp.iter_content(chunk_size=8192)
        try:
            for raw, is_last in _with_last(chunks):
                # A malformed/truncated encrypted stream is an integrity failure;
                # surface it as VerificationError, not a raw ValueError.
                for plaintext in decrypter.push(raw, done=is_last):
                    text = plaintext.decode("utf-8", errors="replace")
                    frames.append(text)
                    for parsed in _iter_sse_objects(text):
                        full_content += _delta_content(parsed)
                        _accumulate_tool_calls(tool_calls, parsed)
                        if isinstance(parsed.get("tee_signature"), str) or isinstance(parsed.get("tee_output_hash"), str):
                            final_frame = parsed
        except ValueError as exc:
            raise VerificationError(f"malformed TEE stream: {exc}") from exc

        if final_frame is None:
            raise VerificationError("TEE stream missing a signed final frame")

        # The gateway signs the assistant text, except for tool-call responses
        # where it signs json.dumps(tool_calls, sort_keys=True) of the buffered
        # calls — mirror that so honest tool-call streams verify.
        if _finish_reason(final_frame) == "tool_calls" and tool_calls:
            response_content = json.dumps([tool_calls[i] for i in sorted(tool_calls)], sort_keys=True)
        else:
            response_content = full_content

        proof = verify_response(
            canonical_request=canonical,
            response_body=final_frame,
            response_content=response_content,
            signing_key_pem=self._signing_key_pem,
            expected_tee_id=self._tee.tee_id,
            tee_host=self._tee.endpoint,
        )
        return VerifiedChatResponse(body=final_frame, content=response_content, proof=proof, stream_frames=frames)

    def _headers(self, *, stream: bool) -> dict:
        headers = {
            "Content-Type": OHTTP_REQUEST_MEDIA_TYPE,
            "Accept": OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE if stream else OHTTP_RESPONSE_MEDIA_TYPE,
            "X-TEE-ID": self._tee.tee_id,
        }
        if stream:
            headers["X-OHTTP-Stream"] = "true"
        if self._auth_headers:
            headers.update(self._auth_headers())
        return headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_inner(decoded) -> tuple[int, dict]:
    """Accept both ``{status, body}`` envelopes and a bare response object."""
    if isinstance(decoded, dict) and isinstance(decoded.get("status"), int) and isinstance(decoded.get("body"), dict):
        return decoded["status"], decoded["body"]
    if isinstance(decoded, dict):
        return 200, decoded
    raise VerificationError("malformed inner response")


def _iter_sse_objects(text: str):
    """Yield the parsed JSON objects from a decrypted SSE frame's ``data:`` lines."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if isinstance(parsed.get("error"), str):
            raise RelayError(502, parsed["error"])
        yield parsed


def _accumulate_tool_calls(buffered: dict[int, dict], frame: dict) -> None:
    """Fold a streamed ``delta.tool_calls`` fragment into ``buffered`` (keyed by index).

    Mirrors the gateway's streaming tool-call buffer so the reconstructed list
    matches the signed output: ids/names are set when present, argument fragments
    are concatenated in arrival order.
    """
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return
    fragments = delta.get("tool_calls")
    if not isinstance(fragments, list):
        return
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        idx = frag.get("index", 0)
        slot = buffered.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if frag.get("id"):
            slot["id"] = frag["id"]
        if frag.get("type"):
            slot["type"] = frag["type"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


def _finish_reason(frame: dict) -> Optional[str]:
    choices = frame.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


def _delta_content(frame: dict) -> str:
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def _with_last(iterable: Iterable[bytes]):
    """Yield ``(item, is_last)`` pairs, with ``is_last`` True only for the final item."""
    iterator = iter(iterable)
    try:
        prev = next(iterator)
    except StopIteration:
        # Empty stream: signal one final, empty push so the decrypter can report
        # the missing-final-marker truncation error rather than hanging.
        yield b"", True
        return
    for item in iterator:
        yield prev, False
        prev = item
    yield prev, True


def _error_message(content: bytes) -> str:
    try:
        body = json.loads(content.decode("utf-8"))
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("error") or "relay error")
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return content.decode("utf-8", errors="replace")[:500] or "relay error"
