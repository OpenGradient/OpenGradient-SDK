"""Cryptographic verification of TEE inference responses.

Every response from an OpenGradient TEE gateway carries an RSA-PSS signature over
``keccak256(requestHash || outputHash || uint256(timestamp))``, produced inside
the enclave with the key the on-chain registry records for that TEE. The trust
chain is:

    reproducible build -> PCRs -> on-chain registry entry (pcrHash + signing key)
                       -> per-response RSA-PSS signature

So if you trust a TEE's registry signing key (optionally pinned to an expected
``pcrHash``) and the signature verifies, the response was produced inside that
attested enclave and was not modified in transit — no trust in the relay, the
host, or us is required. The relay never holds the signing key and so cannot
forge a response.

This module mirrors the gateway's signing (``compute_tee_msg_hash`` + RSA-PSS,
salt length 32) and the chat-app's browser verification, kept in one place so all
clients verify identically.
"""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from eth_hash.auto import keccak


class VerificationError(Exception):
    """Raised when a response fails any step of TEE verification.

    Callers should treat this as fatal: never surface content that failed
    verification to the end user.
    """


class UnsupportedRequestError(Exception):
    """Raised when an OpenAI-style request cannot be expressed as a gateway request."""


@dataclass
class TeeProof:
    """The verified provenance of a single response.

    Attributes:
        tee_id: The TEE identity (``0x`` + keccak256 of the signing key DER).
        request_hash: keccak256 of the canonical request, hex (no ``0x``).
        output_hash: keccak256 of the signed output content, hex (no ``0x``).
        timestamp: The enclave-asserted signing timestamp (unix seconds).
        signature: The base64 RSA-PSS signature that was verified.
        signing_key_pem: The PEM signing key the signature verified against.
        tee_host: Optional host the response came from, for display.
    """

    tee_id: str
    request_hash: str
    output_hash: str
    timestamp: int
    signature: str
    signing_key_pem: str
    tee_host: Optional[str] = None


# ---------------------------------------------------------------------------
# Signing-key helpers
# ---------------------------------------------------------------------------


def pem_from_der(signing_public_key_der: bytes) -> str:
    """Convert a DER (SPKI) public key (e.g. ``TEEEndpoint.signing_public_key_der``) to PEM."""
    key = serialization.load_der_public_key(bytes(signing_public_key_der))
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def tee_id_for_key(signing_key_pem: str) -> str:
    """Return ``0x`` + keccak256(DER(SubjectPublicKeyInfo)).

    Matches the gateway's ``TEEKeyManager.tee_id`` and the registry's keyed tee_id.
    """
    key = serialization.load_pem_public_key(signing_key_pem.encode("utf-8"))
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "0x" + keccak(der).hex()


# ---------------------------------------------------------------------------
# Request canonicalization (must byte-match the gateway's request hashing)
# ---------------------------------------------------------------------------


def canonical_user_content(content: Any) -> Any:
    """Canonicalize user-message content for request hashing.

    Plain strings pass through. For multimodal content (a list of parts), text is
    kept verbatim and every attachment is reduced to ``{type[, filename]}`` — the
    inline bytes are dropped (they ride inside the encrypted envelope and are never
    hashed). Mirrors the gateway's ``canonical_user_content``.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    canonical: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            canonical.append({"type": "text", "text": str(part)})
            continue
        if part.get("type") == "text":
            canonical.append({"type": "text", "text": part.get("text", "") or ""})
            continue
        entry: dict[str, Any] = {"type": part.get("type")}
        file_obj = part.get("file")
        filename = (file_obj.get("filename") if isinstance(file_obj, dict) else None) or part.get("filename")
        if filename:
            entry["filename"] = filename
        canonical.append(entry)
    return canonical


def _canonical_message(msg: dict) -> dict:
    """Shape one message exactly as the gateway does before hashing."""
    role = msg.get("role")
    if role in ("system", "developer"):  # gateway treats developer as system
        return {"role": "system", "content": msg.get("content")}
    if role == "user":
        return {"role": "user", "content": canonical_user_content(msg.get("content"))}
    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
                for tc in tool_calls
            ]
        return out
    if role == "tool":
        return {"role": "tool", "content": msg.get("content"), "tool_call_id": msg.get("tool_call_id")}
    if role == "function":
        return {"role": "function", "content": msg.get("content"), "name": msg.get("name")}
    raise UnsupportedRequestError(f"unknown message role: {role!r}")


def build_inner_request(body: dict) -> tuple[dict, dict]:
    """Build ``(wire, canonical)`` request dicts from an OpenAI chat-completions ``body``.

    The gateway commits (in its signed request hash) to only a fixed subset of
    fields, so anything else the caller sends (``n``, ``top_p``, ``tool_choice``,
    ...) is intentionally dropped — it would not be covered by the signature.

    Args:
        body: An OpenAI ``/v1/chat/completions`` request body.

    Returns:
        ``(wire, canonical)`` where ``wire`` is the object to encrypt and send
        (full message content preserved so the model sees attachment bytes), and
        ``canonical`` is the dict whose ``json.dumps(sort_keys=True)`` the gateway
        hashes (attachment bytes stripped). Pass ``canonical`` to
        :func:`verify_response`.

    Raises:
        UnsupportedRequestError: If the body is missing ``model`` or ``messages`` or
            contains an unknown message role.
    """
    if not isinstance(body, dict):
        raise UnsupportedRequestError("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise UnsupportedRequestError("request is missing a 'model'")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise UnsupportedRequestError("request is missing 'messages'")

    canonical_messages = [_canonical_message(m) for m in messages]

    temperature = body.get("temperature")
    canonical: dict[str, Any] = {
        "model": model,
        "messages": canonical_messages,
        "temperature": float(temperature) if temperature is not None else 0.0,
    }
    if body.get("max_tokens") is not None:
        canonical["max_tokens"] = body["max_tokens"]
    if body.get("stop"):
        canonical["stop"] = body["stop"]
    if body.get("tools"):
        tools = body["tools"]
        canonical["tools"] = tools if isinstance(tools, list) else list(tools)
    if body.get("response_format"):
        canonical["response_format"] = body["response_format"]
    if body.get("web_search"):
        canonical["web_search"] = True

    wire = copy.deepcopy(canonical)
    wire["messages"] = [
        ({**cm, "content": orig.get("content")} if cm.get("role") == "user" else cm)
        for cm, orig in zip(wire["messages"], messages)
    ]
    return wire, canonical


def canonical_request_bytes(canonical_request: dict) -> bytes:
    """Serialize a canonical request exactly as the gateway hashes it: ``json.dumps(sort_keys=True)``."""
    return json.dumps(canonical_request, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Response content + signature verification
# ---------------------------------------------------------------------------


def response_content_for_hash(response_body: dict) -> str:
    """Extract the exact string the gateway hashed as the signed output.

    For tool-call responses the gateway hashes ``json.dumps(tool_calls, sort_keys=True)``;
    otherwise it hashes the assistant message text (generated image bytes are
    excluded — they ride out-of-band and are not signed).
    """
    choice = _first_choice(response_body)
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        if choice.get("finish_reason") == "tool_calls" and isinstance(message.get("tool_calls"), list):
            return json.dumps(message["tool_calls"], sort_keys=True)
        return _content_text(message.get("content"))
    return ""


def verify_response(
    *,
    canonical_request: dict,
    response_body: dict,
    response_content: str,
    signing_key_pem: str,
    expected_tee_id: Optional[str] = None,
    tee_host: Optional[str] = None,
) -> TeeProof:
    """Verify a (decrypted) TEE gateway response.

    Args:
        canonical_request: The canonical request dict (see :func:`build_inner_request`)
            whose ``json.dumps(sort_keys=True)`` the gateway hashed.
        response_body: The parsed inner JSON (single-shot body, or the final SSE
            frame for streams), carrying ``tee_signature``, ``tee_request_hash``,
            ``tee_output_hash``, ``tee_timestamp`` and ``tee_id``.
        response_content: The exact text/JSON the gateway hashed as output — use
            :func:`response_content_for_hash`, or the accumulated stream text.
        signing_key_pem: The enclave's RSA public key from the on-chain registry
            (the trust anchor; convert DER via :func:`pem_from_der`).
        expected_tee_id: If given, require the response/key tee_id to match.
        tee_host: Optional host, recorded on the returned proof for display.

    Returns:
        A :class:`TeeProof` describing the verified provenance.

    Raises:
        VerificationError: If any check fails (missing fields, tee_id mismatch,
            request/output hash mismatch, or bad signature).
    """
    signature_b64 = _require_str(response_body, "tee_signature")
    reported_request_hash = _require_str(response_body, "tee_request_hash")
    reported_output_hash = _require_str(response_body, "tee_output_hash")
    timestamp = _require_int(response_body, "tee_timestamp")
    reported_tee_id = _require_str(response_body, "tee_id")

    # 1. The signing key must key the tee_id the response claims, so a signature
    #    from enclave A cannot be replayed as enclave B's.
    key_tee_id = tee_id_for_key(signing_key_pem)
    if _strip0x(reported_tee_id).lower() != _strip0x(key_tee_id).lower():
        raise VerificationError(f"tee_id mismatch: response says {reported_tee_id}, signing key is {key_tee_id}")
    if expected_tee_id and _strip0x(reported_tee_id).lower() != _strip0x(expected_tee_id).lower():
        raise VerificationError(f"tee_id mismatch: response says {reported_tee_id}, expected {expected_tee_id}")

    # 2. Recompute the request hash from what we actually sent.
    computed_request_hash = keccak(canonical_request_bytes(canonical_request)).hex()
    if computed_request_hash != _strip0x(reported_request_hash):
        raise VerificationError(
            "request hash mismatch: the gateway signed a different request than we sent "
            f"(computed {computed_request_hash}, signed {reported_request_hash})"
        )

    # 3. Recompute the output hash from the content we're about to return.
    computed_output_hash = keccak(response_content.encode("utf-8")).hex()
    if computed_output_hash != _strip0x(reported_output_hash):
        raise VerificationError(
            "output hash mismatch: the response content does not match the signed output "
            f"(computed {computed_output_hash}, signed {reported_output_hash})"
        )

    # 4. Rebuild the signed message hash and verify the RSA-PSS signature.
    #    msg_hash = keccak256(inputHash || outputHash || uint256(timestamp))
    input_hash = bytes.fromhex(computed_request_hash)
    output_hash = bytes.fromhex(computed_output_hash)
    msg_hash = keccak(input_hash + output_hash + timestamp.to_bytes(32, "big"))

    key = serialization.load_pem_public_key(signing_key_pem.encode("utf-8"))
    if not isinstance(key, RSAPublicKey):
        raise VerificationError("signing key is not an RSA public key")
    try:
        key.verify(
            base64.b64decode(signature_b64),
            msg_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise VerificationError("RSA-PSS signature verification failed") from exc

    return TeeProof(
        tee_id=reported_tee_id,
        request_hash=computed_request_hash,
        output_hash=computed_output_hash,
        timestamp=timestamp,
        signature=signature_b64,
        signing_key_pem=signing_key_pem,
        tee_host=tee_host,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_choice(body: dict) -> dict:
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return ""


def _strip0x(value: str) -> str:
    return value[2:] if value.startswith("0x") else value


def _require_str(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise VerificationError(f"response is missing a string '{key}' — cannot verify")
    return value


def _require_int(body: dict, key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise VerificationError(f"response is missing an integer '{key}' — cannot verify")
    try:
        return int(value)
    except ValueError as exc:
        raise VerificationError(f"'{key}' is not an integer: {value!r}") from exc
