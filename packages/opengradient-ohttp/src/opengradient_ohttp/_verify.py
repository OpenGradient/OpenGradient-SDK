"""Request canonicalization and TEE response verification.

Ported from ``opengradient.client.tee_verify``, with these changes:

  - ``max_completion_tokens`` is mapped to ``max_tokens`` (the gateway only
    knows ``max_tokens``), instead of being dropped silently.
  - ``aspect_ratio``, ``resolution`` and ``reasoning_effort`` are part of the
    request, as in the gateway's ``_chat_request_to_dict`` and the dashboard.
  - ``n > 1`` is rejected (the gateway returns one choice), and any other field
    the gateway would not sign produces one ``UserWarning`` naming it.
  - A ``provider/model`` id is reduced to the bare model name, as the dashboard
    and chat-app do.

The trust chain: the TEE's RSA signing key comes from the on-chain registry
(never from the relay), the response's ``tee_id`` must equal keccak256 of that
key, the request hash is recomputed from what was actually sent, the output
hash from what will be returned, and the RSA-PSS signature must verify over
``keccak256(request_hash || output_hash || uint256(timestamp))``.
"""

from __future__ import annotations

import base64
import copy
import json
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from eth_hash.auto import keccak

from ._errors import TeeVerificationError, UnsupportedRequestError


@dataclass(frozen=True)
class TeeProof:
    """The verified provenance of one response."""

    tee_id: str
    request_hash: str
    output_hash: str
    timestamp: int
    signature: str
    tee_host: Optional[str] = None


# ---------------------------------------------------------------------------
# Signing-key helpers
# ---------------------------------------------------------------------------


def pem_from_der(signing_public_key_der: bytes) -> str:
    """DER SubjectPublicKeyInfo → PEM."""
    key = serialization.load_der_public_key(bytes(signing_public_key_der))
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def tee_id_for_key(signing_key_pem: str) -> str:
    """``0x`` + keccak256(DER SubjectPublicKeyInfo): the registry's and gateway's tee_id."""
    key = serialization.load_pem_public_key(signing_key_pem.encode("utf-8"))
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "0x" + keccak(der).hex()


# ---------------------------------------------------------------------------
# Request canonicalization (must byte-match the gateway's request hashing)
# ---------------------------------------------------------------------------

# Fields the gateway folds into its signed request hash (besides model/messages/temperature).
_SIGNED_OPTIONAL_FIELDS = ("stop", "tools", "response_format", "aspect_ratio", "resolution")

# Fields forwarded on the wire that the gateway honours but does not sign.
_WIRE_ONLY_FIELDS = ("tool_choice",)

# Fields that are consumed here and need no warning when present.
_HANDLED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "web_search",
        "reasoning_effort",
        "n",
        "stream",
        "stream_options",
        *_SIGNED_OPTIONAL_FIELDS,
        *_WIRE_ONLY_FIELDS,
    }
)


def canonical_user_content(content: Any) -> Any:
    """Text verbatim; every attachment reduced to ``{type[, filename]}`` (bytes are never hashed)."""
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
    role = msg.get("role")
    if role in ("system", "developer"):  # the gateway treats developer as system
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
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": (tc.get("function") or {}).get("arguments", ""),
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


def split_model(model: str) -> tuple[str, Optional[str]]:
    """``"anthropic/claude-haiku-4-5"`` → ``("claude-haiku-4-5", "anthropic")``; bare ids pass through."""
    if "/" in model:
        provider, bare = model.split("/", 1)
        return bare, provider or None
    return model, None


def build_inner_request(body: Any) -> tuple[dict, dict]:
    """Build ``(wire, canonical)`` from an OpenAI chat-completions body.

    ``wire`` is what gets sealed and sent (full user content, so the model sees
    attachments); ``canonical`` is what the gateway hashes with
    ``json.dumps(sort_keys=True)``. ``stream`` is never part of either; the
    caller adds it to ``wire``.

    Raises:
        UnsupportedRequestError: Missing ``model``/``messages``, an unknown
            role, ``n > 1``, or ``tools`` that isn't a list.
    """
    if not isinstance(body, dict):
        raise UnsupportedRequestError("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise UnsupportedRequestError("request is missing a 'model'")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise UnsupportedRequestError("request is missing 'messages'")
    for m in messages:
        if not isinstance(m, dict):
            raise UnsupportedRequestError(f"each message must be a JSON object, got {type(m).__name__}")
    n = body.get("n")
    if n is not None and n != 1:
        raise UnsupportedRequestError("only n=1 is supported over OHTTP")

    bare_model, _ = split_model(model)
    temperature = body.get("temperature")
    canonical: dict[str, Any] = {
        "model": bare_model,
        "messages": [_canonical_message(m) for m in messages],
        "temperature": float(temperature) if temperature is not None else 0.0,
    }

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is not None:
        canonical["max_tokens"] = max_tokens

    for field in _SIGNED_OPTIONAL_FIELDS:
        if body.get(field):
            canonical[field] = body[field]
    if "tools" in canonical and not isinstance(canonical["tools"], list):
        raise UnsupportedRequestError(f"'tools' must be a list, got {type(canonical['tools']).__name__}")
    if body.get("web_search"):
        canonical["web_search"] = True
    if body.get("reasoning_effort") is not None:
        canonical["reasoning_effort"] = body["reasoning_effort"]

    dropped = sorted(k for k, v in body.items() if k not in _HANDLED_FIELDS and v is not None)
    if dropped:
        warnings.warn(
            "opengradient-ohttp: the TEE gateway does not sign these request fields, so they were not sent: " + ", ".join(dropped),
            UserWarning,
            stacklevel=2,
        )

    wire = copy.deepcopy(canonical)
    # Restore full user content (attachment bytes) on the wire; only the hash strips them.
    wire["messages"] = [
        ({**cm, "content": orig.get("content")} if cm.get("role") == "user" else cm) for cm, orig in zip(wire["messages"], messages)
    ]
    for field in _WIRE_ONLY_FIELDS:
        if body.get(field) is not None:
            wire[field] = body[field]
    return wire, canonical


def canonical_request_bytes(canonical_request: dict) -> bytes:
    """Serialize exactly as the gateway hashes: ``json.dumps(sort_keys=True)``."""
    return json.dumps(canonical_request, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Response verification
# ---------------------------------------------------------------------------


def response_content_for_hash(response_body: dict) -> str:
    """The exact string the gateway signed for a single-shot response.

    Tool-call responses: ``json.dumps(tool_calls, sort_keys=True)``; otherwise
    the assistant text (generated images ride outside the signature).
    """
    choices = response_body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if isinstance(message, dict):
        if choice.get("finish_reason") == "tool_calls" and isinstance(message.get("tool_calls"), list):
            return json.dumps(message["tool_calls"], sort_keys=True)
        return content_text(message.get("content"))
    return ""


def content_text(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return ""


def verify_response(
    *,
    canonical_request: dict,
    response_body: dict,
    response_content: str,
    signing_key_pem: str,
    expected_tee_id: str,
    tee_host: Optional[str] = None,
) -> TeeProof:
    """Verify a decrypted gateway response (single-shot body, or a stream's signed final frame).

    Raises:
        TeeVerificationError: Missing fields, tee_id mismatch, request/output
            hash mismatch, or a bad signature.
    """
    signature_b64 = _require_str(response_body, "tee_signature")
    reported_request_hash = _require_str(response_body, "tee_request_hash")
    reported_output_hash = _require_str(response_body, "tee_output_hash")
    timestamp = _require_int(response_body, "tee_timestamp")
    reported_tee_id = _require_str(response_body, "tee_id")

    key_tee_id = tee_id_for_key(signing_key_pem)
    if _norm(reported_tee_id) != _norm(key_tee_id):
        raise TeeVerificationError(f"tee_id mismatch: response says {reported_tee_id}, signing key is {key_tee_id}")
    if _norm(reported_tee_id) != _norm(expected_tee_id):
        raise TeeVerificationError(f"tee_id mismatch: response says {reported_tee_id}, request was sealed to {expected_tee_id}")

    computed_request_hash = keccak(canonical_request_bytes(canonical_request)).hex()
    if computed_request_hash != _norm(reported_request_hash):
        raise TeeVerificationError(
            "request hash mismatch: the gateway signed a different request than was sent "
            f"(computed {computed_request_hash}, signed {reported_request_hash})"
        )

    computed_output_hash = keccak(response_content.encode("utf-8")).hex()
    if computed_output_hash != _norm(reported_output_hash):
        raise TeeVerificationError(
            "output hash mismatch: the response content does not match the signed output "
            f"(computed {computed_output_hash}, signed {reported_output_hash})"
        )

    msg_hash = keccak(bytes.fromhex(computed_request_hash) + bytes.fromhex(computed_output_hash) + timestamp.to_bytes(32, "big"))
    key = serialization.load_pem_public_key(signing_key_pem.encode("utf-8"))
    if not isinstance(key, RSAPublicKey):
        raise TeeVerificationError("TEE signing key is not an RSA public key")
    try:
        signature = base64.b64decode(signature_b64)
    except ValueError as exc:
        raise TeeVerificationError("tee_signature is not valid base64") from exc
    try:
        key.verify(
            signature,
            msg_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise TeeVerificationError("RSA-PSS signature verification failed") from exc

    return TeeProof(
        tee_id=reported_tee_id,
        request_hash=computed_request_hash,
        output_hash=computed_output_hash,
        timestamp=timestamp,
        signature=signature_b64,
        tee_host=tee_host,
    )


def _norm(value: str) -> str:
    return (value[2:] if value.startswith("0x") else value).lower()


def _require_str(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise TeeVerificationError(f"response is missing a string '{key}', so it cannot be verified")
    return value


def _require_int(body: dict, key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TeeVerificationError(f"response is missing an integer '{key}', so it cannot be verified")
    try:
        return int(value)
    except ValueError as exc:
        raise TeeVerificationError(f"'{key}' is not an integer: {value!r}") from exc
