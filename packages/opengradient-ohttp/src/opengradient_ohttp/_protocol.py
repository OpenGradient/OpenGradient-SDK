"""The I/O-free half of the transports: request preparation, relay-error
mapping, single-shot response opening, and the streaming verifier.

Shared by :class:`OHTTPTransport` and :class:`AsyncOHTTPTransport`, which only
add the network calls.

Error policy (what the OpenAI SDK sees):

  - Every failure the transport can name becomes an HTTP response with an
    OpenAI-shaped body ``{"error": {"message", "type", "code", "param"}}``, so
    ``openai.AuthenticationError`` / ``RateLimitError`` / ``APIStatusError``
    work as usual.
  - ``x-should-retry: false`` is set on anything a retry can't fix or might
    bill twice: 4xx other than 408/429, inner (model-side) errors, and TEE
    verification failures. Raising an exception instead would make the OpenAI
    SDK retry it twice and report "Connection error.".
  - Inside a stream, an in-band ``data: {"error": ...}`` frame is passed through
    (the OpenAI SDK raises ``openai.APIError`` for it); a verification failure
    or a truncated stream raises :class:`TeeVerificationError` /
    :class:`StreamIncompleteError` from the iterator.
"""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from cryptography.exceptions import InvalidTag

from ._errors import StreamIncompleteError, TeeVerificationError, UnsupportedRequestError
from ._hpke import ChunkedResponseDecrypter, EncapsulatedRequest, decrypt_response, encapsulate_request
from ._registry import TeeTarget
from ._verify import TeeProof, build_inner_request, response_content_for_hash, split_model, verify_response

DEFAULT_RELAY_ORIGIN = "https://chat-api.memchat.io"
DEVELOPER_OHTTP_PATH = "/api/v1/developer/ohttp"
DEFAULT_RELAY_URL = DEFAULT_RELAY_ORIGIN + DEVELOPER_OHTTP_PATH

OHTTP_REQUEST_MEDIA_TYPE = "message/ohttp-req"
OHTTP_RESPONSE_MEDIA_TYPE = "message/ohttp-res"
OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE = "message/ohttp-chunked-res"

# chat-api's code when the TEE a request was sealed to has left the registry.
RELAY_CODE_TEE_NOT_ACTIVE = "tee_not_active"

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

_STATUS_HINTS = {
    401: "The API key is invalid or revoked, or its creator has left the developer account.",
    402: "The developer account for this API key is out of credits.",
    403: "This API key's account is not allowed to make requests.",
    429: "Rate limit reached for this developer account.",
}


def relay_url_from(url: str) -> str:
    """Accept either the relay origin or the full relay URL.

    ``https://chat-api.memchat.io`` → ``https://chat-api.memchat.io/api/v1/developer/ohttp``.
    A URL that already has a path is used as given.
    """
    parsed = httpx.URL(url)
    if parsed.path in ("", "/"):
        return str(parsed.copy_with(path=DEVELOPER_OHTTP_PATH))
    return url.rstrip("/")


def is_chat_completions(request: httpx.Request) -> bool:
    return request.method == "POST" and request.url.path.rstrip("/").endswith("/chat/completions")


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedCall:
    """One OpenAI chat-completions call, ready to seal."""

    wire: dict
    canonical: dict
    stream: bool
    model: str
    provider: Optional[str]


def prepare_call(raw_body: bytes) -> PreparedCall:
    """Parse the OpenAI request body.

    Raises:
        UnsupportedRequestError: If the body isn't a valid chat-completions request.
    """
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnsupportedRequestError("chat-completions request body must be valid JSON") from exc
    wire, canonical = build_inner_request(body)
    stream = bool(body.get("stream"))
    if stream:
        # `stream` rides on the wire but is not part of the signed request hash.
        wire = {**wire, "stream": True}
    _, provider = split_model(body["model"])
    return PreparedCall(wire=wire, canonical=canonical, stream=stream, model=canonical["model"], provider=provider)


def seal(call: PreparedCall, tee: TeeTarget) -> EncapsulatedRequest:
    return encapsulate_request(tee.ohttp_public_key, json.dumps(call.wire).encode("utf-8"), key_id=tee.key_id)


def outer_request(
    *,
    relay_url: str,
    call: PreparedCall,
    tee: TeeTarget,
    sealed: EncapsulatedRequest,
    authorization: Optional[str],
    send_model_hint: bool,
    timeout: dict[str, Any],
    user_agent: str,
) -> httpx.Request:
    """The request chat-api sees: ciphertext plus routing headers, nothing else from the caller."""
    headers = {
        "Content-Type": OHTTP_REQUEST_MEDIA_TYPE,
        "Accept": OHTTP_CHUNKED_RESPONSE_MEDIA_TYPE if call.stream else OHTTP_RESPONSE_MEDIA_TYPE,
        "X-TEE-ID": tee.tee_id,
        "User-Agent": user_agent,
    }
    if call.stream:
        headers["X-OHTTP-Stream"] = "true"
    if authorization:
        headers["Authorization"] = authorization
    if send_model_hint:
        headers["X-OHTTP-Model"] = call.model
        headers["X-OHTTP-Provider"] = call.provider or infer_provider(call.model)
    return httpx.Request("POST", relay_url, content=sealed.wire, headers=headers, extensions={"timeout": timeout})


def infer_provider(model: str) -> str:
    """The dashboard's vendor guess for a bare model id (telemetry only)."""
    if model.startswith("gpt") or model.startswith("o"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("grok"):
        return "x-ai"
    return "unknown"


# ---------------------------------------------------------------------------
# Responses the transport hands back to the OpenAI SDK
# ---------------------------------------------------------------------------


def error_response(
    request: httpx.Request,
    status: int,
    message: str,
    *,
    code: Optional[str],
    error_type: str = "ohttp_error",
    retry: Optional[bool] = False,
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    """An OpenAI-shaped error response. ``retry=None`` leaves the OpenAI SDK's own policy in charge."""
    out_headers = dict(headers or {})
    if retry is not None:
        out_headers["x-should-retry"] = "true" if retry else "false"
    return httpx.Response(
        status,
        json={"error": {"message": message, "type": error_type, "code": code, "param": None}},
        headers=out_headers,
        request=request,
    )


@dataclass(frozen=True)
class RelayError:
    status: int
    message: str
    code: Optional[str]
    retry_after: Optional[str]


def parse_relay_error(status: int, content: bytes, headers: httpx.Headers) -> RelayError:
    """Read chat-api's error shapes.

    ``{"detail": "..."}`` (FastAPI), ``{"detail": {"code", "message", "upstream"}}``
    (TEE relay errors), ``{"error": "..."}`` (slowapi 429, gateway errors).
    """
    message: Optional[str] = None
    code: Optional[str] = None
    try:
        body = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, dict):
            message = detail.get("message") if isinstance(detail.get("message"), str) else None
            code = detail.get("code") if isinstance(detail.get("code"), str) else None
        elif isinstance(detail, list):  # FastAPI validation errors
            message = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
        error = body.get("error")
        if message is None and isinstance(error, str):
            message = error
        elif message is None and isinstance(error, dict):
            message = error.get("message") if isinstance(error.get("message"), str) else None
            code = code or (error.get("code") if isinstance(error.get("code"), str) else None)
        if message is None and isinstance(body.get("message"), str):
            message = body["message"]
    if message is None:
        text = content.decode("utf-8", errors="replace").strip()[:500]
        message = text or f"relay returned HTTP {status}"

    hint = _STATUS_HINTS.get(status)
    if hint and hint not in message:
        message = f"{hint} (relay: {message})"
    return RelayError(status=status, message=message, code=code, retry_after=headers.get("retry-after"))


def relay_error_response(request: httpx.Request, error: RelayError) -> httpx.Response:
    status = error.status
    # 408/429 and 5xx: let the OpenAI SDK retry (chat-api bills only successful
    # upstream calls). Other 4xx can't succeed on retry.
    retry: Optional[bool] = None if status in (408, 429) or status >= 500 else False
    headers = {"retry-after": error.retry_after} if error.retry_after else None
    return error_response(
        request,
        status,
        error.message,
        code=error.code or f"relay_http_{status}",
        error_type="relay_error",
        retry=retry,
        headers=headers,
    )


def should_refresh_tee(error: RelayError) -> bool:
    """A relay error that means the cached TEE choice is stale or unhealthy."""
    return error.status >= 500 and bool(error.code) and str(error.code).startswith("tee_")


class InnerError(Exception):
    """The gateway answered, but the model call itself failed (inner status >= 400)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def open_single_response(
    call: PreparedCall,
    tee: TeeTarget,
    sealed: EncapsulatedRequest,
    content: bytes,
    content_type: str,
) -> tuple[dict, TeeProof]:
    """Decrypt and verify a single-shot response.

    Raises:
        TeeVerificationError: Undecryptable or unverifiable response.
        InnerError: The decrypted envelope carries an error status.
    """
    try:
        inner_bytes = decrypt_response(sealed.response_secret, sealed.enc, content)
    except (ValueError, InvalidTag) as exc:
        raise TeeVerificationError(f"could not decrypt the relay's response (content-type {content_type or 'none'!r})") from exc
    try:
        decoded = json.loads(inner_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeeVerificationError("decrypted response is not JSON") from exc

    if isinstance(decoded, dict) and isinstance(decoded.get("status"), int) and isinstance(decoded.get("body"), dict):
        status, inner = decoded["status"], decoded["body"]
    elif isinstance(decoded, dict):
        status, inner = 200, decoded
    else:
        raise TeeVerificationError("decrypted response is not a JSON object")
    if status >= 400:
        error = inner.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        raise InnerError(status, str(message or inner.get("message") or "TEE gateway error"))

    proof = verify_response(
        canonical_request=call.canonical,
        response_body=inner,
        response_content=response_content_for_hash(inner),
        signing_key_pem=tee.signing_key_pem,
        expected_tee_id=tee.tee_id,
        tee_host=tee.endpoint,
    )
    return inner, proof


def proof_headers(proof: TeeProof) -> dict[str, str]:
    """Expose the proof on the response (read via ``with_raw_response``)."""
    headers = {
        "x-tee-id": proof.tee_id,
        "x-tee-request-hash": proof.request_hash,
        "x-tee-output-hash": proof.output_hash,
        "x-tee-timestamp": str(proof.timestamp),
        "x-tee-signature": proof.signature,
        "x-tee-verified": "true",
    }
    if proof.tee_host:
        headers["x-tee-host"] = proof.tee_host
    return headers


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamVerifier:
    """Decrypts a chunked OHTTP response and checks it as SSE lines go by.

    Lines are passed on as soon as they're complete (UTF-8 and lines are
    reassembled across chunk boundaries). The TEE's signed final frame is
    verified *before* it's passed on, and ``data: [DONE]`` is only passed on
    after that, so a consumer that stops at ``[DONE]`` (the OpenAI SDK does)
    can't receive an unverified stream without an error. Content after the
    signed frame is rejected.

    With ``verify_before_release=True`` nothing is passed on until the whole
    stream has been verified.
    """

    def __init__(
        self,
        call: PreparedCall,
        tee: TeeTarget,
        sealed: EncapsulatedRequest,
        *,
        verify_before_release: bool = False,
    ):
        self._call = call
        self._tee = tee
        self._decrypter = ChunkedResponseDecrypter(sealed.chunked_response_secret, sealed.enc)
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._pending = ""
        self._content: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self._buffered = verify_before_release
        self._held: list[bytes] = []
        self._errored = False
        self.proof: Optional[TeeProof] = None

    def feed(self, raw: bytes) -> bytes:
        """Feed raw response bytes; returns SSE bytes ready to hand on (maybe empty)."""
        try:
            chunks = self._decrypter.push(raw, done=False)
        except InvalidTag as exc:
            raise TeeVerificationError("an encrypted stream chunk failed authentication") from exc
        except ValueError as exc:
            raise TeeVerificationError(f"malformed encrypted stream: {exc}") from exc
        return self._release(self._consume(chunks, final=False), final=False)

    def finish(self) -> bytes:
        """Signal end-of-stream; returns any remaining SSE bytes.

        Raises:
            StreamIncompleteError: The stream ended before its final chunk or signed frame.
            TeeVerificationError: The last chunk failed authentication.
        """
        try:
            chunks = self._decrypter.push(None, done=True)
        except InvalidTag as exc:
            raise TeeVerificationError("the final encrypted stream chunk failed authentication") from exc
        except ValueError as exc:
            if self._errored:
                return self._release("", final=True)
            raise StreamIncompleteError(f"the encrypted stream ended early ({exc}); the reply is incomplete and unverified") from exc
        text = self._consume(chunks, final=True)
        if self.proof is None and not self._errored:
            raise StreamIncompleteError("the stream ended without the TEE's signed final frame; the reply is unverified")
        return self._release(text, final=True)

    # -- internals -----------------------------------------------------------

    def _consume(self, chunks: list[bytes], *, final: bool) -> str:
        try:
            text = "".join(self._decoder.decode(chunk) for chunk in chunks)
            if final:
                text += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise TeeVerificationError("decrypted stream is not valid UTF-8") from exc

        data = self._pending + text
        if final:
            complete, self._pending = data, ""
        else:
            cut = data.rfind("\n")
            if cut < 0:
                self._pending = data
                return ""
            complete, self._pending = data[: cut + 1], data[cut + 1 :]

        # Split on "\n" only (str.splitlines would also split on U+2028 and friends).
        out: list[str] = []
        parts = complete.split("\n")
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                if part:  # an unterminated last line (end of stream only): terminate the event
                    out.append(self._handle_line(part + "\n\n"))
                break
            out.append(self._handle_line(part + "\n"))
        return "".join(out)

    def _handle_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return line
        payload = stripped[len("data:") :].strip()
        if payload == "[DONE]":
            if self.proof is None and not self._errored:
                raise StreamIncompleteError("the stream finished without the TEE's signed final frame; the reply is unverified")
            return line
        if not payload:
            return line
        try:
            frame = json.loads(payload)
        except json.JSONDecodeError:
            return line
        if not isinstance(frame, dict):
            return line

        error = frame.get("error")
        if error:
            # In-band failure after the stream started; the OpenAI SDK raises APIError for it.
            # It only reads a message out of an object, so wrap the gateway's plain string.
            self._errored = True
            if isinstance(error, str):
                wrapped = {"error": {"message": error, "type": "tee_stream_error", "code": None}}
                return f"data: {json.dumps(wrapped)}" + line[len(line.rstrip("\r\n")) :]
            return line

        if self.proof is not None:
            if _delta_content(frame) or _delta_tool_calls(frame):
                raise TeeVerificationError("the stream carried content after the TEE's signed final frame")
            return line

        self._content.append(_delta_content(frame))
        _accumulate_tool_calls(self._tool_calls, frame)
        if isinstance(frame.get("tee_signature"), str) or isinstance(frame.get("tee_output_hash"), str):
            self.proof = self._verify(frame)
        return line

    def _verify(self, final_frame: dict) -> TeeProof:
        # The gateway signs the text, or for tool calls json.dumps(tool_calls, sort_keys=True).
        if _finish_reason(final_frame) == "tool_calls" and self._tool_calls:
            content = json.dumps([self._tool_calls[i] for i in sorted(self._tool_calls)], sort_keys=True)
        else:
            content = "".join(self._content)
        return verify_response(
            canonical_request=self._call.canonical,
            response_body=final_frame,
            response_content=content,
            signing_key_pem=self._tee.signing_key_pem,
            expected_tee_id=self._tee.tee_id,
            tee_host=self._tee.endpoint,
        )

    def _release(self, text: str, *, final: bool) -> bytes:
        data = text.encode("utf-8")
        if self._buffered and not final and not self._errored:
            if data:
                self._held.append(data)
            return b""
        if self._held:
            data = b"".join(self._held) + data
            self._held = []
        return data


def _first_choice(frame: dict) -> dict:
    choices = frame.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _finish_reason(frame: dict) -> Optional[str]:
    reason = _first_choice(frame).get("finish_reason")
    return reason if isinstance(reason, str) else None


def _delta(frame: dict) -> dict:
    delta = _first_choice(frame).get("delta")
    return delta if isinstance(delta, dict) else {}


def _delta_content(frame: dict) -> str:
    content = _delta(frame).get("content")
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else ""


def _delta_tool_calls(frame: dict) -> list:
    calls = _delta(frame).get("tool_calls")
    return calls if isinstance(calls, list) else []


def _accumulate_tool_calls(buffered: dict[int, dict], frame: dict) -> None:
    """Fold streamed ``delta.tool_calls`` fragments into ``buffered``, as the gateway does."""
    for frag in _delta_tool_calls(frame):
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
