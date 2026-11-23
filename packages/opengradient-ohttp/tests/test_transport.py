"""End-to-end through the real OpenAI SDK: OpenAI → OHTTPTransport → (mock) chat-api relay → fake TEE."""

from __future__ import annotations

import json

import httpx
import openai
import pytest
from _fakes import API_KEY, BASE_URL, RELAY_URL, FakeGateway, FakeRelay
from eth_abi import encode as abi_encode

from opengradient_ohttp import (
    AsyncOHTTPTransport,
    OHTTPTransport,
    StreamIncompleteError,
    TeeVerificationError,
)
from opengradient_ohttp._registry import TEE_INFO_ARRAY_TYPE

MESSAGES = [{"role": "user", "content": "a secret prompt"}]


def make_client(relay: FakeRelay, *, max_retries: int = 2, **transport_kwargs) -> openai.OpenAI:
    transport_kwargs.setdefault("tee", relay.gateway.tee)
    transport = OHTTPTransport(transport=httpx.MockTransport(relay), **transport_kwargs)
    return openai.OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=httpx.Client(transport=transport), max_retries=max_retries)


def stream_text(stream) -> str:
    return "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_non_stream_round_trip_reaches_developer_relay(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    completion = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, max_tokens=50)

    assert completion.choices[0].message.content == relay.reply
    assert len(relay.requests) == 1
    outer = relay.requests[0]
    assert str(outer.url) == RELAY_URL
    assert outer.headers["authorization"] == f"Bearer {API_KEY}"
    assert outer.headers["content-type"] == "message/ohttp-req"
    assert outer.headers["accept"] == "message/ohttp-res"
    assert outer.headers["x-tee-id"] == gateway.tee_id
    assert "x-ohttp-stream" not in outer.headers
    assert "x-ohttp-model" not in outer.headers
    # Nothing the OpenAI SDK adds about the caller's machine reaches the relay.
    assert not any(name.lower().startswith("x-stainless") for name in outer.headers)
    # The relay only ever sees ciphertext.
    assert b"secret prompt" not in outer.content
    assert relay.inner_requests[0]["messages"] == MESSAGES
    assert relay.inner_requests[0]["max_tokens"] == 50
    assert "stream" not in relay.inner_requests[0]


def test_proof_is_exposed_on_raw_response(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    raw = client.chat.completions.with_raw_response.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert raw.headers["x-tee-verified"] == "true"
    assert raw.headers["x-tee-id"] == gateway.tee_id
    assert raw.headers["x-tee-host"] == gateway.endpoint
    assert raw.parse().choices[0].message.content == relay.reply


def test_explicit_api_key_wins_over_openai_key(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay, api_key="df_sk_explicit")

    client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert relay.requests[0].headers["authorization"] == "Bearer df_sk_explicit"


def test_relay_origin_gets_developer_path(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay, relay_url="https://relay.example.test")

    client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert str(relay.requests[0].url) == "https://relay.example.test/api/v1/developer/ohttp"


def test_model_hint_is_opt_in(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay, send_model_hint=True)

    client.chat.completions.create(model="anthropic/claude-haiku-4-5", messages=MESSAGES)

    outer = relay.requests[0]
    assert outer.headers["x-ohttp-model"] == "claude-haiku-4-5"
    assert outer.headers["x-ohttp-provider"] == "anthropic"
    assert relay.inner_requests[0]["model"] == "claude-haiku-4-5"


def test_max_completion_tokens_is_sent_as_max_tokens(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, max_completion_tokens=77)

    inner = relay.inner_requests[0]
    assert inner["max_tokens"] == 77
    assert "max_completion_tokens" not in inner


def test_tampered_response_fails_once_without_retries(gateway: FakeGateway):
    relay = FakeRelay(gateway, tamper=True)
    client = make_client(relay, max_retries=2)

    with pytest.raises(openai.InternalServerError) as excinfo:
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert excinfo.value.code == "tee_verification_failed"
    assert len(relay.requests) == 1  # not retried, so not billed again


# ---------------------------------------------------------------------------
# Errors from the relay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "error_type", "fragment"),
    [
        (401, {"detail": "Unauthorized"}, openai.AuthenticationError, "invalid or revoked"),
        (402, {"detail": "Insufficient credits"}, openai.APIStatusError, "out of credits"),
        (403, {"detail": "Account blocked"}, openai.PermissionDeniedError, "Account blocked"),
        (451, {"error": "moderation"}, openai.APIStatusError, "moderation"),
    ],
)
def test_relay_client_errors_are_typed_and_not_retried(gateway: FakeGateway, status, body, error_type, fragment):
    relay = FakeRelay(gateway, status=status, error_body=body)
    client = make_client(relay, max_retries=2)

    with pytest.raises(error_type) as excinfo:
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert excinfo.value.status_code == status
    assert fragment in excinfo.value.message
    assert len(relay.requests) == 1


def test_rate_limit_maps_to_rate_limit_error(gateway: FakeGateway):
    relay = FakeRelay(gateway, status=429, error_body={"error": "Rate limit exceeded: 40 per 1 minute"})
    client = make_client(relay, max_retries=0)

    with pytest.raises(openai.RateLimitError) as excinfo:
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert "40 per 1 minute" in excinfo.value.message


def test_tee_error_detail_is_readable(gateway: FakeGateway):
    body = {"detail": {"code": "tee_timeout", "message": "TEE OHTTP request timed out", "upstream": {"tee_host": "gw.test"}}}
    relay = FakeRelay(gateway, status=504, error_body=body)
    client = make_client(relay, max_retries=0)

    with pytest.raises(openai.InternalServerError) as excinfo:
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert excinfo.value.code == "tee_timeout"
    assert "TEE OHTTP request timed out" in excinfo.value.message
    assert "upstream" not in excinfo.value.message


def test_unsupported_endpoint_is_404_without_network(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    with pytest.raises(openai.NotFoundError):
        client.models.list()

    assert relay.requests == []


def test_n_above_one_is_rejected_locally(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, n=2)

    assert relay.requests == []


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_round_trip(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)

    assert stream_text(stream) == relay.reply
    outer = relay.requests[0]
    assert outer.headers["x-ohttp-stream"] == "true"
    assert outer.headers["accept"] == "message/ohttp-chunked-res"
    assert relay.inner_requests[0]["stream"] is True


def test_stream_reassembles_lines_and_utf8_split_across_chunks(gateway: FakeGateway):
    relay = FakeRelay(gateway, reply="héllo wörld 🌍 ✓ ünïcode done", plaintext_split=7, wire_split=5)
    client = make_client(relay)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)

    assert stream_text(stream) == relay.reply


def test_stream_tampered_raises_verification_error(gateway: FakeGateway):
    relay = FakeRelay(gateway, tamper=True)
    client = make_client(relay)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    with pytest.raises(TeeVerificationError):
        stream_text(stream)


def test_stream_verify_before_release_yields_nothing_unverified(gateway: FakeGateway):
    relay = FakeRelay(gateway, tamper=True)
    client = make_client(relay, verify_before_release=True)

    received = []
    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    with pytest.raises(TeeVerificationError):
        for chunk in stream:
            received.append(chunk)

    assert received == []


def test_stream_verify_before_release_delivers_verified_stream(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_client(relay, verify_before_release=True)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)

    assert stream_text(stream) == relay.reply


def test_stream_cut_short_raises_incomplete(gateway: FakeGateway):
    relay = FakeRelay(gateway, drop_last_chunks=2)  # loses the signed frame and [DONE]
    client = make_client(relay)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    with pytest.raises(StreamIncompleteError):
        stream_text(stream)


def test_stream_in_band_error_surfaces_as_api_error(gateway: FakeGateway):
    relay = FakeRelay(gateway, in_band_error="provider failed mid-stream")
    client = make_client(relay)

    stream = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    with pytest.raises(openai.APIError) as excinfo:
        stream_text(stream)

    assert "provider failed mid-stream" in str(excinfo.value)


def _stream_request() -> httpx.Request:
    return httpx.Request(
        "POST",
        BASE_URL + "/chat/completions",
        json={"model": "claude-haiku-4-5", "messages": MESSAGES, "stream": True},
        headers={"authorization": f"Bearer {API_KEY}"},
    )


def test_stream_body_is_single_pass(gateway: FakeGateway):
    # The OpenAI SDK may iterate response.stream again after [DONE] (to drain it);
    # a second pass must continue, not replay the encrypted stream from the start.
    transport = OHTTPTransport(tee=gateway.tee, transport=httpx.MockTransport(FakeRelay(gateway)))
    response = transport.handle_request(_stream_request())

    first = b"".join(response.stream)
    second = b"".join(response.stream)

    assert b"data: [DONE]" in first
    assert second == b""


async def test_async_stream_body_is_single_pass(gateway: FakeGateway):
    transport = AsyncOHTTPTransport(tee=gateway.tee, transport=httpx.MockTransport(FakeRelay(gateway)))
    response = await transport.handle_async_request(_stream_request())

    first = b"".join([chunk async for chunk in response.stream])
    second = b"".join([chunk async for chunk in response.stream])

    assert b"data: [DONE]" in first
    assert second == b""


# ---------------------------------------------------------------------------
# Registry: lazy lookup, and a fresh pick when chat-api says the TEE is gone
# ---------------------------------------------------------------------------


def _rpc_result(*gateways: FakeGateway) -> dict:
    encoded = abi_encode([TEE_INFO_ARRAY_TYPE], [[g.registry_tuple() for g in gateways]])
    return {"jsonrpc": "2.0", "id": 1, "result": "0x" + encoded.hex()}


def test_retired_tee_is_replaced_from_registry(gateway: FakeGateway, other_gateway: FakeGateway):
    rpc_calls: list[httpx.Request] = []

    def registry(request: httpx.Request) -> httpx.Response:
        rpc_calls.append(request)
        # First lookup lists the retiring TEE; the next one lists its replacement.
        return httpx.Response(200, json=_rpc_result(gateway if len(rpc_calls) == 1 else other_gateway))

    healthy = FakeRelay(other_gateway)
    seen_tee_ids: list[str] = []

    def relay(request: httpx.Request) -> httpx.Response:
        seen_tee_ids.append(request.headers["x-tee-id"])
        if request.headers["x-tee-id"] == gateway.tee_id:
            return httpx.Response(502, json={"detail": {"code": "tee_not_active", "message": "Selected TEE is not active", "upstream": {}}})
        return healthy(request)

    transport = OHTTPTransport(transport=httpx.MockTransport(relay), registry_transport=httpx.MockTransport(registry))
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=httpx.Client(transport=transport), max_retries=0)

    completion = client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert completion.choices[0].message.content == healthy.reply
    assert seen_tee_ids == [gateway.tee_id, other_gateway.tee_id]
    assert len(rpc_calls) == 2
    assert rpc_calls[0].url.host == "ogevmdevnet.opengradient.ai"
    rpc_body = json.loads(rpc_calls[0].content)
    assert rpc_body["method"] == "eth_call"
    assert rpc_body["params"][0]["to"] == "0x703cB174AEadB35D611858369B4b1111dC9Abda6"

    # The replacement is cached: no third registry read.
    client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)
    assert len(rpc_calls) == 2


def test_registry_outage_is_a_503(gateway: FakeGateway):
    def registry(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="rpc down")

    relay = FakeRelay(gateway)
    transport = OHTTPTransport(transport=httpx.MockTransport(relay), registry_transport=httpx.MockTransport(registry))
    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=httpx.Client(transport=transport), max_retries=0)

    with pytest.raises(openai.InternalServerError) as excinfo:
        client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "tee_registry_unavailable"
    assert relay.requests == []


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def make_async_client(relay: FakeRelay, **transport_kwargs) -> openai.AsyncOpenAI:
    transport = AsyncOHTTPTransport(tee=relay.gateway.tee, transport=httpx.MockTransport(relay), **transport_kwargs)
    return openai.AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=httpx.AsyncClient(transport=transport), max_retries=0)


async def test_async_non_stream(gateway: FakeGateway):
    relay = FakeRelay(gateway)
    client = make_async_client(relay)

    completion = await client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES)

    assert completion.choices[0].message.content == relay.reply
    assert relay.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    await client.close()


async def test_async_stream(gateway: FakeGateway):
    relay = FakeRelay(gateway, reply="async 🌍 stream", plaintext_split=3)
    client = make_async_client(relay)

    stream = await client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    text = ""
    async for chunk in stream:
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""

    assert text == relay.reply
    await client.close()


async def test_async_tampered_stream_raises(gateway: FakeGateway):
    relay = FakeRelay(gateway, tamper=True)
    client = make_async_client(relay)

    stream = await client.chat.completions.create(model="claude-haiku-4-5", messages=MESSAGES, stream=True)
    with pytest.raises(TeeVerificationError):
        async for _ in stream:
            pass
    await client.close()
