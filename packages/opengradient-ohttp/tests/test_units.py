"""Unit tests for request canonicalization, registry decoding and relay-error parsing."""

from __future__ import annotations

import json

import httpx
import pytest
from _fakes import FakeGateway
from eth_abi import encode as abi_encode

from opengradient_ohttp import RegistryError, UnsupportedRequestError
from opengradient_ohttp._protocol import parse_relay_error, relay_url_from
from opengradient_ohttp._registry import TEE_INFO_ARRAY_TYPE, get_active_tees_call, parse_active_tees
from opengradient_ohttp._verify import build_inner_request

# ---------------------------------------------------------------------------
# build_inner_request
# ---------------------------------------------------------------------------


def test_canonical_matches_gateway_shape():
    wire, canonical = build_inner_request(
        {
            "model": "anthropic/claude-haiku-4-5",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "file", "file": {"filename": "a.pdf", "file_data": "QUJD"}}]},
            ],
            "temperature": 1,
            "max_completion_tokens": 64,
            "reasoning_effort": "low",
            "tool_choice": "auto",
            "stream": True,
        }
    )
    assert canonical == {
        "model": "claude-haiku-4-5",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "file", "filename": "a.pdf"}]},
        ],
        "temperature": 1.0,
        "max_tokens": 64,
        "reasoning_effort": "low",
    }
    # The wire keeps the attachment bytes and the unsigned tool_choice; stream is added by the caller.
    assert wire["messages"][1]["content"][1]["file"]["file_data"] == "QUJD"
    assert wire["tool_choice"] == "auto"
    assert "stream" not in wire
    # Python's json.dumps(sort_keys=True), as the gateway hashes it: 1 -> 1.0.
    assert '"temperature": 1.0' in json.dumps(canonical, sort_keys=True)


def test_explicit_max_tokens_wins():
    _, canonical = build_inner_request({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5, "max_completion_tokens": 9})
    assert canonical["max_tokens"] == 5


def test_image_fields_are_signed():
    _, canonical = build_inner_request(
        {"model": "qwen-image", "messages": [{"role": "user", "content": "a cat"}], "aspect_ratio": "16:9", "resolution": "2k"}
    )
    assert canonical["aspect_ratio"] == "16:9"
    assert canonical["resolution"] == "2k"


def test_unsigned_fields_warn():
    with pytest.warns(UserWarning, match="seed, top_p"):
        wire, canonical = build_inner_request({"model": "m", "messages": [{"role": "user", "content": "x"}], "top_p": 0.5, "seed": 1})
    assert "top_p" not in wire and "seed" not in wire


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "x"}]},
        {"model": "m", "messages": []},
        {"model": "m", "messages": [{"role": "wizard", "content": "x"}]},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 3},
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": {"not": "a list"}},
    ],
)
def test_bad_requests_are_rejected(body):
    with pytest.raises(UnsupportedRequestError):
        build_inner_request(body)


# ---------------------------------------------------------------------------
# Registry decoding
# ---------------------------------------------------------------------------


def _rpc(tuples: list) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": "0x" + abi_encode([TEE_INFO_ARRAY_TYPE], [tuples]).hex()}


def test_get_active_tees_call_payload():
    payload = get_active_tees_call("0x703cB174AEadB35D611858369B4b1111dC9Abda6")
    # keccak("getActiveTEEs(uint8)")[:4] followed by uint8 0 as one 32-byte word.
    data = payload["params"][0]["data"]
    assert data.startswith("0x") and len(data) == 2 + 8 + 64
    assert data.endswith("0" * 64)
    assert payload["params"][1] == "latest"


def test_parse_active_tees_keeps_only_usable(gateway: FakeGateway, other_gateway: FakeGateway):
    wrong_suite = FakeGateway(endpoint="https://wrong-suite.test")
    short_key = FakeGateway(endpoint="https://short-key.test")
    rpc = _rpc(
        [
            gateway.registry_tuple(),
            other_gateway.registry_tuple(),
            wrong_suite.registry_tuple(kem_id=0x0010),
            short_key.registry_tuple(hpke_public=b"\x01" * 16),
        ]
    )

    targets = parse_active_tees(rpc, excluded_tee_ids=[other_gateway.tee_id.upper().replace("0X", "0x")])

    assert [t.tee_id for t in targets] == [gateway.tee_id]
    target = targets[0]
    assert target.endpoint == gateway.endpoint
    assert target.ohttp_public_key == gateway.hpke_public
    assert target.signing_key_pem == gateway.tee.signing_key_pem
    assert target.key_id == 1


def test_parse_active_tees_rpc_error():
    with pytest.raises(RegistryError):
        parse_active_tees({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}})


# ---------------------------------------------------------------------------
# Relay errors and URLs
# ---------------------------------------------------------------------------


def test_parse_relay_error_shapes():
    tee = parse_relay_error(502, json.dumps({"detail": {"code": "tee_not_active", "message": "gone", "upstream": {}}}).encode(), httpx.Headers())
    assert (tee.message, tee.code) == ("gone", "tee_not_active")

    credits = parse_relay_error(402, b'{"detail":"Insufficient credits"}', httpx.Headers())
    assert "out of credits" in credits.message and "Insufficient credits" in credits.message

    limited = parse_relay_error(429, b'{"error":"Rate limit exceeded: 40 per 1 minute"}', httpx.Headers({"retry-after": "12"}))
    assert limited.retry_after == "12" and "40 per 1 minute" in limited.message

    plain = parse_relay_error(500, b"Internal Server Error", httpx.Headers())
    assert plain.message == "Internal Server Error" and plain.code is None


def test_relay_url_from():
    assert relay_url_from("https://chat-api.memchat.io") == "https://chat-api.memchat.io/api/v1/developer/ohttp"
    assert relay_url_from("https://chat-api.memchat.io/") == "https://chat-api.memchat.io/api/v1/developer/ohttp"
    assert relay_url_from("http://localhost:8000/api/v1/developer/ohttp/") == "http://localhost:8000/api/v1/developer/ohttp"
