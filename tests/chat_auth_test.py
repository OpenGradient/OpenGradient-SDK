"""Tests for the browser-based chat-account login (CLI-auth flow).

Drives ``login_chat_account`` in a background thread and plays the part of the
browser: parses the authorization URL it surfaces, sends the CORS preflight and
the JSON POST to the loopback listener, and asserts the parsed session.
"""

from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from opengradient.client.chat_auth import BUNDLE_TYPE, ChatAccountAuth, _parse_bundle, login_chat_account


def _bundle(access_token="tok_abc"):
    return {
        "type": BUNDLE_TYPE,
        "version": 1,
        "access_token": access_token,
        "refresh_token": "refresh_xyz",
        "token_type": "bearer",
        "expires_at": 1_700_000_000,
        "expires_in": 3600,
        "user": {"id": "u1", "email": "dev@example.com", "is_anonymous": False},
        "config": {
            "app_env": "production",
            "chat_api_base_url": "https://chat-api.opengradient.ai",
            "tee_registry_rpc_url": "https://rpc.example",
            "tee_registry_address": "0xabc",
            "tee_registry_tee_type": 0,
        },
    }


def test_parse_bundle_maps_fields_and_helpers():
    auth = _parse_bundle(_bundle())
    assert auth.access_token == "tok_abc"
    assert auth.email == "dev@example.com"
    assert auth.chat_api_base_url == "https://chat-api.opengradient.ai"
    assert auth.tee_registry_rpc_url == "https://rpc.example"
    assert auth.tee_registry_address == "0xabc"
    assert auth.auth_headers() == {"Authorization": "Bearer tok_abc"}


def test_parse_bundle_requires_access_token():
    bad = _bundle()
    del bad["access_token"]
    with pytest.raises(ValueError):
        _parse_bundle(bad)


def test_is_expired():
    assert _parse_bundle({**_bundle(), "expires_at": 1}).is_expired is True
    # No expiry known -> not treated as expired.
    assert _parse_bundle({**_bundle(), "expires_at": None}).is_expired is False


def test_login_end_to_end_via_loopback():
    """Full flow: the listener answers the CORS preflight, then accepts the POST."""
    captured_url: dict[str, str] = {}
    ready = threading.Event()
    result: dict[str, ChatAccountAuth] = {}
    error: dict[str, BaseException] = {}

    def on_ready(url: str) -> None:
        captured_url["url"] = url
        ready.set()

    def run_login() -> None:
        try:
            result["auth"] = login_chat_account(
                chat_app_url="https://chat.example",
                open_browser=False,
                timeout=10.0,
                on_ready=on_ready,
            )
        except BaseException as exc:  # noqa: BLE001 - surface to the main thread
            error["exc"] = exc

    thread = threading.Thread(target=run_login)
    thread.start()

    assert ready.wait(timeout=5.0), "login never became ready"
    parsed = urlparse(captured_url["url"])
    assert parsed.path == "/cli-auth"
    query = parse_qs(parsed.query)
    assert query["app_name"] == ["opengradient-sdk"]
    redirect_uri = query["redirect_uri"][0]

    target = urlparse(redirect_uri)
    assert target.hostname == "127.0.0.1"

    # 1) CORS preflight — the browser sends this before a JSON POST.
    preflight = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
    preflight.request(
        "OPTIONS",
        "/",
        headers={
            "Origin": "https://chat.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    pre_resp = preflight.getresponse()
    pre_resp.read()
    assert pre_resp.status == 204
    assert pre_resp.getheader("Access-Control-Allow-Origin") == "https://chat.example"
    assert "POST" in (pre_resp.getheader("Access-Control-Allow-Methods") or "")
    preflight.close()

    # 2) The actual session POST.
    body = json.dumps(_bundle()).encode()
    post = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
    post.request("POST", "/", body=body, headers={"Content-Type": "application/json", "Origin": "https://chat.example"})
    post_resp = post.getresponse()
    payload = post_resp.read()
    assert post_resp.status == 200
    assert json.loads(payload)["ok"] is True
    post.close()

    thread.join(timeout=5.0)
    assert not error, f"login raised: {error.get('exc')}"
    auth = result["auth"]
    assert auth.access_token == "tok_abc"
    assert auth.chat_api_base_url == "https://chat-api.opengradient.ai"
