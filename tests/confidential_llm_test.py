"""Tests for the high-level ConfidentialLLM (Oblivious HTTP) convenience.

Exercises the parts ConfidentialLLM adds on top of OhttpRelayClient — building
the request body from arguments, normalizing the gateway model id, and appending
the confidential-inference path — driving a full encrypt/sign/verify round trip
through the self-contained ``recipient`` fixture.
"""

from __future__ import annotations

import base64
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from eth_hash.auto import keccak

from opengradient.client.confidential_llm import (
    OHTTP_CHAT_ENDPOINT,
    ConfidentialLLM,
    _confidential_inference_url,
    _gateway_model,
)
from opengradient.client.tee_registry import OhttpConfig, TEEEndpoint
from opengradient.client.tee_verify import build_inner_request, tee_id_for_key


def _sign_fields(priv, canonical, output_content, ts):
    input_hash = keccak(json.dumps(canonical, sort_keys=True).encode())
    output_hash = keccak(output_content.encode())
    msg_hash = keccak(input_hash + output_hash + ts.to_bytes(32, "big"))
    sig = priv.sign(msg_hash, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    return {
        "tee_signature": base64.b64encode(sig).decode(),
        "tee_request_hash": input_hash.hex(),
        "tee_output_hash": output_hash.hex(),
        "tee_timestamp": ts,
    }


class _FakeResp:
    def __init__(self, content=b""):
        self.ok = True
        self.status_code = 200
        self.content = content


def _make_endpoint(recipient):
    hpke_priv, hpke_pub = recipient.generate_keypair()
    rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    der = rsa_priv.public_key().public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    pem = (
        rsa_priv.public_key()
        .public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    endpoint = TEEEndpoint(
        tee_id=tee_id_for_key(pem),
        endpoint="https://gw.example",
        tls_cert_der=b"",
        payment_address="0x" + "11" * 20,
        signing_public_key_der=der,
        ohttp_config=OhttpConfig(1, 0x0020, 0x0001, 0x0003, hpke_pub, b"kc", 0),
    )
    return endpoint, hpke_priv, rsa_priv


def _session_with(fake_post):
    return type("S", (), {"post": staticmethod(fake_post)})()


def test_gateway_model_strips_provider_prefix():
    assert _gateway_model("anthropic/claude-haiku-4-5") == "claude-haiku-4-5"
    assert _gateway_model("gpt-4.1") == "gpt-4.1"


def test_confidential_inference_url_appends_path_once():
    assert _confidential_inference_url("https://chat-api.example.com/") == "https://chat-api.example.com" + OHTTP_CHAT_ENDPOINT
    # A full endpoint URL is left as-is (no double append).
    full = "https://relay" + OHTTP_CHAT_ENDPOINT
    assert _confidential_inference_url(full) == full


def test_from_tee_chat_roundtrip_and_verify(recipient):
    endpoint, hpke_priv, rsa_priv = _make_endpoint(recipient)
    content = "hello from the enclave"

    # The gateway hashes the model id without the provider prefix, so the
    # ConfidentialLLM must have reduced "anthropic/claude-haiku-4-5" to
    # "claude-haiku-4-5" for both wire and canonical request to match.
    expected_body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    _wire, canonical = build_inner_request(expected_body)

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        # The relay base URL had the confidential-inference path appended.
        assert url == "https://relay" + OHTTP_CHAT_ENDPOINT
        assert headers["X-TEE-ID"] == endpoint.tee_id
        assert headers["Authorization"] == "Bearer t0ken"
        decap = recipient.decapsulate_request(hpke_priv, data)
        sent = json.loads(decap.plaintext.decode())
        assert sent["model"] == "claude-haiku-4-5"
        resp = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            **_sign_fields(rsa_priv, canonical, content, 1_700_000_000),
            "tee_id": endpoint.tee_id,
        }
        sealed = recipient.encapsulate_response(decap.response_key, decap.enc, json.dumps(resp).encode())
        return _FakeResp(content=sealed)

    client = ConfidentialLLM.from_tee(
        "https://relay",
        endpoint,
        auth_headers=lambda: {"Authorization": "Bearer t0ken"},
        session=_session_with(fake_post),
    )
    assert client.relay_url == "https://relay" + OHTTP_CHAT_ENDPOINT
    assert client.tee is endpoint

    result = client.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
    )
    assert result.content == content
    assert result.proof.tee_id == endpoint.tee_id
    assert result.proof.timestamp == 1_700_000_000
