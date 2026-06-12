"""End-to-end test for the high-level OhttpRelayClient.

Simulates the full path — client encrypts -> (fake relay+gateway) decapsulates,
signs, seals -> client decrypts + verifies — for both single-shot and streaming,
using the real tee-gateway recipient crypto. Skips if no tee-gateway checkout.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from eth_hash.auto import keccak

from opengradient.client.tee_ohttp_client import OhttpRelayClient
from opengradient.client.tee_registry import OhttpConfig, TEEEndpoint
from opengradient.client.tee_verify import build_inner_request, tee_id_for_key


def _load_server_ohttp():
    override = os.getenv("OG_TEE_GATEWAY")
    candidates = [Path(override)] if override else []
    candidates.append(Path(__file__).resolve().parents[2] / "tee-gateway")
    for root in candidates:
        if (root / "tee_gateway" / "ohttp.py").exists():
            sys.path.insert(0, str(root))
            import tee_gateway.ohttp as srv

            return srv
    return None


@pytest.fixture(scope="module")
def srv():
    s = _load_server_ohttp()
    if s is None:
        pytest.skip("tee-gateway checkout not found (set OG_TEE_GATEWAY)")
    return s


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
    def __init__(self, content=b"", chunks=None):
        self.ok = True
        self.status_code = 200
        self.content = content
        self._chunks = chunks or []

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


def _make_endpoint(srv):
    hpke_priv, hpke_pub = srv.generate_keypair()
    rsa_priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    der = rsa_priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pem = rsa_priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    endpoint = TEEEndpoint(
        tee_id=tee_id_for_key(pem),
        endpoint="https://gw.example",
        tls_cert_der=b"",
        payment_address="0x" + "11" * 20,
        signing_public_key_der=der,
        ohttp_config=OhttpConfig(1, 0x0020, 0x0001, 0x0003, hpke_pub, b"kc", 0),
    )
    return endpoint, hpke_priv, rsa_priv


def test_single_shot_roundtrip_and_verify(srv):
    endpoint, hpke_priv, rsa_priv = _make_endpoint(srv)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]}
    _wire, canonical = build_inner_request(body)
    content = "hello from the enclave"

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        assert headers["X-TEE-ID"] == endpoint.tee_id
        assert headers["Authorization"] == "Bearer t0ken"
        decap = srv.decapsulate_request(hpke_priv, data)
        assert json.loads(decap.plaintext.decode())["model"] == "gpt-4.1"
        resp = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            **_sign_fields(rsa_priv, canonical, content, 1_700_000_000),
            "tee_id": endpoint.tee_id,
        }
        sealed = srv.encapsulate_response(decap.response_key, decap.enc, json.dumps(resp).encode())
        return _FakeResp(content=sealed)

    class _Sess:
        post = staticmethod(fake_post)

    client = OhttpRelayClient(
        "https://relay/api/v1/chat/ohttp",
        endpoint,
        auth_headers=lambda: {"Authorization": "Bearer t0ken"},
        session=_Sess(),
    )
    result = client.chat_completion(body)
    assert result.content == content
    assert result.proof.tee_id == endpoint.tee_id
    assert result.proof.timestamp == 1_700_000_000


def test_streaming_roundtrip_and_verify(srv):
    endpoint, hpke_priv, rsa_priv = _make_endpoint(srv)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]}
    _wire, canonical = build_inner_request(body)
    full = "Hello world"

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        assert stream is True
        assert headers["X-OHTTP-Stream"] == "true"
        decap = srv.decapsulate_request(hpke_priv, data)
        encr = srv.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
        wire = encr.header()
        wire += encr.encrypt_chunk(b'data: {"choices":[{"delta":{"content":"Hello "},"index":0}]}\n\n', is_final=False)
        wire += encr.encrypt_chunk(b'data: {"choices":[{"delta":{"content":"world"},"index":0}]}\n\n', is_final=False)
        final = {
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            **_sign_fields(rsa_priv, canonical, full, 1_700_000_001),
            "tee_id": endpoint.tee_id,
        }
        wire += encr.encrypt_chunk(f"data: {json.dumps(final)}\n\n".encode(), is_final=True)
        # Deliver as a couple of network chunks to exercise buffering.
        mid = len(wire) // 2
        return _FakeResp(chunks=[wire[:mid], wire[mid:]])

    class _Sess:
        post = staticmethod(fake_post)

    client = OhttpRelayClient("https://relay/api/v1/chat/ohttp", endpoint, session=_Sess())
    result = client.stream_chat_completion(body)
    assert result.content == full
    assert result.proof.timestamp == 1_700_000_001
    assert result.stream_frames is not None and len(result.stream_frames) == 3
