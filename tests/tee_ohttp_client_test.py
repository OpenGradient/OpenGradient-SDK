"""End-to-end tests for the high-level OhttpRelayClient.

Simulates the full path — client encrypts -> (fake relay + recipient) decapsulates,
signs, seals -> client decrypts + verifies — for single-shot, streaming text, and
streaming tool calls, using the self-contained ``recipient`` fixture so it runs in
CI without an external checkout.
"""

from __future__ import annotations

import base64
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from eth_hash.auto import keccak

from opengradient.client.tee_ohttp_client import OhttpRelayClient
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
    def __init__(self, content=b"", chunks=None):
        self.ok = True
        self.status_code = 200
        self.content = content
        self._chunks = chunks or []

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


def _make_endpoint(recipient):
    hpke_priv, hpke_pub = recipient.generate_keypair()
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


def _session_with(fake_post):
    return type("S", (), {"post": staticmethod(fake_post)})()


def test_single_shot_roundtrip_and_verify(recipient):
    endpoint, hpke_priv, rsa_priv = _make_endpoint(recipient)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]}
    _wire, canonical = build_inner_request(body)
    content = "hello from the enclave"

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        assert headers["X-TEE-ID"] == endpoint.tee_id
        assert headers["Authorization"] == "Bearer t0ken"
        decap = recipient.decapsulate_request(hpke_priv, data)
        assert json.loads(decap.plaintext.decode())["model"] == "gpt-4.1"
        resp = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            **_sign_fields(rsa_priv, canonical, content, 1_700_000_000),
            "tee_id": endpoint.tee_id,
        }
        sealed = recipient.encapsulate_response(decap.response_key, decap.enc, json.dumps(resp).encode())
        return _FakeResp(content=sealed)

    client = OhttpRelayClient(
        "https://relay/api/v1/chat/ohttp",
        endpoint,
        auth_headers=lambda: {"Authorization": "Bearer t0ken"},
        session=_session_with(fake_post),
    )
    result = client.chat_completion(body)
    assert result.content == content
    assert result.proof.tee_id == endpoint.tee_id
    assert result.proof.timestamp == 1_700_000_000


def test_streaming_roundtrip_and_verify(recipient):
    endpoint, hpke_priv, rsa_priv = _make_endpoint(recipient)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]}
    _wire, canonical = build_inner_request(body)
    full = "Hello world"

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        assert stream is True and headers["X-OHTTP-Stream"] == "true"
        decap = recipient.decapsulate_request(hpke_priv, data)
        encr = recipient.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
        wire = encr.header()
        wire += encr.encrypt_chunk(b'data: {"choices":[{"delta":{"content":"Hello "},"index":0}]}\n\n', is_final=False)
        wire += encr.encrypt_chunk(b'data: {"choices":[{"delta":{"content":"world"},"index":0}]}\n\n', is_final=False)
        final = {
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            **_sign_fields(rsa_priv, canonical, full, 1_700_000_001),
            "tee_id": endpoint.tee_id,
        }
        wire += encr.encrypt_chunk(f"data: {json.dumps(final)}\n\n".encode(), is_final=True)
        mid = len(wire) // 2
        return _FakeResp(chunks=[wire[:mid], wire[mid:]])

    client = OhttpRelayClient("https://relay/api/v1/chat/ohttp", endpoint, session=_session_with(fake_post))
    result = client.stream_chat_completion(body)
    assert result.content == full
    assert result.proof.timestamp == 1_700_000_001
    assert result.stream_frames is not None and len(result.stream_frames) == 3


def test_streaming_tool_calls_verify(recipient):
    """Tool calls streamed as deltas must reconstruct the gateway's signed output."""
    endpoint, hpke_priv, rsa_priv = _make_endpoint(recipient)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "weather?"}]}
    _wire, canonical = build_inner_request(body)
    # What the gateway signs for a tool-call response:
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}}]
    signed_output = json.dumps(tool_calls, sort_keys=True)

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        decap = recipient.decapsulate_request(hpke_priv, data)
        encr = recipient.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
        wire = encr.header()

        # Stream the tool call as fragments (id+name first, then argument chunks).
        def _tc(frag):
            return {"choices": [{"delta": {"tool_calls": [frag]}, "index": 0}]}

        f1 = _tc({"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}})
        f2 = _tc({"index": 0, "function": {"arguments": '{"city":'}})
        f3 = _tc({"index": 0, "function": {"arguments": '"NYC"}'}})
        wire += encr.encrypt_chunk(f"data: {json.dumps(f1)}\n\n".encode(), is_final=False)
        wire += encr.encrypt_chunk(f"data: {json.dumps(f2)}\n\n".encode(), is_final=False)
        wire += encr.encrypt_chunk(f"data: {json.dumps(f3)}\n\n".encode(), is_final=False)
        final = {
            "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}],
            **_sign_fields(rsa_priv, canonical, signed_output, 1_700_000_002),
            "tee_id": endpoint.tee_id,
        }
        wire += encr.encrypt_chunk(f"data: {json.dumps(final)}\n\n".encode(), is_final=True)
        return _FakeResp(chunks=[wire])

    client = OhttpRelayClient("https://relay/api/v1/chat/ohttp", endpoint, session=_session_with(fake_post))
    result = client.stream_chat_completion(body)
    assert result.content == signed_output
    assert result.proof.timestamp == 1_700_000_002


def test_malformed_stream_raises_verification_error(recipient):
    from opengradient.client import VerificationError

    endpoint, hpke_priv, _rsa = _make_endpoint(recipient)
    body = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}]}

    def fake_post(url, data=None, headers=None, timeout=None, stream=False):
        decap = recipient.decapsulate_request(hpke_priv, data)
        encr = recipient.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
        # No final marker -> truncated stream.
        wire = encr.header() + encr.encrypt_chunk(b"data: x\n\n", is_final=False)
        return _FakeResp(chunks=[wire])

    client = OhttpRelayClient("https://relay/api/v1/chat/ohttp", endpoint, session=_session_with(fake_post))
    try:
        client.stream_chat_completion(body)
        assert False, "expected VerificationError"
    except VerificationError:
        pass
