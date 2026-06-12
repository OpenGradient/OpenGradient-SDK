"""Verify the TEE response verifier against an independently-built gateway signature.

We reconstruct exactly what the gateway signs — RSA-PSS(salt=32, SHA256) over
``keccak256(inputHash || outputHash || uint256(ts))`` — and confirm
``verify_response`` accepts a good signature and rejects every tampered variant.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from eth_hash.auto import keccak

from opengradient.client import tee_verify as verify
from opengradient.client.tee_verify import build_inner_request


def _make_key():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return priv, pem


def _sign(priv, canonical, output_content, timestamp):
    request_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    input_hash = keccak(request_bytes)
    output_hash = keccak(output_content.encode("utf-8"))
    msg_hash = keccak(input_hash + output_hash + timestamp.to_bytes(32, "big"))
    sig = priv.sign(
        msg_hash,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )
    return {
        "tee_signature": base64.b64encode(sig).decode(),
        "tee_request_hash": input_hash.hex(),
        "tee_output_hash": output_hash.hex(),
        "tee_timestamp": timestamp,
    }


def _good_case():
    priv, pem = _make_key()
    tee_id = verify.tee_id_for_key(pem)
    _wire, canonical = build_inner_request(
        {"model": "gpt-4.1", "messages": [{"role": "user", "content": "Hello!"}]}
    )
    content = "Hi there!"
    response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        **_sign(priv, canonical, content, 1_700_000_000),
        "tee_id": tee_id,
    }
    return pem, tee_id, canonical, content, response


def test_valid_signature_verifies():
    pem, tee_id, canonical, content, response = _good_case()
    proof = verify.verify_response(
        canonical_request=canonical,
        response_body=response,
        response_content=content,
        signing_key_pem=pem,
        expected_tee_id=tee_id,
    )
    assert proof.tee_id == tee_id
    assert proof.timestamp == 1_700_000_000


def test_tampered_content_is_rejected():
    pem, _tee_id, canonical, _content, response = _good_case()
    with pytest.raises(verify.VerificationError, match="output hash"):
        verify.verify_response(
            canonical_request=canonical,
            response_body=response,
            response_content="Hi there! (tampered)",
            signing_key_pem=pem,
        )


def test_tampered_request_is_rejected():
    pem, _tee_id, _canonical, content, response = _good_case()
    other = build_inner_request({"model": "gpt-4.1", "messages": [{"role": "user", "content": "different"}]})[1]
    with pytest.raises(verify.VerificationError, match="request hash"):
        verify.verify_response(
            canonical_request=other,
            response_body=response,
            response_content=content,
            signing_key_pem=pem,
        )


def test_wrong_signing_key_is_rejected():
    _pem, _tee_id, canonical, content, response = _good_case()
    _other_priv, other_pem = _make_key()
    with pytest.raises(verify.VerificationError, match="tee_id mismatch"):
        verify.verify_response(
            canonical_request=canonical,
            response_body=response,
            response_content=content,
            signing_key_pem=other_pem,
        )


def test_tool_call_output_hashing():
    priv, pem = _make_key()
    tee_id = verify.tee_id_for_key(pem)
    _wire, canonical = build_inner_request({"model": "gpt-4.1", "messages": [{"role": "user", "content": "weather?"}]})
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get", "arguments": "{}"}}]
    output_content = json.dumps(tool_calls, sort_keys=True)
    response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
        **_sign(priv, canonical, output_content, 1_700_000_001),
        "tee_id": tee_id,
    }
    assert verify.response_content_for_hash(response) == output_content
    proof = verify.verify_response(
        canonical_request=canonical,
        response_body=response,
        response_content=verify.response_content_for_hash(response),
        signing_key_pem=pem,
    )
    assert proof.timestamp == 1_700_000_001


def test_build_inner_request_strips_attachment_bytes_from_hash_only():
    body = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    wire, canonical = build_inner_request(body)
    # Wire keeps the bytes so the model sees the image...
    assert wire["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
    # ...but the canonical (hashed) form commits only to type, not bytes.
    assert canonical["messages"][0]["content"][1] == {"type": "image_url"}
    assert canonical["temperature"] == 0.7
    assert canonical["max_tokens"] == 256


def test_non_list_tools_rejected():
    with pytest.raises(verify.UnsupportedRequestError, match="tools"):
        build_inner_request(
            {"model": "gpt-4.1", "messages": [{"role": "user", "content": "x"}], "tools": {"a": 1}}
        )


def test_non_dict_message_rejected():
    with pytest.raises(verify.UnsupportedRequestError, match="message must be"):
        build_inner_request({"model": "gpt-4.1", "messages": ["not a dict"]})


def test_tool_choice_preserved_on_wire_but_not_hashed():
    body = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": "auto",
    }
    wire, canonical = build_inner_request(body)
    # Forwarded to the gateway so caller intent isn't silently dropped...
    assert wire["tool_choice"] == "auto"
    # ...but excluded from the signed request hash (the gateway doesn't commit to it).
    assert "tool_choice" not in canonical
