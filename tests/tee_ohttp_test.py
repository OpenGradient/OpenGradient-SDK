"""OHTTP client wire-compatibility round-trips.

These run in CI against a self-contained in-repo recipient (the ``recipient``
fixture), so the encapsulation/decryption code is always exercised. When a real
tee-gateway checkout is present, ``test_cross_check_against_real_gateway`` also
round-trips against the actual server code to catch any drift.
"""

from __future__ import annotations

import pytest

from opengradient.client import tee_ohttp as cli


def test_request_and_single_shot_response(recipient):
    priv, pub_raw = recipient.generate_keypair()
    plaintext = b'{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}'

    enc_req = cli.encapsulate_request(pub_raw, plaintext)
    decap = recipient.decapsulate_request(priv, enc_req.wire)
    assert decap.plaintext == plaintext
    assert decap.enc == enc_req.enc

    resp_pt = b'{"choices":[{"message":{"content":"hello"}}]}'
    sealed = recipient.encapsulate_response(decap.response_key, decap.enc, resp_pt)
    assert cli.decrypt_response(enc_req.response_secret, enc_req.enc, sealed) == resp_pt


def test_chunked_response_whole_and_incremental(recipient):
    priv, pub_raw = recipient.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b"{}")
    decap = recipient.decapsulate_request(priv, enc_req.wire)

    encr = recipient.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
    wire = encr.header()
    frames = [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]
    wire += encr.encrypt_chunk(frames[0], is_final=False)
    wire += encr.encrypt_chunk(frames[1], is_final=False)
    wire += encr.encrypt_chunk(frames[2], is_final=True)

    dec = cli.ChunkedResponseDecrypter(enc_req.chunked_response_secret, enc_req.enc)
    assert dec.push(wire, done=True) == frames

    dec2 = cli.ChunkedResponseDecrypter(enc_req.chunked_response_secret, enc_req.enc)
    out: list[bytes] = []
    for i, b in enumerate(wire):
        out += dec2.push(bytes([b]), done=(i == len(wire) - 1))
    assert out == frames


def test_truncated_chunked_stream_is_rejected(recipient):
    priv, pub_raw = recipient.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b"{}")
    decap = recipient.decapsulate_request(priv, enc_req.wire)
    encr = recipient.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
    wire = encr.header() + encr.encrypt_chunk(b"data: a\n\n", is_final=False)
    dec = cli.ChunkedResponseDecrypter(enc_req.chunked_response_secret, enc_req.enc)
    with pytest.raises(ValueError):
        dec.push(wire, done=True)


def test_rejects_wrong_size_public_key():
    with pytest.raises(ValueError):
        cli.encapsulate_request(b"too short", b"{}")


def test_rejects_unsupported_suite():
    with pytest.raises(ValueError, match="unsupported HPKE suite"):
        cli.encapsulate_request(b"\x00" * 32, b"{}", aead_id=0x0001)


def test_custom_key_id_round_trips(recipient):
    # A TEE that rotated to key_id=0x07 must still decapsulate (the id is carried
    # in the header and bound into the HPKE info string on both sides).
    priv, pub_raw = recipient.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b"{}", key_id=0x07)
    assert enc_req.wire[0] == 0x07
    decap = recipient.decapsulate_request(priv, enc_req.wire)
    assert decap.plaintext == b"{}"


def test_cross_check_against_real_gateway(real_tee_gateway):
    """When a tee-gateway checkout is present, confirm we're byte-compatible with it."""
    priv, pub_raw = real_tee_gateway.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b'{"ping":1}')
    decap = real_tee_gateway.decapsulate_request(priv, enc_req.wire)
    assert decap.plaintext == b'{"ping":1}'
    sealed = real_tee_gateway.encapsulate_response(decap.response_key, decap.enc, b'{"pong":1}')
    assert cli.decrypt_response(enc_req.response_secret, enc_req.enc, sealed) == b'{"pong":1}'
