"""Wire-compatibility round-trips for the client OHTTP encapsulation.

When a tee-gateway checkout is available (``OG_TEE_GATEWAY`` env var, or a sibling
``../tee-gateway``), these round-trip our client crypto against the *actual*
server recipient code, guaranteeing the two stay byte-compatible. The gateway's
``tee_gateway/ohttp.py`` only needs ``cryptography`` + ``pyhpke``, so importing it
standalone is cheap. The tests skip cleanly when no checkout is found.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from opengradient.client import tee_ohttp as cli


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
def server_ohttp():
    srv = _load_server_ohttp()
    if srv is None:
        pytest.skip("tee-gateway checkout not found (set OG_TEE_GATEWAY)")
    return srv


def test_request_and_single_shot_response(server_ohttp):
    priv, pub_raw = server_ohttp.generate_keypair()
    plaintext = b'{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}'

    enc_req = cli.encapsulate_request(pub_raw, plaintext)
    decap = server_ohttp.decapsulate_request(priv, enc_req.wire)
    assert decap.plaintext == plaintext
    assert decap.enc == enc_req.enc

    resp_pt = b'{"choices":[{"message":{"content":"hello"}}]}'
    sealed = server_ohttp.encapsulate_response(decap.response_key, decap.enc, resp_pt)
    assert cli.decrypt_response(enc_req.response_secret, enc_req.enc, sealed) == resp_pt


def test_chunked_response_whole_and_incremental(server_ohttp):
    priv, pub_raw = server_ohttp.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b"{}")
    decap = server_ohttp.decapsulate_request(priv, enc_req.wire)

    encr = server_ohttp.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
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


def test_truncated_chunked_stream_is_rejected(server_ohttp):
    priv, pub_raw = server_ohttp.generate_keypair()
    enc_req = cli.encapsulate_request(pub_raw, b"{}")
    decap = server_ohttp.decapsulate_request(priv, enc_req.wire)
    encr = server_ohttp.ChunkedResponseEncrypter(decap.response_key_chunked, decap.enc)
    wire = encr.header() + encr.encrypt_chunk(b"data: a\n\n", is_final=False)
    dec = cli.ChunkedResponseDecrypter(enc_req.chunked_response_secret, enc_req.enc)
    with pytest.raises(ValueError):
        dec.push(wire, done=True)


def test_rejects_wrong_size_public_key():
    with pytest.raises(ValueError):
        cli.encapsulate_request(b"too short", b"{}")
