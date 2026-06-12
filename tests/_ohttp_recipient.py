"""Self-contained OHTTP *recipient* for tests.

A minimal, dependency-light port of the tee-gateway's recipient side
(``tee_gateway/ohttp.py``) using the same primitives this SDK already depends on
(pyhpke + cryptography). It lets the OHTTP wire-compatibility and end-to-end
verification tests run in CI **without** requiring an external tee-gateway
checkout. The real gateway is still cross-checked when available (see the
``real_tee_gateway`` fixture in ``conftest.py``).

If this ever diverges from the gateway, the cross-check test fails — so it can't
silently rot.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

KEY_CONFIG_ID = 0x01
KEM_ID_X25519 = 0x0020
KDF_ID_HKDF_SHA256 = 0x0001
AEAD_ID_CHACHA20_POLY1305 = 0x0003

_NK = 32
_NN = 12

_LABEL_REQUEST = b"message/bhttp request"
_LABEL_RESPONSE = b"message/bhttp response"
_LABEL_CHUNKED_RESPONSE = b"message/bhttp chunked response"

_SUITE = CipherSuite.new(KEMId.DHKEM_X25519_HKDF_SHA256, KDFId.HKDF_SHA256, AEADId.CHACHA20_POLY1305)


def _header_bytes(key_id: int = KEY_CONFIG_ID) -> bytes:
    return bytes([key_id]) + struct.pack(">HHH", KEM_ID_X25519, KDF_ID_HKDF_SHA256, AEAD_ID_CHACHA20_POLY1305)


def encode_varint(value: int) -> bytes:
    if value < (1 << 6):
        return bytes([value])
    if value < (1 << 14):
        return bytes([0x40 | (value >> 8), value & 0xFF])
    if value < (1 << 30):
        return struct.pack(">I", 0x80000000 | value)
    return struct.pack(">Q", 0xC000000000000000 | value)


def generate_keypair():
    pair = _SUITE.kem.derive_key_pair(os.urandom(32))
    return pair.private_key, pair.public_key.to_public_bytes()


@dataclass
class DecapsulatedRequest:
    plaintext: bytes
    response_key: bytes
    response_key_chunked: bytes
    enc: bytes


def decapsulate_request(private_key, encapsulated_request: bytes) -> DecapsulatedRequest:
    key_id = encapsulated_request[0]
    enc = encapsulated_request[7 : 7 + 32]
    aead_ct = encapsulated_request[7 + 32 :]
    info = _LABEL_REQUEST + b"\x00" + _header_bytes(key_id)
    recipient = _SUITE.create_recipient_context(enc, private_key, info=info)
    plaintext = recipient.open(aead_ct, aad=b"")
    export_len = max(_NN, _NK)
    return DecapsulatedRequest(
        plaintext=plaintext,
        response_key=recipient.export(_LABEL_RESPONSE, export_len),
        response_key_chunked=recipient.export(_LABEL_CHUNKED_RESPONSE, export_len),
        enc=enc,
    )


def _derive_response_keys(response_secret: bytes, enc: bytes, response_nonce: bytes) -> tuple[bytes, bytes]:
    h = hmac.HMAC(enc + response_nonce, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    aead_key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(prk)
    aead_nonce = HKDFExpand(algorithm=hashes.SHA256(), length=_NN, info=b"nonce").derive(prk)
    return aead_key, aead_nonce


def encapsulate_response(response_secret: bytes, enc: bytes, plaintext: bytes) -> bytes:
    response_nonce = os.urandom(max(_NN, _NK))
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, response_nonce)
    return response_nonce + ChaCha20Poly1305(aead_key).encrypt(aead_nonce, plaintext, b"")


class ChunkedResponseEncrypter:
    def __init__(self, response_secret: bytes, enc: bytes):
        self._response_nonce = os.urandom(max(_NN, _NK))
        self._aead_key, self._aead_nonce = _derive_response_keys(response_secret, enc, self._response_nonce)
        self._aead = ChaCha20Poly1305(self._aead_key)
        self._counter = 0

    def header(self) -> bytes:
        return self._response_nonce

    def encrypt_chunk(self, plaintext: bytes, is_final: bool) -> bytes:
        ctr = self._counter.to_bytes(_NN, "big")
        chunk_nonce = bytes(a ^ b for a, b in zip(self._aead_nonce, ctr))
        aad = b"final" if is_final else b""
        sealed = self._aead.encrypt(chunk_nonce, plaintext, aad)
        self._counter += 1
        length_prefix = encode_varint(0) if is_final else encode_varint(len(sealed))
        return length_prefix + sealed
