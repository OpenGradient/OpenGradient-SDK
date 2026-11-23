"""Client-side Oblivious HTTP (RFC 9458) encapsulation.

Ported unchanged in behaviour from ``opengradient.client.tee_ohttp``: the
sender side of what the tee-gateway implements as the recipient and the
DreamFort dashboard implements in the browser (``src/lib/ohttp-chat.ts``).

Fixed ciphersuite, which must match the TEE's on-chain OHTTP config:

  - KEM:  DHKEM(X25519, HKDF-SHA256) (0x0020)
  - KDF:  HKDF-SHA256                (0x0001)
  - AEAD: ChaCha20-Poly1305          (0x0003)

Wire formats:
    Request:   header(7) || enc(32) || AEAD ciphertext
    Response:  response_nonce(32) || AEAD ciphertext                 (single-shot)
    Chunked:   response_nonce(32) || (varint(len)||sealed)+ || varint(0)||final
"""

from __future__ import annotations

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
SUPPORTED_SUITE = (KEM_ID_X25519, KDF_ID_HKDF_SHA256, AEAD_ID_CHACHA20_POLY1305)

_NK = 32  # AEAD key length / response_nonce length (== max(Nn, Nk))
_NN = 12  # AEAD nonce length

_LABEL_REQUEST = b"message/bhttp request"
_LABEL_RESPONSE = b"message/bhttp response"
_LABEL_CHUNKED_RESPONSE = b"message/bhttp chunked response"

_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256,
    KDFId.HKDF_SHA256,
    AEADId.CHACHA20_POLY1305,
)


def _header_bytes(key_id: int) -> bytes:
    return bytes([key_id]) + struct.pack(">HHH", KEM_ID_X25519, KDF_ID_HKDF_SHA256, AEAD_ID_CHACHA20_POLY1305)


@dataclass(frozen=True)
class EncapsulatedRequest:
    """An HPKE-sealed request plus the secrets needed to open its response."""

    wire: bytes
    enc: bytes
    response_secret: bytes
    chunked_response_secret: bytes


def encapsulate_request(public_key_raw: bytes, plaintext: bytes, *, key_id: int = KEY_CONFIG_ID) -> EncapsulatedRequest:
    """HPKE-seal ``plaintext`` to a TEE's raw 32-byte X25519 public key.

    Raises:
        ValueError: If ``public_key_raw`` is not 32 bytes.
    """
    if len(public_key_raw) != 32:
        raise ValueError("X25519 public key must be 32 bytes")

    pkr = _SUITE.kem.deserialize_public_key(public_key_raw)
    info = _LABEL_REQUEST + b"\x00" + _header_bytes(key_id)
    enc, sender = _SUITE.create_sender_context(pkr, info=info)

    ciphertext = sender.seal(plaintext, aad=b"")
    export_len = max(_NN, _NK)
    return EncapsulatedRequest(
        wire=_header_bytes(key_id) + bytes(enc) + ciphertext,
        enc=bytes(enc),
        response_secret=sender.export(_LABEL_RESPONSE, export_len),
        chunked_response_secret=sender.export(_LABEL_CHUNKED_RESPONSE, export_len),
    )


def _derive_response_keys(response_secret: bytes, enc: bytes, response_nonce: bytes) -> tuple[bytes, bytes]:
    """HKDF-Extract(salt=enc||response_nonce, ikm=response_secret), then Expand "key" / "nonce"."""
    h = hmac.HMAC(enc + response_nonce, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    aead_key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(prk)
    aead_nonce = HKDFExpand(algorithm=hashes.SHA256(), length=_NN, info=b"nonce").derive(prk)
    return aead_key, aead_nonce


def decrypt_response(response_secret: bytes, enc: bytes, sealed: bytes) -> bytes:
    """Open a single-shot OHTTP response.

    Raises:
        ValueError: If the response is too short to be well-formed.
        cryptography.exceptions.InvalidTag: If the ciphertext was tampered with.
    """
    if len(sealed) <= _NK:
        raise ValueError("malformed OHTTP response")
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, sealed[:_NK])
    return ChaCha20Poly1305(aead_key).decrypt(aead_nonce, sealed[_NK:], b"")


def _decode_varint(buf: bytearray, offset: int) -> tuple[int, int] | None:
    """Parse one QUIC varint; ``(value, new_offset)`` or ``None`` if more bytes are needed."""
    if offset >= len(buf):
        return None
    first = buf[offset]
    length = 1 << (first >> 6)
    if offset + length > len(buf):
        return None
    value = first & 0x3F
    for i in range(1, length):
        value = (value << 8) | buf[offset + i]
    return value, offset + length


class ChunkedResponseDecrypter:
    """Incrementally open a chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08).

    Feed raw bytes as they arrive; returns decrypted plaintext chunks. The last
    chunk carries AAD ``b"final"``; reaching end-of-stream without it is
    reported as truncation.
    """

    def __init__(self, response_secret: bytes, enc: bytes):
        self._response_secret = response_secret
        self._enc = enc
        self._buffer = bytearray()
        self._aead: ChaCha20Poly1305 | None = None
        self._nonce: bytes | None = None
        self._counter = 0
        self._saw_final = False

    def push(self, chunk: bytes | None, done: bool) -> list[bytes]:
        """Feed bytes; return any newly decrypted plaintext chunks.

        Raises:
            ValueError: On a malformed or truncated stream.
            cryptography.exceptions.InvalidTag: If a chunk was tampered with.
        """
        if chunk:
            if self._saw_final:
                raise ValueError("data after the final OHTTP chunk")
            self._buffer.extend(chunk)

        if self._aead is None or self._nonce is None:
            if len(self._buffer) < _NK:
                if done:
                    raise ValueError("chunked OHTTP response ended before its header")
                return []
            key, self._nonce = _derive_response_keys(self._response_secret, self._enc, bytes(self._buffer[:_NK]))
            self._aead = ChaCha20Poly1305(key)
            del self._buffer[:_NK]

        out: list[bytes] = []
        while self._buffer:
            frame = _decode_varint(self._buffer, 0)
            if frame is None:
                if done:
                    raise ValueError("truncated chunk length in chunked OHTTP response")
                break
            sealed_len, offset = frame

            if sealed_len == 0:
                # A zero-length prefix marks the final chunk, which runs to end-of-stream.
                if not done:
                    break
                out.append(self._open(bytes(self._buffer[offset:]), is_final=True))
                self._buffer.clear()
                self._saw_final = True
                break

            if len(self._buffer) < offset + sealed_len:
                if done:
                    raise ValueError("truncated chunk in chunked OHTTP response")
                break

            out.append(self._open(bytes(self._buffer[offset : offset + sealed_len]), is_final=False))
            del self._buffer[: offset + sealed_len]

        if done and not self._saw_final:
            raise ValueError("chunked OHTTP response ended without its final chunk")
        return out

    def _open(self, ciphertext: bytes, *, is_final: bool) -> bytes:
        assert self._aead is not None and self._nonce is not None
        ctr = self._counter.to_bytes(_NN, "big")
        chunk_nonce = bytes(a ^ b for a, b in zip(self._nonce, ctr))
        plaintext = self._aead.decrypt(chunk_nonce, ciphertext, b"final" if is_final else b"")
        self._counter += 1
        return plaintext
