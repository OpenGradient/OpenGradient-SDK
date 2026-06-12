"""Client-side Oblivious HTTP (RFC 9458) encapsulation for anonymous TEE inference.

This is the *sender* side of the construction the tee-gateway implements on the
recipient side and the chat-app implements in the browser. Using it, a client can
HPKE-encrypt an inference request to a TEE's published X25519 key, send it through
an untrusted relay, and decrypt the (single-shot or chunked-streaming) response —
the relay only ever sees ciphertext.

The ciphersuite is fixed and must match the enclave and the on-chain
`opengradient.client.tee_registry.OhttpConfig`:

  - KEM:  DHKEM(X25519, HKDF-SHA256) (0x0020)
  - KDF:  HKDF-SHA256                (0x0001)
  - AEAD: ChaCha20-Poly1305          (0x0003)

We use `pyhpke` for the HPKE sender context (the same library the gateway uses on
the recipient side, guaranteeing wire compatibility) and derive the response keys
with the same manual HKDF the gateway uses, so responses decrypt byte-for-byte.

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

# RFC 9180 / 9458 algorithm identifiers (fixed suite).
KEY_CONFIG_ID = 0x01
KEM_ID_X25519 = 0x0020
KDF_ID_HKDF_SHA256 = 0x0001
AEAD_ID_CHACHA20_POLY1305 = 0x0003

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


def _header_bytes() -> bytes:
    return bytes([KEY_CONFIG_ID]) + struct.pack(">HHH", KEM_ID_X25519, KDF_ID_HKDF_SHA256, AEAD_ID_CHACHA20_POLY1305)


@dataclass
class EncapsulatedRequest:
    """An HPKE-sealed request plus the secrets needed to open its response.

    Attributes:
        wire: The bytes to send to the relay (header || enc || ciphertext).
        enc: Our ephemeral X25519 public key; salts the response keying.
        response_secret: Exported secret for a single-shot response.
        chunked_response_secret: Exported secret for a chunked-streaming response.
    """

    wire: bytes
    enc: bytes
    response_secret: bytes
    chunked_response_secret: bytes


def encapsulate_request(public_key_raw: bytes, plaintext: bytes) -> EncapsulatedRequest:
    """HPKE-seal ``plaintext`` to a TEE's raw X25519 public key.

    Args:
        public_key_raw: The 32-byte raw X25519 public key from the TEE's OHTTP
            config (``OhttpConfig.public_key``).
        plaintext: The inner request body (typically a UTF-8 JSON chat request).

    Returns:
        An `EncapsulatedRequest` ready to send to a relay.

    Raises:
        ValueError: If ``public_key_raw`` is not 32 bytes.
    """
    if len(public_key_raw) != 32:
        raise ValueError("X25519 public key must be 32 bytes")

    pkr = _SUITE.kem.deserialize_public_key(public_key_raw)
    info = _LABEL_REQUEST + b"\x00" + _header_bytes()
    enc, sender = _SUITE.create_sender_context(pkr, info=info)

    ciphertext = sender.seal(plaintext, aad=b"")
    wire = _header_bytes() + bytes(enc) + ciphertext

    export_len = max(_NN, _NK)
    return EncapsulatedRequest(
        wire=wire,
        enc=bytes(enc),
        response_secret=sender.export(_LABEL_RESPONSE, export_len),
        chunked_response_secret=sender.export(_LABEL_CHUNKED_RESPONSE, export_len),
    )


def _derive_response_keys(response_secret: bytes, enc: bytes, response_nonce: bytes) -> tuple[bytes, bytes]:
    """HKDF-Extract(salt=enc||response_nonce, ikm=response_secret) then Expand.

    Byte-identical to the gateway's response-key derivation, so both single-shot
    and chunked responses decrypt correctly.
    """
    h = hmac.HMAC(enc + response_nonce, hashes.SHA256())
    h.update(response_secret)
    prk = h.finalize()
    aead_key = HKDFExpand(algorithm=hashes.SHA256(), length=_NK, info=b"key").derive(prk)
    aead_nonce = HKDFExpand(algorithm=hashes.SHA256(), length=_NN, info=b"nonce").derive(prk)
    return aead_key, aead_nonce


def decrypt_response(response_secret: bytes, enc: bytes, sealed: bytes) -> bytes:
    """Decrypt a single-shot OHTTP response (RFC 9458 §4.5).

    Args:
        response_secret: ``EncapsulatedRequest.response_secret``.
        enc: ``EncapsulatedRequest.enc``.
        sealed: The full response body from the relay.

    Returns:
        The decrypted inner response bytes.

    Raises:
        ValueError: If the response is too short to be well-formed.
    """
    if len(sealed) <= _NK:
        raise ValueError("malformed OHTTP response")
    response_nonce = sealed[:_NK]
    ciphertext = sealed[_NK:]
    aead_key, aead_nonce = _derive_response_keys(response_secret, enc, response_nonce)
    return ChaCha20Poly1305(aead_key).decrypt(aead_nonce, ciphertext, b"")


def _decode_varint(buf: bytes, offset: int) -> tuple[int, int] | None:
    """Parse one QUIC varint; returns ``(value, new_offset)`` or ``None`` if more bytes are needed."""
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
    """Incrementally decrypt a chunked OHTTP response stream (draft-ietf-ohai-chunked-ohttp-08).

    Feed it raw response bytes as they arrive; it yields decrypted plaintext
    frames (typically the inner SSE ``data:`` events). The final frame carries
    AAD=b"final"; its absence at end-of-stream is treated as truncation, so a
    network attacker cannot silently cut a stream short.
    """

    def __init__(self, response_secret: bytes, enc: bytes):
        self._response_secret = response_secret
        self._enc = enc
        self._buffer = bytearray()
        self._key: bytes | None = None
        self._nonce: bytes | None = None
        self._counter = 0
        self._saw_final = False

    def push(self, chunk: bytes | None, done: bool) -> list[bytes]:
        """Feed bytes and return any newly-decrypted plaintext frames.

        Args:
            chunk: Newly-received bytes (or ``None``).
            done: Whether the underlying stream has ended.

        Returns:
            A list of decrypted plaintext frames (possibly empty).

        Raises:
            ValueError: On a malformed or truncated stream.
        """
        if chunk:
            self._buffer.extend(chunk)

        if self._key is None or self._nonce is None:
            if len(self._buffer) < _NK:
                if done:
                    raise ValueError("malformed chunked OHTTP response")
                return []
            response_nonce = bytes(self._buffer[:_NK])
            self._key, self._nonce = _derive_response_keys(self._response_secret, self._enc, response_nonce)
            del self._buffer[:_NK]

        out: list[bytes] = []
        while self._buffer:
            frame = _decode_varint(self._buffer, 0)
            if frame is None:
                if done:
                    raise ValueError("malformed chunked OHTTP response")
                break
            sealed_len, offset = frame

            if sealed_len == 0:
                # Zero-length prefix marks the final chunk; AAD=b"final".
                if not done:
                    break
                ciphertext = bytes(self._buffer[offset:])
                out.append(self._decrypt_chunk(ciphertext, is_final=True))
                self._buffer.clear()
                self._saw_final = True
                break

            if len(self._buffer) < offset + sealed_len:
                if done:
                    raise ValueError("truncated chunked OHTTP response")
                break

            ciphertext = bytes(self._buffer[offset : offset + sealed_len])
            out.append(self._decrypt_chunk(ciphertext, is_final=False))
            del self._buffer[: offset + sealed_len]

        if done and not self._saw_final:
            raise ValueError("chunked OHTTP response missing final marker")
        return out

    def _decrypt_chunk(self, ciphertext: bytes, is_final: bool) -> bytes:
        assert self._key is not None and self._nonce is not None
        ctr = self._counter.to_bytes(_NN, "big")
        chunk_nonce = bytes(a ^ b for a, b in zip(self._nonce, ctr))
        aad = b"final" if is_final else b""
        plaintext = ChaCha20Poly1305(self._key).decrypt(chunk_nonce, ciphertext, aad)
        self._counter += 1
        return plaintext
