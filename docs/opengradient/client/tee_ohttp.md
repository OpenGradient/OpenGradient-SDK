---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / tee_ohttp

# Package opengradient.client.tee_ohttp

Client-side Oblivious HTTP (RFC 9458) encapsulation for anonymous TEE inference.

This is the *sender* side of the construction the tee-gateway implements on the
recipient side and the chat-app implements in the browser. Using it, a client can
HPKE-encrypt an inference request to a TEE's published X25519 key, send it through
an untrusted relay, and decrypt the (single-shot or chunked-streaming) response —
the relay only ever sees ciphertext.

The ciphersuite is fixed and must match the enclave and the on-chain
[OhttpConfig](./tee_registry):

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

## Functions

---

### `decrypt_response()`

```python
def decrypt_response(response_secret: bytes, enc: bytes, sealed: bytes) ‑> bytes
```
Decrypt a single-shot OHTTP response (RFC 9458 §4.5).

**Arguments**

* **`response_secret`**: ``EncapsulatedRequest.response_secret``.
* **`enc`**: ``EncapsulatedRequest.enc``.
* **`sealed`**: The full response body from the relay.

**Returns**

The decrypted inner response bytes.

**Raises**

* **`ValueError`**: If the response is too short to be well-formed.

---

### `encapsulate_request()`

```python
def encapsulate_request(
    public_key_raw: bytes,
    plaintext: bytes,
    *,
    key_id: int = 1,
    kem_id: int = 32,
    kdf_id: int = 1,
    aead_id: int = 3
) ‑> `EncapsulatedRequest`
```
HPKE-seal ``plaintext`` to a TEE's raw X25519 public key.

**Arguments**

* **`public_key_raw`**: The 32-byte raw X25519 public key from the TEE's OHTTP
        config (``OhttpConfig.public_key``).
* **`plaintext`**: The inner request body (typically a UTF-8 JSON chat request).
* **`key_id`**: The OHTTP key-config id from the TEE's ``OhttpConfig.key_id``.
        Threaded into the request header so a TEE that rotated to a new
        key_id (while keeping this suite) can still decapsulate. Defaults to
        the canonical ``0x01``.
    kem_id, kdf_id, aead_id: The HPKE algorithm ids from the TEE's config.
        This client implements one fixed suite; mismatching ids are rejected
        rather than silently producing an undecryptable request.

**Returns**

An `EncapsulatedRequest` ready to send to a relay.

**`EncapsulatedRequest` fields:**

* **`wire`**: The bytes to send to the relay (header || enc || ciphertext).
* **`enc`**: Our ephemeral X25519 public key; salts the response keying.
* **`response_secret`**: Exported secret for a single-shot response.
* **`chunked_response_secret`**: Exported secret for a chunked-streaming response.

**Raises**

* **`ValueError`**: If ``public_key_raw`` is not 32 bytes, or the algorithm ids
        don't match this client's supported suite.

## Classes

### `ChaCha20Poly1305`

#### Constructor

```python
def __init__(key)
```

#### Static methods

---

#### `generate_key()`

```python
static def generate_key()
```

#### Methods

---

#### `decrypt()`

```python
def decrypt(
    self,
    /,
    nonce,
    data,
    associated_data
)
```

---

#### `encrypt()`

```python
def encrypt(
    self,
    /,
    nonce,
    data,
    associated_data
)
```

### `ChunkedResponseDecrypter`

Incrementally decrypt a chunked OHTTP response stream (draft-ietf-ohai-chunked-ohttp-08).

Feed it raw response bytes as they arrive; it yields decrypted plaintext
frames (typically the inner SSE ``data:`` events). The final frame carries
AAD=b"final"; its absence at end-of-stream is treated as truncation, so a
network attacker cannot silently cut a stream short.

#### Constructor

```python
def __init__(response_secret: bytes, enc: bytes)
```

#### Methods

---

#### `push()`

```python
def push(self, chunk: bytes | None, done: bool) ‑> list[bytes]
```
Feed bytes and return any newly-decrypted plaintext frames.

**Arguments**

* **`chunk`**: Newly-received bytes (or ``None``).
* **`done`**: Whether the underlying stream has ended.

**Returns**

A list of decrypted plaintext frames (possibly empty).

**Raises**

* **`ValueError`**: On a malformed or truncated stream.

### `EncapsulatedRequest`

An HPKE-sealed request plus the secrets needed to open its response.

**Attributes**

* **`wire`**: The bytes to send to the relay (header || enc || ciphertext).
* **`enc`**: Our ephemeral X25519 public key; salts the response keying.
* **`response_secret`**: Exported secret for a single-shot response.
* **`chunked_response_secret`**: Exported secret for a chunked-streaming response.

#### Constructor

```python
def __init__(
    wire: bytes,
    enc: bytes,
    response_secret: bytes,
    chunked_response_secret: bytes
)
```