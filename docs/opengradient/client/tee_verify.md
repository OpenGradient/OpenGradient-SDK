---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / tee_verify

# Package opengradient.client.tee_verify

Cryptographic verification of TEE inference responses.

Every response from an OpenGradient TEE gateway carries an RSA-PSS signature over
``keccak256(requestHash || outputHash || uint256(timestamp))``, produced inside
the enclave with the key the on-chain registry records for that TEE. The trust
chain is:

    reproducible build -> PCRs -> on-chain registry entry (pcrHash + signing key)
                       -> per-response RSA-PSS signature

So if you trust a TEE's registry signing key (optionally pinned to an expected
``pcrHash``) and the signature verifies, the response was produced inside that
attested enclave and was not modified in transit — no trust in the relay, the
host, or us is required. The relay never holds the signing key and so cannot
forge a response.

This module mirrors the gateway's signing (``compute_tee_msg_hash`` + RSA-PSS,
salt length 32) and the chat-app's browser verification, kept in one place so all
clients verify identically.

## Functions

---

### `build_inner_request()`

```python
def build_inner_request(body: dict) ‑> tuple[dict, dict]
```
Build ``(wire, canonical)`` request dicts from an OpenAI chat-completions ``body``.

The gateway commits (in its signed request hash) to only a fixed subset of
fields, so anything else the caller sends (``n``, ``top_p``, ``tool_choice``,
...) is intentionally dropped — it would not be covered by the signature.

**Arguments**

* **`body`**: An OpenAI ``/v1/chat/completions`` request body.

**Returns**

``(wire, canonical)`` where ``wire`` is the object to encrypt and send
(full message content preserved so the model sees attachment bytes), and
``canonical`` is the dict whose ``json.dumps(sort_keys=True)`` the gateway
hashes (attachment bytes stripped). Pass ``canonical`` to
:func:`verify_response`.

**Raises**

* **`UnsupportedRequestError`**: If the body is missing ``model`` or ``messages`` or
        contains an unknown message role.

---

### `canonical_request_bytes()`

```python
def canonical_request_bytes(canonical_request: dict) ‑> bytes
```
Serialize a canonical request exactly as the gateway hashes it: ``json.dumps(sort_keys=True)``.

---

### `canonical_user_content()`

```python
def canonical_user_content(content: Any) ‑> Any
```
Canonicalize user-message content for request hashing.

Plain strings pass through. For multimodal content (a list of parts), text is
kept verbatim and every attachment is reduced to ``{type[, filename]}`` — the
inline bytes are dropped (they ride inside the encrypted envelope and are never
hashed). Mirrors the gateway's ``canonical_user_content``.

---

### `pem_from_der()`

```python
def pem_from_der(signing_public_key_der: bytes) ‑> str
```
Convert a DER (SPKI) public key (e.g. ``TEEEndpoint.signing_public_key_der``) to PEM.

---

### `response_content_for_hash()`

```python
def response_content_for_hash(response_body: dict) ‑> str
```
Extract the exact string the gateway hashed as the signed output.

For tool-call responses the gateway hashes ``json.dumps(tool_calls, sort_keys=True)``;
otherwise it hashes the assistant message text (generated image bytes are
excluded — they ride out-of-band and are not signed).

---

### `tee_id_for_key()`

```python
def tee_id_for_key(signing_key_pem: str) ‑> str
```
Return ``0x`` + keccak256(DER(SubjectPublicKeyInfo)).

Matches the gateway's ``TEEKeyManager.tee_id`` and the registry's keyed tee_id.

---

### `verify_response()`

```python
def verify_response(
    *,
    canonical_request: dict,
    response_body: dict,
    response_content: str,
    signing_key_pem: str,
    expected_tee_id: Optional[str] = None,
    tee_host: Optional[str] = None
) ‑> `TeeProof`
```
Verify a (decrypted) TEE gateway response.

**Arguments**

* **`canonical_request`**: The canonical request dict (see :func:`build_inner_request`)
        whose ``json.dumps(sort_keys=True)`` the gateway hashed.
* **`response_body`**: The parsed inner JSON (single-shot body, or the final SSE
        frame for streams), carrying ``tee_signature``, ``tee_request_hash``,
        ``tee_output_hash``, ``tee_timestamp`` and ``tee_id``.
* **`response_content`**: The exact text/JSON the gateway hashed as output — use
        :func:`response_content_for_hash`, or the accumulated stream text.
* **`signing_key_pem`**: The enclave's RSA public key from the on-chain registry
        (the trust anchor; convert DER via :func:`pem_from_der`).
* **`expected_tee_id`**: If given, require the response/key tee_id to match.
* **`tee_host`**: Optional host, recorded on the returned proof for display.

**Returns**

A :class:`TeeProof` describing the verified provenance.

**`TeeProof` fields:**

* **`tee_id`**: The TEE identity (``0x`` + keccak256 of the signing key DER).
* **`request_hash`**: keccak256 of the canonical request, hex (no ``0x``).
* **`output_hash`**: keccak256 of the signed output content, hex (no ``0x``).
* **`timestamp`**: The enclave-asserted signing timestamp (unix seconds).
* **`signature`**: The base64 RSA-PSS signature that was verified.
* **`signing_key_pem`**: The PEM signing key the signature verified against.
* **`tee_host`**: Optional host the response came from, for display.

**Raises**

* **`VerificationError`**: If any check fails (missing fields, tee_id mismatch,
        request/output hash mismatch, or bad signature).

## Classes

### `TeeProof`

The verified provenance of a single response.

**Attributes**

* **`tee_id`**: The TEE identity (``0x`` + keccak256 of the signing key DER).
* **`request_hash`**: keccak256 of the canonical request, hex (no ``0x``).
* **`output_hash`**: keccak256 of the signed output content, hex (no ``0x``).
* **`timestamp`**: The enclave-asserted signing timestamp (unix seconds).
* **`signature`**: The base64 RSA-PSS signature that was verified.
* **`signing_key_pem`**: The PEM signing key the signature verified against.
* **`tee_host`**: Optional host the response came from, for display.

#### Constructor

```python
def __init__(
    tee_id: str,
    request_hash: str,
    output_hash: str,
    timestamp: int,
    signature: str,
    signing_key_pem: str,
    tee_host: Optional[str] = None
)
```

### `UnsupportedRequestError`

Raised when an OpenAI-style request cannot be expressed as a gateway request.

#### Constructor

```python
def __init__(*args, **kwargs)
```

### `VerificationError`

Raised when a response fails any step of TEE verification.

Callers should treat this as fatal: never surface content that failed
verification to the end user.

#### Constructor

```python
def __init__(*args, **kwargs)
```