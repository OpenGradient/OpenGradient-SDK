---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / tee_ohttp_client

# Package opengradient.client.tee_ohttp_client

High-level Oblivious HTTP relay client for verified, private TEE inference.

This ties together the three lower-level pieces so an integrator doesn't have to:

  1. [tee_registry](./tee_registry) — discover a TEE (endpoint, OHTTP key,
     signing key) from the on-chain registry.
  2. [tee_ohttp](./tee_ohttp) — HPKE-encrypt the request and decrypt the
     response.
  3. [tee_verify](./tee_verify) — verify the enclave's RSA-PSS signature.

The relay (which holds the x402 wallet / account credentials and pays per
request) only ever sees ciphertext. Authentication to the relay is left to the
caller: pass an ``auth_headers`` provider returning whatever the relay expects
(e.g. ``{"Authorization": "Bearer <token>"}``), so this client works for any
relay deployment without baking in a credential scheme.

Verification happens **before** any content is returned. For streaming requests
the full encrypted stream is buffered, verified, and only then handed back as
decrypted SSE frames — so a caller can guarantee no unverified token ever
reaches the end user, at the cost of streaming latency.

## Classes

### `OhttpRelayClient`

Send verified, private chat completions to a TEE through an OHTTP relay.

#### Constructor

```python
def __init__(
    relay_url: str,
    tee: TEEEndpoint,
    *,
    auth_headers: Optional[AuthHeaderProvider] = None,
    session: Optional[`Session`] = None,
    timeout: float = 120.0
)
```

**Arguments**

* **`relay_url`**: Full URL to POST encapsulated requests to (e.g.
        ``https://chat-api.example.com/api/v1/chat/ohttp``).
* **`tee`**: The :class:`opengradient.client.tee_registry.TEEEndpoint` to encrypt
        to (must carry an ``ohttp_config`` and ``signing_public_key_der``).
* **`auth_headers`**: Optional callable returning headers to authenticate to the
        relay (called per request so tokens can be refreshed).
* **`session`**: Optional ``requests.Session`` to reuse connections.
* **`timeout`**: Per-request timeout in seconds.

#### Methods

---

#### `chat_completion()`

```python
def chat_completion(self, body: dict) ‑> `VerifiedChatResponse`
```
Send a non-streaming chat completion and return a verified response.

**Arguments**

* **`body`**: An OpenAI ``/v1/chat/completions`` request body.

**Returns**

A :class:`VerifiedChatResponse`.

**`VerifiedChatResponse` fields:**

* **`body`**: The inner response JSON (the single-shot body, or the final SSE
        frame for a stream).
* **`content`**: The assistant text (or tool-calls JSON) that was verified.
* **`proof`**: The :class:`opengradient.client.tee_verify.TeeProof`.
* **`stream_frames`**: For streaming requests, the decrypted inner SSE ``data:``
        event strings (already verified), ready to replay to a client;
        ``None`` for single-shot requests.

**Raises**

* **`RelayError`**: If the relay or the inner request errored.
* **`VerificationError`**: If the response signature could not be verified.
    opengradient.client.tee_verify.UnsupportedRequestError: If the body is invalid.

---

#### `stream_chat_completion()`

```python
def stream_chat_completion(self, body: dict) ‑> `VerifiedChatResponse`
```
Send a streaming chat completion, verify it, then return decrypted frames.

The encrypted stream is fully buffered and verified before returning, so
the returned ``stream_frames`` are safe to replay to an end user. (This
trades streaming latency for the "no unverified token leaves the machine"
guarantee.)

**Arguments**

* **`body`**: An OpenAI ``/v1/chat/completions`` request body (``stream`` is
        forced on for the wire request).

**Returns**

A :class:`VerifiedChatResponse` with ``stream_frames`` populated.

**`VerifiedChatResponse` fields:**

* **`body`**: The inner response JSON (the single-shot body, or the final SSE
        frame for a stream).
* **`content`**: The assistant text (or tool-calls JSON) that was verified.
* **`proof`**: The :class:`opengradient.client.tee_verify.TeeProof`.
* **`stream_frames`**: For streaming requests, the decrypted inner SSE ``data:``
        event strings (already verified), ready to replay to a client;
        ``None`` for single-shot requests.

### `RelayError`

The relay returned a non-success status, or the inner response was an error.

**Attributes**

* **`status_code`**: The HTTP (or inner) status code.
* **`message`**: A human-readable error message extracted from the response.

#### Constructor

```python
def __init__(status_code: int, message: str)
```

### `VerifiedChatResponse`

A TEE chat response that has passed signature verification.

**Attributes**

* **`body`**: The inner response JSON (the single-shot body, or the final SSE
        frame for a stream).
* **`content`**: The assistant text (or tool-calls JSON) that was verified.
* **`proof`**: The :class:`opengradient.client.tee_verify.TeeProof`.
* **`stream_frames`**: For streaming requests, the decrypted inner SSE ``data:``
        event strings (already verified), ready to replay to a client;
        ``None`` for single-shot requests.

#### Constructor

```python
def __init__(
    body: dict,
    content: str,
    proof: TeeProof,
    stream_frames: Optional[list[str]] = None
)
```