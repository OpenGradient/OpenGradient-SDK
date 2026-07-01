---
outline: [2,4]
---

[opengradient](../index) / client

# Package opengradient.client

OpenGradient Client -- service modules for the SDK.

## Modules

- **[llm](./llm)** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement (Base OPG tokens)
- **[confidential_llm](./confidential_llm)** -- One-call confidential (Oblivious HTTP) chat: auto-resolves an OHTTP-capable TEE and verifies the response, no wallet needed on the caller
- **[chat_auth](./chat_auth)** -- Browser-based login to an OpenGradient Chat account (the CLI-auth flow): returns the access token and client config for the confidential-inference relay
- **[model_hub](./model_hub)** -- Model repository management: create, version, and upload ML models
- **[alpha](./alpha)** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **[twins](./twins)** -- Digital twins chat via OpenGradient verifiable inference
- **`opengradient.client.opg_token`** -- OPG token Permit2 approval utilities for x402 payments
- **[tee_registry](./tee_registry)** -- TEE registry client for verified endpoints and TLS certificates

## Usage

```python
import opengradient as og

# LLM inference (Base OPG tokens)
llm = og.LLM(private_key="0x...")
llm.ensure_opg_approval(min_allowance=5)
result = await llm.chat(model=og.TEE_LLM.CLAUDE_HAIKU_4_5, messages=[...])

# On-chain model inference (OpenGradient testnet gas tokens)
alpha = og.Alpha(private_key="0x...")
result = alpha.infer(model_cid, og.InferenceMode.VANILLA, model_input)

# Model Hub (requires email auth)
hub = og.ModelHub(email="you@example.com", password="...")
repo = hub.create_model("my-model", "A price prediction model")
```

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

### `login_chat_account()`

```python
def login_chat_account(
    chat_app_url: str = 'https://chat.opengradient.ai',
    *,
    app_name: str = 'opengradient-sdk',
    timeout: float = 300.0,
    open_browser: bool = True,
    host: str = '127.0.0.1',
    port: int = 0,
    on_ready: Optional[Callable[[str], None]] = None
) ‑> `ChatAccountAuth`
```
Sign in to an OpenGradient Chat account via the browser and return the session.

Starts a loopback HTTP listener, opens the chat app's ``/cli-auth`` page
pointed at it, and blocks until the page posts back the session bundle (after
the user signs in and authorizes) or ``timeout`` elapses.

**Arguments**

* **`chat_app_url`**: Base URL of the chat app hosting ``/cli-auth``. Defaults to
        ``https://chat.opengradient.ai``.
* **`app_name`**: Name shown to the user on the authorization page.
* **`timeout`**: Seconds to wait for the user to authorize before giving up.
* **`open_browser`**: Open the authorization URL in the default browser. When
        False (or if no browser is available), the URL is surfaced via
        ``on_ready`` / logged so the user can open it manually.
* **`host`**: Loopback host to bind the listener to (``127.0.0.1`` or
        ``localhost`` — the page rejects any non-loopback callback).
* **`port`**: Loopback port to bind. ``0`` picks a free ephemeral port.
* **`on_ready`**: Optional callback invoked with the full authorization URL once
        the listener is up. Defaults to logging it. Use this to surface the
        URL yourself (e.g. print a QR code).

**Returns**

A `ChatAccountAuth` with the session tokens and client config.

**`ChatAccountAuth` fields:**

* **`access_token`**: The bearer token to authenticate to the chat-api relay.
* **`refresh_token`**: The Supabase refresh token (for obtaining a new access
        token when this one expires).
* **`token_type`**: The token type reported by Supabase (typically ``"bearer"``).
* **`expires_at`**: Unix seconds at which ``access_token`` expires, if known.
* **`expires_in`**: Seconds until expiry at issue time, if known.
* **`user`**: The signed-in user record (``id``, ``email``, ``is_anonymous``).
* **`config`**: Public client configuration (``chat_api_base_url``,
        ``tee_registry_rpc_url``, ``tee_registry_address``, ...).
* **`raw`**: The full bundle exactly as received, for forward compatibility.

**Raises**

* **`TimeoutError`**: If the user did not authorize within ``timeout`` seconds.
* **`ValueError`**: If the received bundle was malformed.

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

### `ChatAccountAuth`

A signed-in OpenGradient Chat session handed back by the browser.

Mirrors the chat app's ``CliAuthBundle``. Besides the Supabase session
tokens, it carries the public ``config`` the CLI/SDK needs to talk to
chat-api and read the TEE registry — a single source of truth so nothing has
to be hardcoded.

**Attributes**

* **`access_token`**: The bearer token to authenticate to the chat-api relay.
* **`refresh_token`**: The Supabase refresh token (for obtaining a new access
        token when this one expires).
* **`token_type`**: The token type reported by Supabase (typically ``"bearer"``).
* **`expires_at`**: Unix seconds at which ``access_token`` expires, if known.
* **`expires_in`**: Seconds until expiry at issue time, if known.
* **`user`**: The signed-in user record (``id``, ``email``, ``is_anonymous``).
* **`config`**: Public client configuration (``chat_api_base_url``,
        ``tee_registry_rpc_url``, ``tee_registry_address``, ...).
* **`raw`**: The full bundle exactly as received, for forward compatibility.

#### Constructor

```python
def __init__(
    access_token: str,
    refresh_token: str,
    token_type: str,
    expires_at: Optional[int],
    expires_in: Optional[int],
    user: Dict[str, Any] = &lt;factory&gt;,
    config: Dict[str, Any] = &lt;factory&gt;,
    raw: Dict[str, Any] = &lt;factory&gt;
)
```

#### Methods

---

#### `auth_headers()`

```python
def auth_headers(self) ‑> Dict[str, str]
```
Return the ``Authorization`` header for the relay.

Shaped as a zero-argument callable so it can be handed directly to
`ConfidentialLLM(auth_headers=...)` / `OhttpRelayClient`, which call it
per request.

### `OhttpRelayClient`

Send verified, private chat completions to a TEE through an OHTTP relay.

#### Constructor

```python
def __init__(
    relay_url: str,
    tee: `TEEEndpoint`,
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

### `TEEEndpoint`

A verified TEE resolved from the registry.

Carries everything needed for both trust paths: the endpoint + pinned TLS
cert for a direct x402 connection, and the OHTTP/HPKE key material +
signing key for the oblivious-HTTP relay path.

**Attributes**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.

#### Constructor

```python
def __init__(
    tee_id: str,
    endpoint: str,
    tls_cert_der: bytes,
    payment_address: str,
    signing_public_key_der: bytes = b'',
    ohttp_config: Optional[`OhttpConfig`] = None,
    pcr_hash: str = ''
)
```

### `TEERegistry`

Queries the on-chain TEE Registry contract to retrieve verified TEE endpoints
and their TLS certificates.

Instead of blindly trusting the TLS certificate presented by a TEE server
(TOFU), this class fetches the certificate that was submitted and verified
during TEE registration.  Any certificate that does not match the one stored
in the registry should be rejected.

#### Constructor

```python
def __init__(rpc_url: str, registry_address: str)
```

**Arguments**

* **`rpc_url`**: RPC endpoint for the chain where the registry is deployed.
* **`registry_address`**: Address of the deployed TEERegistry contract.

#### Methods

---

#### `get_active_tees_by_type()`

```python
def get_active_tees_by_type(self, tee_type: int) ‑> List[`TEEEndpoint`]
```
Return all active TEEs of the given type with their endpoints and TLS certs.

Uses the contract's ``getActiveTEEs(teeType)`` which returns only TEEs that
are enabled, have a valid (non-revoked) PCR, and a fresh heartbeat — all in
a single on-chain call.

**Arguments**

* **`tee_type`**: Integer TEE type (0=LLMProxy, 1=Validator).

**Returns**

List of TEEEndpoint objects for active TEEs of that type.

---

#### `get_llm_tee()`

```python
def get_llm_tee(self) ‑> Optional[`TEEEndpoint`]
```
Return a random active LLM proxy TEE from the registry.

The returned ``TEEEndpoint`` is the full record: endpoint + pinned TLS
cert for direct x402 connections, plus the OHTTP/HPKE ``ohttp_config``
and ``signing_public_key_der`` for the oblivious-HTTP relay path.

**Returns**

TEEEndpoint for an active LLM proxy TEE, or None if none are available.
**`TEEEndpoint` fields:**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.

---

#### `get_llm_tee_ohttp_config()`

```python
def get_llm_tee_ohttp_config(self) ‑> Optional[`TEEEndpoint`]
```
Return a random active LLM proxy TEE that advertises an OHTTP config.

Like ``get_llm_tee`` but skips TEEs missing HPKE key material, so the
result is guaranteed usable for the Oblivious HTTP path.

**Returns**

A TEEEndpoint with a non-empty ``ohttp_config``, or None.
**`TEEEndpoint` fields:**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.

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

### `VerificationError`

Raised when a response fails any step of TEE verification.

Callers should treat this as fatal: never surface content that failed
verification to the end user.

#### Constructor

```python
def __init__(*args, **kwargs)
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
    proof: `TeeProof`,
    stream_frames: Optional[list[str]] = None
)
```