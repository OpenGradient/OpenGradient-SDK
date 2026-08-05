---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / confidential_llm

# Package opengradient.client.confidential_llm

High-level confidential (Oblivious HTTP) LLM inference.

This is the one-call entry point for verified, private chat completions over
Oblivious HTTP — the same path the OpenGradient chat app uses in the browser
(``lib/api/ohttp.ts``). It ties together the pieces an integrator would
otherwise have to wire up by hand:

  1. [tee_registry](./tee_registry) — discover an OHTTP-capable TEE
     (endpoint, HPKE key, signing key) from the on-chain registry.
  2. [tee_ohttp_client](./tee_ohttp_client) — HPKE-encrypt the request, POST the
     ciphertext to the relay's confidential-inference path
     (``/api/v1/chat/ohttp``), then decrypt and verify the response.

Unlike [LLM](./llm), this client needs **no wallet on the
caller's side**: the request travels end-to-end encrypted to the enclave, and
the untrusted relay (which holds the x402 account and pays per request) only
ever sees ciphertext. Authentication to the relay is left to the caller via an
``auth_headers`` provider, so the client works against any relay deployment
without baking in a credential scheme.

Every response is signature-verified inside the client before any content is
returned — see [tee_verify](./tee_verify) for the trust chain.

Usage:
    ```python
    import opengradient as og

    # Resolve an OHTTP-capable TEE from the on-chain registry and target the
    # relay's confidential-inference path automatically.
    client = og.ConfidentialLLM(
        relay_url="https://chat-api.opengradient.ai",
        auth_headers=lambda: {"Authorization": "Bearer <token>"},
    )

    result = client.chat(
        model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=200,
    )
    print(result.content)          # verified assistant text
    print(result.proof.tee_id)     # attested enclave identity
    ```

## Classes

### `ConfidentialLLM`

Verified, private chat completions over Oblivious HTTP through a relay.

Discovers an OHTTP-capable TEE from the on-chain registry, POSTs
HPKE-encrypted requests to the relay's confidential-inference path
(``/api/v1/chat/ohttp``), and verifies the enclave's signature before
returning any content — matching the browser chat app's OHTTP flow.

**Raises**

* **`RuntimeError`**: If the registry has no active OHTTP-capable LLM TEE.

#### Constructor

```python
def __init__(
    relay_url: str,
    *,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    registry_address: str = '0x703cB174AEadB35D611858369B4b1111dC9Abda6',
    auth_headers: Optional[AuthHeaderProvider] = None,
    session: Optional[`Session`] = None,
    timeout: float = 120.0
)
```

**Arguments**

* **`relay_url`**: The relay base URL (e.g. ``"https://chat-api.opengradient.ai"``)
        or the full confidential-inference URL. The
        ``/api/v1/chat/ohttp`` path is appended automatically when the given
        URL does not already end with it.
* **`rpc_url`**: RPC endpoint for the chain the TEE registry is deployed on.
        Defaults to the OpenGradient devnet.
* **`registry_address`**: Address of the deployed ``TEERegistry`` contract.
* **`auth_headers`**: Optional callable returning headers to authenticate to the
        relay (called per request so tokens can be refreshed), e.g.
        ``lambda: {"Authorization": "Bearer <token>"}``.
* **`session`**: Optional ``requests.Session`` to reuse connections.
* **`timeout`**: Per-request timeout in seconds.

#### Static methods

---

#### `from_tee()`

```python
static def from_tee(
    relay_url: str,
    tee: TEEEndpoint,
    *,
    auth_headers: Optional[AuthHeaderProvider] = None,
    session: Optional[`Session`] = None,
    timeout: float = 120.0
) ‑> `ConfidentialLLM`
```
Create a client for a TEE you have already resolved.

Use this to skip the registry lookup — for a self-hosted TEE, a pinned
enclave, or when you have already selected a `TEEEndpoint` (which must
carry an ``ohttp_config`` and ``signing_public_key_der``).

**Arguments**

* **`relay_url`**: The relay base URL or full confidential-inference URL (the
        ``/api/v1/chat/ohttp`` path is appended when missing).
* **`tee`**: The `opengradient.client.tee_registry.TEEEndpoint` to encrypt to.
* **`auth_headers`**: Optional per-request relay auth header provider.
* **`session`**: Optional ``requests.Session`` to reuse connections.
* **`timeout`**: Per-request timeout in seconds.

#### Methods

---

#### `chat()`

```python
def chat(
    self,
    model: "Union['TEE_LLM', str]",
    messages: List[Dict],
    max_tokens: int = 100,
    stop_sequence: Optional[List[str]] = None,
    temperature: float = 0.0,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    response_format: Optional[Dict] = None,
    web_search: bool = False,
    stream: bool = False
) ‑> VerifiedChatResponse
```
Send a verified, private chat completion over Oblivious HTTP.

Builds an OpenAI ``/v1/chat/completions`` request from the arguments (the
same shape the chat app builds in ``buildInnerChatRequest``), encrypts
it to the resolved TEE, and returns the signature-verified response.

**Arguments**

* **`model`**: The model to use, e.g. ``og.TEE_LLM.CLAUDE_HAIKU_4_5``. A
        ``"provider/model"`` value is reduced to the gateway model id
        (the part after the ``/``), mirroring the chat app.
* **`messages`**: OpenAI-style chat messages.
* **`max_tokens`**: Maximum output tokens. Default 100.
* **`stop_sequence`**: Optional list of stop sequences.
* **`temperature`**: Sampling temperature (0-1). Default 0.0.
* **`tools`**: Optional function-calling tool definitions.
* **`tool_choice`**: Optional tool-choice directive (forwarded on the wire but
        not part of the signed request hash).
* **`response_format`**: Optional OpenAI ``response_format`` dict.
* **`web_search`**: Enable the provider's native web search. Default False.
* **`stream`**: When True, the encrypted stream is fully buffered and verified
        before returning; ``result.stream_frames`` holds the decrypted SSE
        frames. Default False.

**Returns**

A `opengradient.client.tee_ohttp_client.VerifiedChatResponse`.

---

#### `chat_completion()`

```python
def chat_completion(self, body: Dict) ‑> `VerifiedChatResponse`
```
Send a raw OpenAI chat-completions ``body`` (non-streaming) and verify it.

Escape hatch for callers that build the request body themselves. See
`chat` for the argument-driven convenience.

**Returns**

**`VerifiedChatResponse` fields:**

* **`body`**: The inner response JSON (the single-shot body, or the final SSE
        frame for a stream).
* **`content`**: The assistant text (or tool-calls JSON) that was verified.
* **`proof`**: The :class:`opengradient.client.tee_verify.TeeProof`.
* **`stream_frames`**: For streaming requests, the decrypted inner SSE ``data:``
        event strings (already verified), ready to replay to a client;
        ``None`` for single-shot requests.

---

#### `stream_chat_completion()`

```python
def stream_chat_completion(self, body: Dict) ‑> `VerifiedChatResponse`
```
Send a raw OpenAI chat-completions ``body`` as a stream and verify it.

The encrypted stream is fully buffered and verified before returning, so
``result.stream_frames`` is safe to replay to an end user.

**Returns**

**`VerifiedChatResponse` fields:**

* **`body`**: The inner response JSON (the single-shot body, or the final SSE
        frame for a stream).
* **`content`**: The assistant text (or tool-calls JSON) that was verified.
* **`proof`**: The :class:`opengradient.client.tee_verify.TeeProof`.
* **`stream_frames`**: For streaming requests, the decrypted inner SSE ``data:``
        event strings (already verified), ready to replay to a client;
        ``None`` for single-shot requests.

#### Variables

* `relay_url` : str - The full confidential-inference URL requests are POSTed to.
* [**`tee`**](./tee_registry): The resolved TEE this client encrypts requests to.