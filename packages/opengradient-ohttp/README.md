# opengradient-ohttp

End-to-end encrypted, TEE-verified chat completions for the OpenAI Python SDK.

- Your request is sealed with Oblivious HTTP (RFC 9458, HPKE) to an attested
  OpenGradient TEE.
- The DreamFort relay forwards only the ciphertext, and bills your developer
  account.
- Every response is checked against the TEE's on-chain signing key before you
  see it.

## Install

```bash
pip install "opengradient-ohttp[openai]"
```

Python 3.10+. The transport itself only needs `httpx`. The `[openai]` extra
pulls in the OpenAI SDK.

## Use

```python
import httpx
from openai import OpenAI
from opengradient_ohttp import OHTTPTransport

client = OpenAI(
    base_url="https://ohttp.invalid/v1",   # never contacted; the transport routes to the relay
    api_key="df_sk_...",                   # your DreamFort developer API key
    http_client=httpx.Client(transport=OHTTPTransport()),
)

completion = client.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(completion.choices[0].message.content)
```

The `.invalid` host is deliberate. If a client is ever built without the
transport, the call fails, so a plaintext prompt is never sent.

### Async

```python
from openai import AsyncOpenAI
from opengradient_ohttp import AsyncOHTTPTransport

client = AsyncOpenAI(
    base_url="https://ohttp.invalid/v1",
    api_key="df_sk_...",
    http_client=httpx.AsyncClient(transport=AsyncOHTTPTransport()),
)
```

### Streaming

`stream=True` works as usual, and deltas arrive as they are decrypted. The
TEE's signed final frame is checked before it (and `[DONE]`) reach you:

- A bad signature raises `opengradient_ohttp.TeeVerificationError` from the
  stream.
- A stream that is cut short raises `StreamIncompleteError`.

To get nothing until the whole reply has been verified, use:

```python
OHTTPTransport(verify_before_release=True)
```

### Reading the proof

A non-streaming response carries the proof in its headers:

```python
raw = client.chat.completions.with_raw_response.create(model=..., messages=...)
raw.headers["x-tee-verified"]   # "true"
raw.headers["x-tee-id"], raw.headers["x-tee-signature"], raw.headers["x-tee-request-hash"]
completion = raw.parse()
```

The same `tee_*` fields are on the completion object. For a stream, they are
on the last chunk.

## Errors

| HTTP | OpenAI exception | Meaning |
|---|---|---|
| 401 | `AuthenticationError` | The key is invalid or revoked, or its creator left the account |
| 402 | `APIStatusError` | The developer account is out of credits |
| 403 | `PermissionDeniedError` | The key's creator is blocked |
| 429 | `RateLimitError` | The per-account rate limit was hit (retried by the OpenAI SDK) |
| 502 `tee_verification_failed` | `InternalServerError` | The response failed TEE verification (not retried) |
| 503 `tee_registry_unavailable` | `InternalServerError` | The on-chain TEE registry couldn't be read |
| 404 `unsupported_endpoint` | `NotFoundError` | Only `POST /chat/completions` goes over OHTTP |

## Options

All `OHTTPTransport(...)` arguments are keyword-only:

| Argument | Default | |
|---|---|---|
| `relay_url` | `https://chat-api.memchat.io/api/v1/developer/ohttp` | An origin alone gets the developer path appended |
| `api_key` | the OpenAI client's key | Set it to use a different key on the relay |
| `verify_before_release` | `False` | Buffer streams until they are verified |
| `send_model_hint` | `False` | Send the model name in plaintext headers, for relay metrics |
| `registry_rpc_url`, `registry_address` | OpenGradient devnet registry | Where TEEs and their keys come from |
| `tee` | — | Pin one `TeeTarget` instead of reading the registry |
| `tee_cache_ttl` | `300` | Seconds to keep a registry pick |
| `transport` | `httpx.HTTPTransport()` | The outer hop. Pass your own for proxies; TLS verification stays on by default |

## What is sent where

- **The relay (chat-api)** sees:
  - your API key;
  - the TEE id;
  - the ciphertext;
  - the size and timing of the request and response.
- **The TEE** sees the plaintext request.
- **Only these request fields are sent**, because the gateway signs them:
  - `model`, `messages`, `temperature`, `max_tokens` (or `max_completion_tokens`);
  - `stop`, `tools`, `tool_choice`, `response_format`;
  - `reasoning_effort`, `aspect_ratio`, `resolution`, `web_search`.

  Other fields, such as `top_p` and `seed`, are dropped with a warning, and
  `n > 1` is rejected.
- **OpenAI SDK headers** (`x-stainless-*`) are never forwarded.

## Development

```bash
uv run --extra test pytest
```
