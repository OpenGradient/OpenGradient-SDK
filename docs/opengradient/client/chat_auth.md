---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / chat_auth

# Package opengradient.client.chat_auth

Browser-based login to an OpenGradient Chat account.

Opens the chat app's CLI-authorization page (``…/cli-auth``) in the user's
browser, runs a short-lived loopback HTTP listener to receive the session the
page hands back, and returns it. This mirrors the flow the chat app implements
in ``app/(standalone)/cli-auth`` (``CliAuth.tsx``): the page only ever POSTs the
session to a loopback (``127.0.0.1`` / ``localhost``) address, so the token never
leaves the user's machine.

The returned `ChatAccountAuth` carries the access token **and** the public client
config the SDK needs to reach the confidential-inference relay, so it drops
straight into [ConfidentialLLM](./confidential_llm):

    ```python
    import opengradient as og

    auth = og.login_chat_account()          # opens the browser, waits for sign-in
    client = og.ConfidentialLLM(
        relay_url=auth.chat_api_base_url,   # from the returned config
        auth_headers=auth.auth_headers,     # Bearer <access_token>
    )
    result = client.chat(
        model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(result.content)
    ```

## Functions

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