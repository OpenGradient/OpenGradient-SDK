"""Browser-based login to an OpenGradient Chat account.

Opens the chat app's CLI-authorization page (``…/cli-auth``) in the user's
browser, runs a short-lived loopback HTTP listener to receive the session the
page hands back, and returns it. This mirrors the flow the chat app implements
in ``app/(standalone)/cli-auth`` (``CliAuth.tsx``): the page only ever POSTs the
session to a loopback (``127.0.0.1`` / ``localhost``) address, so the token never
leaves the user's machine.

The returned `ChatAccountAuth` carries the access token **and** the public client
config the SDK needs to reach the confidential-inference relay, so it drops
straight into `opengradient.client.confidential_llm.ConfidentialLLM`:

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
"""

from __future__ import annotations

import json
import logging
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# The chat app that hosts the CLI-authorization page.
DEFAULT_CHAT_APP_URL = "https://chat.opengradient.ai"
# Path of the device-authorization page (see chat-app app/(standalone)/cli-auth).
CLI_AUTH_PATH = "/cli-auth"
# The ``type`` discriminator on the JSON bundle the page posts back.
BUNDLE_TYPE = "opengradient-cli-auth"


@dataclass(frozen=True)
class ChatAccountAuth:
    """A signed-in OpenGradient Chat session handed back by the browser.

    Mirrors the chat app's ``CliAuthBundle``. Besides the Supabase session
    tokens, it carries the public ``config`` the CLI/SDK needs to talk to
    chat-api and read the TEE registry — a single source of truth so nothing has
    to be hardcoded.

    Attributes:
        access_token: The bearer token to authenticate to the chat-api relay.
        refresh_token: The Supabase refresh token (for obtaining a new access
            token when this one expires).
        token_type: The token type reported by Supabase (typically ``"bearer"``).
        expires_at: Unix seconds at which ``access_token`` expires, if known.
        expires_in: Seconds until expiry at issue time, if known.
        user: The signed-in user record (``id``, ``email``, ``is_anonymous``).
        config: Public client configuration (``chat_api_base_url``,
            ``tee_registry_rpc_url``, ``tee_registry_address``, ...).
        raw: The full bundle exactly as received, for forward compatibility.
    """

    access_token: str
    refresh_token: str
    token_type: str
    expires_at: Optional[int]
    expires_in: Optional[int]
    user: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def email(self) -> Optional[str]:
        """The signed-in account's email, if any."""
        value = self.user.get("email")
        return value if isinstance(value, str) else None

    @property
    def chat_api_base_url(self) -> Optional[str]:
        """The relay base URL to pass to `ConfidentialLLM`."""
        value = self.config.get("chat_api_base_url")
        return value if isinstance(value, str) else None

    @property
    def tee_registry_rpc_url(self) -> Optional[str]:
        """The RPC URL for the TEE registry the chat app is configured against."""
        value = self.config.get("tee_registry_rpc_url")
        return value if isinstance(value, str) else None

    @property
    def tee_registry_address(self) -> Optional[str]:
        """The TEE registry contract address the chat app is configured against."""
        value = self.config.get("tee_registry_address")
        return value if isinstance(value, str) else None

    @property
    def is_expired(self) -> bool:
        """Whether ``access_token`` is past its ``expires_at`` (False if unknown)."""
        return self.expires_at is not None and time.time() >= self.expires_at

    def auth_headers(self) -> Dict[str, str]:
        """Return the ``Authorization`` header for the relay.

        Shaped as a zero-argument callable so it can be handed directly to
        `ConfidentialLLM(auth_headers=...)` / `OhttpRelayClient`, which call it
        per request.
        """
        return {"Authorization": f"Bearer {self.access_token}"}


def login_chat_account(
    chat_app_url: str = DEFAULT_CHAT_APP_URL,
    *,
    app_name: str = "opengradient-sdk",
    timeout: float = 300.0,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    on_ready: Optional[Callable[[str], None]] = None,
) -> ChatAccountAuth:
    """Sign in to an OpenGradient Chat account via the browser and return the session.

    Starts a loopback HTTP listener, opens the chat app's ``/cli-auth`` page
    pointed at it, and blocks until the page posts back the session bundle (after
    the user signs in and authorizes) or ``timeout`` elapses.

    Args:
        chat_app_url: Base URL of the chat app hosting ``/cli-auth``. Defaults to
            ``https://chat.opengradient.ai``.
        app_name: Name shown to the user on the authorization page.
        timeout: Seconds to wait for the user to authorize before giving up.
        open_browser: Open the authorization URL in the default browser. When
            False (or if no browser is available), the URL is surfaced via
            ``on_ready`` / logged so the user can open it manually.
        host: Loopback host to bind the listener to (``127.0.0.1`` or
            ``localhost`` — the page rejects any non-loopback callback).
        port: Loopback port to bind. ``0`` picks a free ephemeral port.
        on_ready: Optional callback invoked with the full authorization URL once
            the listener is up. Defaults to logging it. Use this to surface the
            URL yourself (e.g. print a QR code).

    Returns:
        A `ChatAccountAuth` with the session tokens and client config.

    Raises:
        TimeoutError: If the user did not authorize within ``timeout`` seconds.
        ValueError: If the received bundle was malformed.
    """
    server = HTTPServer((host, port), _CallbackHandler)
    # Where the browser POSTs the session; single-shot, cleared once received.
    server.received_bundle = None  # type: ignore[attr-defined]

    bound_port = int(server.server_address[1])
    redirect_uri = f"http://{host}:{bound_port}/"
    query = urlencode({"redirect_uri": redirect_uri, "app_name": app_name})
    auth_url = f"{chat_app_url.rstrip('/')}{CLI_AUTH_PATH}?{query}"

    if on_ready is not None:
        on_ready(auth_url)
    else:
        logger.info("Open this URL to authorize the SDK:\n  %s", auth_url)

    opened = False
    if open_browser:
        try:
            opened = webbrowser.open(auth_url)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.debug("Could not open a browser automatically: %s", exc)
    if open_browser and not opened:
        logger.info("Could not open a browser automatically — open the URL above manually.")

    deadline = time.monotonic() + timeout
    try:
        while server.received_bundle is None:  # type: ignore[attr-defined]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out after {timeout:.0f}s waiting for chat account authorization.")
            # handle_request() blocks up to server.timeout, then returns so we can
            # re-check the deadline. A preflight OPTIONS and the POST are separate
            # requests, so keep looping until the bundle actually arrives.
            server.timeout = remaining
            server.handle_request()
    finally:
        server.server_close()

    return _parse_bundle(server.received_bundle)  # type: ignore[attr-defined]


def _parse_bundle(bundle: Dict[str, Any]) -> ChatAccountAuth:
    """Coerce a received CLI-auth bundle into a `ChatAccountAuth`."""
    if not isinstance(bundle, dict):
        raise ValueError("chat auth bundle was not a JSON object")
    access_token = bundle.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("chat auth bundle is missing an access_token")
    user = bundle.get("user")
    config = bundle.get("config")
    return ChatAccountAuth(
        access_token=access_token,
        refresh_token=bundle.get("refresh_token") or "",
        token_type=bundle.get("token_type") or "bearer",
        expires_at=_as_int(bundle.get("expires_at")),
        expires_in=_as_int(bundle.get("expires_in")),
        user=user if isinstance(user, dict) else {},
        config=config if isinstance(config, dict) else {},
        raw=bundle,
    )


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Loopback listener that accepts the session the chat app posts back.

    The page fetches the redirect with ``POST`` + ``Content-Type: application/json``,
    which the browser treats as a non-simple cross-origin request — so it first
    sends a CORS preflight (``OPTIONS``). Both are answered with permissive CORS
    headers; without the preflight response the browser would block the POST and
    fall back to manual copy/paste.
    """

    def _send_cors(self) -> None:
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")

    def do_OPTIONS(self) -> None:  # noqa: N802 - required handler name
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - required handler name
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None

        if isinstance(data, dict) and data.get("type") == BUNDLE_TYPE:
            self.server.received_bundle = data  # type: ignore[attr-defined]
            self._respond_json(200, b'{"ok":true}')
        else:
            self._respond_json(400, b'{"ok":false,"error":"unexpected payload"}')

    def _respond_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("cli-auth callback: " + format, *args)
