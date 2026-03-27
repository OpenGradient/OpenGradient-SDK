"""Manages the lifecycle of a connection to a TEE endpoint."""

import asyncio
import logging
import ssl
from dataclasses import dataclass
from typing import Dict, Optional, Union

from x402 import x402Client
from x402.http.clients import x402HttpxClient

from .tee_registry import TEE_TYPE_LLM_PROXY, TEERegistry, build_ssl_context_from_der

logger = logging.getLogger(__name__)

_TEE_REFRESH_INTERVAL = 300  # Re-resolve TEE from registry every 5 minutes


@dataclass(frozen=True)
class ActiveTEE:
    """Snapshot of the currently connected TEE."""

    endpoint: str
    http_client: x402HttpxClient
    tee_id: Optional[str]
    payment_address: Optional[str]

    def metadata(self) -> Dict:
        """Return TEE metadata dict for decorating responses."""
        return dict(
            tee_id=self.tee_id,
            tee_endpoint=self.endpoint,
            tee_payment_address=self.payment_address,
        )


class TEEConnection:
    """Maintains a verified connection to a single TEE endpoint.

    Handles initial resolution from the on-chain registry (or an explicit URL),
    TLS certificate pinning, background health checks, and automatic failover
    when the current TEE becomes unavailable.

    Use ``get()`` to obtain the current ``ActiveTEE`` snapshot for making requests.

    Args:
        x402_client: Configured x402 payment client for creating HTTP clients.
        registry: TEERegistry for looking up active TEEs. None when using an explicit URL.
        llm_server_url: Bypass the registry and connect directly to this URL.
    """

    def __init__(
        self,
        x402_client: x402Client,
        registry: Optional[TEERegistry] = None,
        llm_server_url: Optional[str] = None,
    ):
        self._x402_client = x402_client
        self._registry = registry
        self._llm_server_url = llm_server_url

        self._active: Optional[ActiveTEE] = None
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None

        self._connect()

    # ── Public API ──────────────────────────────────────────────────────

    def get(self) -> ActiveTEE:
        """Return a snapshot of the current TEE connection."""
        return self._active

    # ── Connection management ───────────────────────────────────────────

    def _connect(self) -> None:
        """Resolve TEE from registry and create a secure HTTP client."""
        endpoint, tls_cert_der, tee_id, payment_address = self._resolve_tee(
            self._llm_server_url,
            self._registry,
        )

        ssl_ctx = build_ssl_context_from_der(tls_cert_der) if tls_cert_der else None
        tls_verify: Union[ssl.SSLContext, bool] = ssl_ctx if ssl_ctx else (self._llm_server_url is None)

        self._active = ActiveTEE(
            endpoint=endpoint,
            http_client=x402HttpxClient(self._x402_client, verify=tls_verify),
            tee_id=tee_id,
            payment_address=payment_address,
        )

    async def reconnect(self) -> None:
        """Connect to a new TEE from the registry and rebuild the HTTP client."""
        async with self._refresh_lock:
            old_client = self._active.http_client
            self._connect()
            try:
                await old_client.aclose()
            except Exception:
                logger.debug("Failed to close previous HTTP client during TEE refresh.", exc_info=True)

    # ── Background health check ─────────────────────────────────────────

    def ensure_refresh_loop(self) -> None:
        """Start the background TEE refresh loop if not already running.

        No-op when ``llm_server_url`` is set (bypasses the registry).
        Called lazily from async request methods since ``__init__`` is synchronous.
        """
        if self._llm_server_url is not None:
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.create_task(self._tee_refresh_loop())

    async def _tee_refresh_loop(self) -> None:
        """Periodically check that the current TEE is still active in the registry.

        If the current TEE is no longer active, performs a full refresh to pick
        a new one.  Does nothing when the current TEE is still healthy.
        """
        while True:
            await asyncio.sleep(_TEE_REFRESH_INTERVAL)
            try:
                active_tees = self._registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
                if any(t.tee_id == self._active.tee_id for t in active_tees):
                    logger.debug("Current TEE %s still active; no refresh needed.", self._active.tee_id)
                    continue
                logger.info("Current TEE %s no longer active; switching to a new one.", self._active.tee_id)
                await self.reconnect()
            except Exception:
                logger.warning("Background TEE health check failed; will retry next cycle.", exc_info=True)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel the background refresh loop and close the HTTP client."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._active is not None:
            await self._active.http_client.aclose()

    # ── Static helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_tee(
        tee_endpoint_override: Optional[str],
        registry: Optional[TEERegistry],
    ) -> tuple:
        """Resolve TEE endpoint and metadata from the on-chain registry or explicit URL.

        Returns:
            (endpoint, tls_cert_der, tee_id, payment_address)
        """
        if tee_endpoint_override is not None:
            return tee_endpoint_override, None, None, None

        if registry is None:
            raise ValueError("Either llm_server_url or a TEERegistry instance must be provided.")

        try:
            tee = registry.get_llm_tee()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch LLM TEE endpoint from registry: {e}") from e

        if tee is None:
            raise ValueError("No active LLM proxy TEE found in the registry. Pass llm_server_url explicitly to override.")

        logger.info("Using TEE endpoint from registry: %s (teeId=%s)", tee.endpoint, tee.tee_id)
        return tee.endpoint, tee.tls_cert_der, tee.tee_id, tee.payment_address
