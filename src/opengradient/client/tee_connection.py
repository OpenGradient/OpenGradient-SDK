"""Manages the lifecycle of a connection to a TEE endpoint."""

import asyncio
import logging
import ssl
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Union

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


class TEEConnectionInterface(Protocol):
    """Interface for TEE connection implementations."""

    def get(self) -> ActiveTEE: ...
    def ensure_refresh_loop(self) -> None: ...
    async def reconnect(self) -> None: ...
    async def close(self) -> None: ...


class StaticTEEConnection:
    """TEE connection with a hardcoded endpoint URL.

    No registry lookup, no background refresh. TLS certificate verification
    is disabled because self-hosted TEE servers typically use self-signed certs.

    Args:
        x402_client: Configured x402 payment client for creating HTTP clients.
        endpoint: The TEE endpoint URL to connect to.
    """

    def __init__(self, x402_client: x402Client, endpoint: str):
        self._x402_client = x402_client
        self._endpoint = endpoint
        self._active: ActiveTEE = self._connect()

    def get(self) -> ActiveTEE:
        """Return a snapshot of the current TEE connection."""
        return self._active

    def _connect(self) -> ActiveTEE:
        return ActiveTEE(
            endpoint=self._endpoint,
            http_client=x402HttpxClient(self._x402_client, verify=False),
            tee_id=None,
            payment_address=None,
        )

    def ensure_refresh_loop(self) -> None:
        """No-op — static connections don't refresh."""
        pass

    async def reconnect(self) -> None:
        """Rebuild the HTTP client (same endpoint)."""
        old_client = self._active.http_client
        self._active = self._connect()
        try:
            await old_client.aclose()
        except Exception:
            logger.debug("Failed to close previous HTTP client during reconnect.", exc_info=True)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._active.http_client.aclose()


class RegistryTEEConnection:
    """TEE connection resolved from the on-chain registry.

    Handles TLS certificate pinning, background health checks, and automatic
    failover when the current TEE becomes unavailable.

    Args:
        x402_client: Configured x402 payment client for creating HTTP clients.
        registry: TEERegistry for looking up active TEEs.
    """

    def __init__(self, x402_client: x402Client, registry: TEERegistry):
        self._x402_client = x402_client
        self._registry = registry

        self._refresh_lock = asyncio.Lock()
        self._refresh_task: Optional[asyncio.Task] = None

        self._active: ActiveTEE = self._connect()

    # ── Public API ──────────────────────────────────────────────────────

    def get(self) -> ActiveTEE:
        """Return a snapshot of the current TEE connection."""
        return self._active

    # ── Connection management ───────────────────────────────────────────

    def _resolve_tee(self):
        """Resolve TEE endpoint and metadata from the on-chain registry.

        Returns:
            The TEE object from the registry.

        Raises:
            RuntimeError: If the registry lookup fails.
            ValueError: If no active LLM proxy TEE is found.
        """
        try:
            tee = self._registry.get_llm_tee()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch LLM TEE endpoint from registry: {e}") from e

        if tee is None:
            raise ValueError("No active LLM proxy TEE found in the registry.")

        logger.info("Using TEE endpoint from registry: %s (teeId=%s)", tee.endpoint, tee.tee_id)
        return tee

    def _connect(self) -> ActiveTEE:
        """Resolve TEE from registry and create a secure HTTP client."""
        tee = self._resolve_tee()

        ssl_ctx = build_ssl_context_from_der(tee.tls_cert_der)
        return ActiveTEE(
            endpoint=tee.endpoint,
            http_client=x402HttpxClient(self._x402_client, verify=ssl_ctx),
            tee_id=tee.tee_id,
            payment_address=tee.payment_address,
        )

    async def reconnect(self) -> None:
        """Connect to a new TEE from the registry and rebuild the HTTP client."""
        async with self._refresh_lock:
            old_client = self._active.http_client
            self._active = self._connect()
            try:
                await old_client.aclose()
            except Exception:
                logger.debug("Failed to close previous HTTP client during TEE refresh.", exc_info=True)

    # ── Background health check ─────────────────────────────────────────

    def ensure_refresh_loop(self) -> None:
        """Start the background TEE refresh loop if not already running.

        Called lazily from async request methods since ``__init__`` is synchronous.
        """
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
            except asyncio.CancelledError:
                logger.debug("Background TEE health check cancelled; exiting loop.")
                raise
            except Exception:
                logger.warning("Background TEE health check failed; will retry next cycle.", exc_info=True)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cancel the background refresh loop and close the HTTP client."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        await self._active.http_client.aclose()

