"""Tests for StaticTEEConnection, RegistryTEEConnection, and ActiveTEE.

Covers TEE resolution, connection lifecycle, reconnect, background refresh,
and the ActiveTEE data snapshot.
"""

import asyncio
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.opengradient.client.tee_connection import (
    ActiveTEE,
    RegistryTEEConnection,
    StaticTEEConnection,
    _TEE_REFRESH_INTERVAL,
)
from src.opengradient.client.tee_registry import TEE_TYPE_LLM_PROXY


# ── Helpers ──────────────────────────────────────────────────────────


class FakeHTTPClient:
    """Minimal stand-in for x402HttpxClient."""

    def __init__(self, *_a, **_kw):
        self.closed = False

    async def aclose(self):
        self.closed = True


def _mock_x402_client():
    return MagicMock()


def _make_static_connection(
    *,
    endpoint: str = "https://test.tee",
    http_factory=None,
):
    """Build a StaticTEEConnection with patched externals."""
    factory = http_factory or FakeHTTPClient
    with patch(
        "src.opengradient.client.tee_connection.x402HttpxClient",
        side_effect=factory,
    ):
        return StaticTEEConnection(
            x402_client=_mock_x402_client(),
            endpoint=endpoint,
        )


def _make_registry_connection(*, registry=None, http_factory=None):
    """Build a RegistryTEEConnection with patched externals."""
    factory = http_factory or FakeHTTPClient
    with patch(
        "src.opengradient.client.tee_connection.x402HttpxClient",
        side_effect=factory,
    ):
        return RegistryTEEConnection(
            x402_client=_mock_x402_client(),
            registry=registry,
        )


def _mock_registry_with_tee(endpoint="https://tee.endpoint", tls_cert_der=None, tee_id="tee-1", payment_address="0xPay"):
    mock_reg = MagicMock()
    mock_tee = MagicMock()
    mock_tee.endpoint = endpoint
    mock_tee.tls_cert_der = tls_cert_der
    mock_tee.tee_id = tee_id
    mock_tee.payment_address = payment_address
    mock_reg.get_llm_tee.return_value = mock_tee
    return mock_reg


# ── ActiveTEE tests ─────────────────────────────────────────────────


class TestActiveTEE:
    def test_metadata_returns_dict(self):
        tee = ActiveTEE(
            endpoint="https://ep",
            http_client=MagicMock(),
            tee_id="tee-1",
            payment_address="0xPay",
        )
        meta = tee.metadata()

        assert meta == {
            "tee_id": "tee-1",
            "tee_endpoint": "https://ep",
            "tee_payment_address": "0xPay",
        }

    def test_metadata_with_none_values(self):
        tee = ActiveTEE(
            endpoint="https://ep",
            http_client=MagicMock(),
            tee_id=None,
            payment_address=None,
        )
        meta = tee.metadata()

        assert meta["tee_id"] is None
        assert meta["tee_payment_address"] is None

    def test_frozen_dataclass(self):
        tee = ActiveTEE(
            endpoint="https://ep",
            http_client=MagicMock(),
            tee_id="tee-1",
            payment_address="0xPay",
        )
        with pytest.raises(AttributeError):
            tee.endpoint = "https://other"


# ── StaticTEEConnection tests ───────────────────────────────────────


class TestStaticConnectionInit:
    def test_get_returns_active_tee(self):
        conn = _make_static_connection()
        active = conn.get()

        assert isinstance(active, ActiveTEE)
        assert active.endpoint == "https://test.tee"

    def test_sets_none_tee_id_and_payment(self):
        conn = _make_static_connection(endpoint="https://custom.url")
        active = conn.get()

        assert active.tee_id is None
        assert active.payment_address is None

    def test_ssl_verify_false(self):
        """Static connections disable TLS verification."""
        clients = []

        def capture_client(*args, **kwargs):
            c = FakeHTTPClient(*args, **kwargs)
            c._verify = kwargs.get("verify")
            clients.append(c)
            return c

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=capture_client,
        ):
            StaticTEEConnection(
                x402_client=_mock_x402_client(),
                endpoint="https://custom.url",
            )

        assert clients[0]._verify is False

    def test_ensure_refresh_loop_is_noop(self):
        conn = _make_static_connection()
        conn.ensure_refresh_loop()
        # No task created, no error raised


@pytest.mark.asyncio
class TestStaticReconnect:
    async def test_replaces_active_tee(self):
        clients_created = []

        def make_client(*args, **kwargs):
            c = FakeHTTPClient()
            clients_created.append(c)
            return c

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=make_client,
        ):
            conn = StaticTEEConnection(
                x402_client=_mock_x402_client(),
                endpoint="https://test.tee",
            )
            old_client = conn.get().http_client

            await conn.reconnect()

        assert conn.get().http_client is not old_client
        assert len(clients_created) == 2

    async def test_closes_old_client(self):
        conn = _make_static_connection()
        old_client = conn.get().http_client
        old_client.aclose = AsyncMock()

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            await conn.reconnect()

        old_client.aclose.assert_awaited_once()

    async def test_close_failure_is_swallowed(self):
        conn = _make_static_connection()
        old_client = conn.get().http_client
        old_client.aclose = AsyncMock(side_effect=OSError("already closed"))

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            # Should not raise
            await conn.reconnect()


@pytest.mark.asyncio
class TestStaticClose:
    async def test_closes_http_client(self):
        conn = _make_static_connection()
        conn.get().http_client.aclose = AsyncMock()

        await conn.close()

        conn.get().http_client.aclose.assert_awaited_once()


# ── RegistryTEEConnection._resolve_tee tests ────────────────────────


class TestResolveTee:
    def test_registry_returns_none_raises(self):
        mock_reg = MagicMock()
        mock_reg.get_llm_tee.return_value = None

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            with pytest.raises(ValueError, match="No active LLM proxy TEE"):
                RegistryTEEConnection(x402_client=_mock_x402_client(), registry=mock_reg)

    def test_registry_exception_wraps_in_runtime_error(self):
        mock_reg = MagicMock()
        mock_reg.get_llm_tee.side_effect = Exception("rpc down")

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch LLM TEE"):
                RegistryTEEConnection(x402_client=_mock_x402_client(), registry=mock_reg)

    def test_registry_success(self):
        mock_reg = _mock_registry_with_tee(
            endpoint="https://registry.tee",
            tls_cert_der=b"cert-bytes",
            tee_id="tee-42",
            payment_address="0xPay",
        )

        with (
            patch(
                "src.opengradient.client.tee_connection.x402HttpxClient",
                side_effect=FakeHTTPClient,
            ),
            patch(
                "src.opengradient.client.tee_connection.build_ssl_context_from_der",
                return_value=MagicMock(spec=ssl.SSLContext),
            ),
        ):
            conn = RegistryTEEConnection(x402_client=_mock_x402_client(), registry=mock_reg)

        assert conn.get().endpoint == "https://registry.tee"
        assert conn.get().tee_id == "tee-42"
        assert conn.get().payment_address == "0xPay"


# ── RegistryTEEConnection init / connect tests ──────────────────────


class TestRegistryConnectionInit:
    def test_get_returns_active_tee(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        active = conn.get()

        assert isinstance(active, ActiveTEE)
        assert active.endpoint == "https://tee.endpoint"

    def test_registry_path_creates_ssl_context(self):
        """When registry provides a TLS cert, an SSLContext should be built."""
        mock_reg = _mock_registry_with_tee(tls_cert_der=b"fake-der")
        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)

        with (
            patch(
                "src.opengradient.client.tee_connection.x402HttpxClient",
                side_effect=FakeHTTPClient,
            ),
            patch(
                "src.opengradient.client.tee_connection.build_ssl_context_from_der",
                return_value=mock_ssl_ctx,
            ) as mock_build,
        ):
            conn = RegistryTEEConnection(
                x402_client=_mock_x402_client(),
                registry=mock_reg,
            )

        mock_build.assert_called_once_with(b"fake-der")
        assert conn.get().endpoint == "https://tee.endpoint"
        assert conn.get().tee_id == "tee-1"


# ── RegistryTEEConnection reconnect tests ───────────────────────────


@pytest.mark.asyncio
class TestRegistryReconnect:
    async def test_replaces_active_tee(self):
        clients_created = []

        def make_client(*args, **kwargs):
            c = FakeHTTPClient()
            clients_created.append(c)
            return c

        mock_reg = _mock_registry_with_tee()

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=make_client,
        ):
            conn = RegistryTEEConnection(
                x402_client=_mock_x402_client(),
                registry=mock_reg,
            )
            old_client = conn.get().http_client

            await conn.reconnect()

        assert conn.get().http_client is not old_client
        assert len(clients_created) == 2

    async def test_closes_old_client(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        old_client = conn.get().http_client
        old_client.aclose = AsyncMock()

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            await conn.reconnect()

        old_client.aclose.assert_awaited_once()

    async def test_close_failure_is_swallowed(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        old_client = conn.get().http_client
        old_client.aclose = AsyncMock(side_effect=OSError("already closed"))

        with patch(
            "src.opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            # Should not raise
            await conn.reconnect()

    async def test_reconnect_is_serialized(self):
        """Concurrent reconnect calls should not race."""
        call_order = []

        original_connect = RegistryTEEConnection._connect

        def slow_connect(self):
            call_order.append("start")
            result = original_connect(self)
            call_order.append("end")
            return result

        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        with patch.object(RegistryTEEConnection, "_connect", slow_connect):
            await asyncio.gather(conn.reconnect(), conn.reconnect())

        # Both should complete without interleaving (lock serializes them)
        assert call_order == ["start", "end", "start", "end"]


# ── ensure_refresh_loop tests ────────────────────────────────────────


@pytest.mark.asyncio
class TestEnsureRefreshLoop:
    async def test_starts_task(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        conn.ensure_refresh_loop()

        assert conn._refresh_task is not None
        assert not conn._refresh_task.done()

        # Cleanup
        conn._refresh_task.cancel()
        try:
            await conn._refresh_task
        except asyncio.CancelledError:
            pass

    async def test_idempotent_when_already_running(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        conn.ensure_refresh_loop()
        first_task = conn._refresh_task

        conn.ensure_refresh_loop()

        assert conn._refresh_task is first_task

        # Cleanup
        conn._refresh_task.cancel()
        try:
            await conn._refresh_task
        except asyncio.CancelledError:
            pass


# ── _tee_refresh_loop tests ──────────────────────────────────────────


@pytest.mark.asyncio
class TestTeeRefreshLoop:
    async def test_no_reconnect_when_tee_still_active(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")

        active_tee = MagicMock()
        active_tee.tee_id = "tee-1"
        mock_reg.get_active_tees_by_type.return_value = [active_tee]

        conn = _make_registry_connection(registry=mock_reg)

        with patch.object(conn, "reconnect", new_callable=AsyncMock) as mock_reconnect:
            with patch(
                "src.opengradient.client.tee_connection.asyncio.sleep",
                side_effect=[None, asyncio.CancelledError],
            ):
                with pytest.raises(asyncio.CancelledError):
                    await conn._tee_refresh_loop()

            mock_reconnect.assert_not_called()

    async def test_reconnects_when_tee_no_longer_active(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")

        # Registry says a different TEE is now active
        other_tee = MagicMock()
        other_tee.tee_id = "tee-99"
        mock_reg.get_active_tees_by_type.return_value = [other_tee]

        conn = _make_registry_connection(registry=mock_reg)

        with patch.object(conn, "reconnect", new_callable=AsyncMock) as mock_reconnect:
            with patch(
                "src.opengradient.client.tee_connection.asyncio.sleep",
                side_effect=[None, asyncio.CancelledError],
            ):
                with pytest.raises(asyncio.CancelledError):
                    await conn._tee_refresh_loop()

            mock_reconnect.assert_awaited_once()

    async def test_registry_error_does_not_crash_loop(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")

        # First check fails, second check cancels
        mock_reg.get_active_tees_by_type.side_effect = [
            RuntimeError("rpc timeout"),
            asyncio.CancelledError,
        ]

        conn = _make_registry_connection(registry=mock_reg)

        with patch(
            "src.opengradient.client.tee_connection.asyncio.sleep",
            side_effect=[None, None],
        ):
            with pytest.raises(asyncio.CancelledError):
                await conn._tee_refresh_loop()

        # The loop survived the first error and ran a second iteration
        assert mock_reg.get_active_tees_by_type.call_count == 2


# ── RegistryTEEConnection close tests ───────────────────────────────


@pytest.mark.asyncio
class TestRegistryClose:
    async def test_closes_http_client(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        conn.get().http_client.aclose = AsyncMock()

        await conn.close()

        conn.get().http_client.aclose.assert_awaited_once()

    async def test_cancels_refresh_task(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        mock_task = MagicMock()
        conn._refresh_task = mock_task

        await conn.close()

        mock_task.cancel.assert_called_once()
        assert conn._refresh_task is None

    async def test_close_without_refresh_task(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        # Should not raise when no refresh task exists
        await conn.close()
