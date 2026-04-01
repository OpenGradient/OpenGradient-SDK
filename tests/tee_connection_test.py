"""Tests for RegistryTEEConnection and ActiveTEE."""

import asyncio
import datetime
import os
import ssl
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from x402 import x402Client

from opengradient.client.tee_connection import (
    ActiveTEE,
    RegistryTEEConnection,
)
from opengradient.client.tee_registry import build_ssl_context_from_der


# ── Helpers ──────────────────────────────────────────────────────────


class FakeHTTPClient:
    """Minimal stand-in for x402HttpxClient."""

    def __init__(self, *_a, **_kw):
        self.closed = False

    async def aclose(self):
        self.closed = True


def _mock_x402_client():
    return MagicMock()


def _make_registry_connection(*, registry=None, http_factory=None):
    """Build a RegistryTEEConnection with patched externals."""
    factory = http_factory or FakeHTTPClient
    with (
        patch(
            "opengradient.client.tee_connection.x402HttpxClient",
            side_effect=factory,
        ),
        patch(
            "opengradient.client.tee_connection.build_ssl_context_from_der",
            return_value=MagicMock(spec=ssl.SSLContext),
        ),
    ):
        return RegistryTEEConnection(
            x402_client=_mock_x402_client(),
            registry=registry,
        )


def _mock_registry_with_tee(endpoint="https://tee.endpoint", tls_cert_der=b"fake-der", tee_id="tee-1", payment_address="0xPay"):
    mock_reg = MagicMock()
    mock_tee = MagicMock()
    mock_tee.endpoint = endpoint
    mock_tee.tls_cert_der = tls_cert_der
    mock_tee.tee_id = tee_id
    mock_tee.payment_address = payment_address
    mock_reg.get_llm_tee.return_value = mock_tee
    return mock_reg


class TestActiveTEE:
    def test_metadata_returns_dict(self):
        tee = ActiveTEE(
            endpoint="https://ep",
            http_client=MagicMock(),
            tee_id="tee-1",
            payment_address="0xPay",
        )
        assert tee.metadata() == {
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


@pytest.mark.asyncio
class TestRegistryTEEConnection:
    # ── init / resolve ───────────────────────────────────────────

    async def test_get_returns_active_tee(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        active = conn.get()

        assert isinstance(active, ActiveTEE)
        assert active.endpoint == "https://tee.endpoint"

    async def test_resolve_none_raises(self):
        mock_reg = MagicMock()
        mock_reg.get_llm_tee.return_value = None

        with patch(
            "opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            with pytest.raises(ValueError, match="No active LLM proxy TEE"):
                RegistryTEEConnection(x402_client=_mock_x402_client(), registry=mock_reg)

    async def test_resolve_exception_wraps_in_runtime_error(self):
        mock_reg = MagicMock()
        mock_reg.get_llm_tee.side_effect = Exception("rpc down")

        with patch(
            "opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch LLM TEE"):
                RegistryTEEConnection(x402_client=_mock_x402_client(), registry=mock_reg)

    async def test_resolve_success(self):
        mock_reg = _mock_registry_with_tee(
            endpoint="https://registry.tee",
            tee_id="tee-42",
            payment_address="0xPay",
        )
        conn = _make_registry_connection(registry=mock_reg)

        assert conn.get().endpoint == "https://registry.tee"
        assert conn.get().tee_id == "tee-42"
        assert conn.get().payment_address == "0xPay"

    async def test_builds_ssl_context_from_der(self):
        mock_reg = _mock_registry_with_tee(tls_cert_der=b"fake-der")
        mock_ssl_ctx = MagicMock(spec=ssl.SSLContext)

        with (
            patch(
                "opengradient.client.tee_connection.x402HttpxClient",
                side_effect=FakeHTTPClient,
            ),
            patch(
                "opengradient.client.tee_connection.build_ssl_context_from_der",
                return_value=mock_ssl_ctx,
            ) as mock_build,
        ):
            conn = RegistryTEEConnection(
                x402_client=_mock_x402_client(),
                registry=mock_reg,
            )

        mock_build.assert_called_once_with(b"fake-der")
        assert conn.get().tee_id == "tee-1"

    # ── reconnect ────────────────────────────────────────────────

    async def test_reconnect_replaces_active_tee(self):
        clients_created = []

        def make_client(*args, **kwargs):
            c = FakeHTTPClient()
            clients_created.append(c)
            return c

        mock_reg = _mock_registry_with_tee()

        with (
            patch(
                "opengradient.client.tee_connection.x402HttpxClient",
                side_effect=make_client,
            ),
            patch(
                "opengradient.client.tee_connection.build_ssl_context_from_der",
                return_value=MagicMock(spec=ssl.SSLContext),
            ),
        ):
            conn = RegistryTEEConnection(
                x402_client=_mock_x402_client(),
                registry=mock_reg,
            )
            old_client = conn.get().http_client
            await conn.reconnect()

        assert conn.get().http_client is not old_client
        assert len(clients_created) == 2

    async def test_reconnect_swallows_close_failure(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        conn.get().http_client.aclose = AsyncMock(side_effect=OSError("already closed"))

        with patch(
            "opengradient.client.tee_connection.x402HttpxClient",
            side_effect=FakeHTTPClient,
        ):
            await conn.reconnect()  # should not raise

    async def test_reconnect_is_serialized(self):
        call_order = []
        original_connect = RegistryTEEConnection._connect

        def slow_connect(self):
            call_order.append("start")
            result = original_connect(self)
            call_order.append("end")
            return result

        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        with (
            patch.object(RegistryTEEConnection, "_connect", slow_connect),
            patch(
                "opengradient.client.tee_connection.build_ssl_context_from_der",
                return_value=MagicMock(spec=ssl.SSLContext),
            ),
            patch(
                "opengradient.client.tee_connection.x402HttpxClient",
                side_effect=FakeHTTPClient,
            ),
        ):
            await asyncio.gather(conn.reconnect(), conn.reconnect())

        assert call_order == ["start", "end", "start", "end"]

    # ── refresh loop ─────────────────────────────────────────────

    async def test_ensure_refresh_loop_starts_task(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        conn.ensure_refresh_loop()

        assert conn._refresh_task is not None
        assert not conn._refresh_task.done()

        conn._refresh_task.cancel()
        try:
            await conn._refresh_task
        except asyncio.CancelledError:
            pass

    async def test_ensure_refresh_loop_is_idempotent(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)

        conn.ensure_refresh_loop()
        first_task = conn._refresh_task
        conn.ensure_refresh_loop()

        assert conn._refresh_task is first_task

        conn._refresh_task.cancel()
        try:
            await conn._refresh_task
        except asyncio.CancelledError:
            pass

    async def test_refresh_loop_skips_when_tee_still_active(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")
        active_tee = MagicMock()
        active_tee.tee_id = "tee-1"
        mock_reg.get_active_tees_by_type.return_value = [active_tee]

        conn = _make_registry_connection(registry=mock_reg)

        with patch.object(conn, "reconnect", new_callable=AsyncMock) as mock_reconnect:
            with patch(
                "opengradient.client.tee_connection.asyncio.sleep",
                side_effect=[None, asyncio.CancelledError],
            ):
                with pytest.raises(asyncio.CancelledError):
                    await conn._tee_refresh_loop()

            mock_reconnect.assert_not_called()

    async def test_refresh_loop_reconnects_when_tee_gone(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")
        other_tee = MagicMock()
        other_tee.tee_id = "tee-99"
        mock_reg.get_active_tees_by_type.return_value = [other_tee]

        conn = _make_registry_connection(registry=mock_reg)

        with patch.object(conn, "reconnect", new_callable=AsyncMock) as mock_reconnect:
            with patch(
                "opengradient.client.tee_connection.asyncio.sleep",
                side_effect=[None, asyncio.CancelledError],
            ):
                with pytest.raises(asyncio.CancelledError):
                    await conn._tee_refresh_loop()

            mock_reconnect.assert_awaited_once()

    async def test_refresh_loop_survives_registry_error(self):
        mock_reg = _mock_registry_with_tee(tee_id="tee-1")
        mock_reg.get_active_tees_by_type.side_effect = [
            RuntimeError("rpc timeout"),
            asyncio.CancelledError,
        ]

        conn = _make_registry_connection(registry=mock_reg)

        with patch(
            "opengradient.client.tee_connection.asyncio.sleep",
            side_effect=[None, None],
        ):
            with pytest.raises(asyncio.CancelledError):
                await conn._tee_refresh_loop()

        assert mock_reg.get_active_tees_by_type.call_count == 2

    # ── close ────────────────────────────────────────────────────

    async def test_close_closes_http_client(self):
        mock_reg = _mock_registry_with_tee()
        conn = _make_registry_connection(registry=mock_reg)
        conn.get().http_client.aclose = AsyncMock()

        await conn.close()

        conn.get().http_client.aclose.assert_awaited_once()

    async def test_close_cancels_refresh_task(self):
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

        await conn.close()  # should not raise


# ── TLS certificate verification (real handshake) ────────────────────


def _make_self_signed_cert():
    """Generate a self-signed cert. Returns (der_bytes, pem_cert_bytes, pem_key_bytes)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return (
        cert.public_bytes(serialization.Encoding.DER),
        cert.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()),
    )


@pytest.fixture
async def tls_server():
    """Spin up a local TLS server with a self-signed cert."""
    der, pem_cert, pem_key = _make_self_signed_cert()

    cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    try:
        cert_file.write(pem_cert)
        cert_file.close()
        key_file.write(pem_key)
        key_file.close()

        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_ctx.load_cert_chain(cert_file.name, key_file.name)

        async def handler(reader, writer):
            await reader.read(4096)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=server_ctx)
        port = server.sockets[0].getsockname()[1]

        yield {"port": port, "der": der}

        server.close()
        await server.wait_closed()
    finally:
        os.unlink(cert_file.name)
        os.unlink(key_file.name)


def _registry_with_real_cert(tls_server):
    """Return a mock registry that serves the local TLS server's real DER cert."""
    return _mock_registry_with_tee(
        endpoint=f"https://127.0.0.1:{tls_server['port']}",
        tls_cert_der=tls_server["der"],
        tee_id="tee-real",
        payment_address="0xRealPay",
    )


@pytest.mark.asyncio
class TestTlsCertVerification:
    """End-to-end TLS handshake tests through RegistryTEEConnection.

    A real local TLS server is started with a self-signed cert.  The registry
    mock returns that cert's DER bytes.  RegistryTEEConnection._connect() runs
    its real code (build_ssl_context_from_der → x402HttpxClient(verify=ctx))
    so the full cert-pinning path is exercised with an actual TLS handshake.
    """

    async def test_connect_succeeds_with_matching_cert(self, tls_server):
        mock_reg = _registry_with_real_cert(tls_server)
        conn = RegistryTEEConnection(x402_client=x402Client(), registry=mock_reg)

        resp = await conn.get().http_client.get(f"https://127.0.0.1:{tls_server['port']}/")
        assert resp.status_code == 200
        assert conn.get().tee_id == "tee-real"
        assert conn.get().payment_address == "0xRealPay"
        await conn.close()

    async def test_connect_fails_with_wrong_cert(self, tls_server):
        wrong_der, _, _ = _make_self_signed_cert()  # different key pair
        mock_reg = _mock_registry_with_tee(
            endpoint=f"https://127.0.0.1:{tls_server['port']}",
            tls_cert_der=wrong_der,
        )
        conn = RegistryTEEConnection(x402_client=x402Client(), registry=mock_reg)

        with pytest.raises(httpx.ConnectError):
            await conn.get().http_client.get(f"https://127.0.0.1:{tls_server['port']}/")
        await conn.close()

    async def test_connect_fails_with_no_cert_pinning(self, tls_server):
        """Without a pinned cert (tls_cert_der=None), build_ssl_context_from_der
        rejects the None value and connection construction fails."""
        mock_reg = _mock_registry_with_tee(
            endpoint=f"https://127.0.0.1:{tls_server['port']}",
            tls_cert_der=None,
        )
        with pytest.raises(TypeError):
            RegistryTEEConnection(x402_client=x402Client(), registry=mock_reg)

    async def test_reconnect_picks_up_new_cert(self, tls_server):
        """After reconnect, the connection uses the freshly-resolved cert."""
        mock_reg = _registry_with_real_cert(tls_server)
        conn = RegistryTEEConnection(x402_client=x402Client(), registry=mock_reg)

        await conn.reconnect()

        resp = await conn.get().http_client.get(f"https://127.0.0.1:{tls_server['port']}/")
        assert resp.status_code == 200
        await conn.close()
