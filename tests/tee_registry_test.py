import os
import ssl
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.opengradient.client.tee_registry import (
    TEE_TYPE_LLM_PROXY,
    TEE_TYPE_VALIDATOR,
    TEEEndpoint,
    TEERegistry,
    build_ssl_context_from_der,
)


# --- Helpers ---


def _make_tee_info(
    endpoint="https://tee.example.com",
    payment_address="0xPayment",
    tls_cert_der=b"\x01\x02\x03",
    active=True,
):
    """Build a tuple matching the TEEInfo struct order from the contract."""
    return (
        "0xOwner",          # owner
        payment_address,    # paymentAddress
        endpoint,           # endpoint
        b"pubkey",          # publicKey
        tls_cert_der,       # tlsCertificate
        b"pcrhash",         # pcrHash
        0,                  # teeType
        active,             # active
        1000,               # registeredAt
        2000,               # lastUpdatedAt
    )


def _make_self_signed_der() -> bytes:
    """Generate a minimal self-signed DER certificate for testing."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
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
    return cert.public_bytes(serialization.Encoding.DER)


# --- Fixtures ---


@pytest.fixture
def mock_contract():
    """Create a TEERegistry with a mocked Web3 contract."""
    with (
        patch("src.opengradient.client.tee_registry.Web3") as mock_web3_cls,
        patch("src.opengradient.client.tee_registry.get_abi") as mock_get_abi,
    ):
        mock_get_abi.return_value = []
        mock_web3 = MagicMock()
        mock_web3_cls.return_value = mock_web3
        mock_web3_cls.HTTPProvider.return_value = MagicMock()
        mock_web3_cls.to_checksum_address.side_effect = lambda x: x

        contract = MagicMock()
        mock_web3.eth.contract.return_value = contract

        registry = TEERegistry(rpc_url="http://localhost:8545", registry_address="0xRegistry")
        yield registry, contract


# --- TEERegistry Tests ---


class TestGetActiveTeesByType:
    def test_returns_active_tees(self, mock_contract):
        registry, contract = mock_contract

        tee_id = b"\xaa" * 32
        contract.functions.getTEEsByType.return_value.call.return_value = [tee_id]
        contract.functions.getTEE.return_value.call.return_value = _make_tee_info()

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)

        assert len(result) == 1
        assert result[0].tee_id == tee_id.hex()
        assert result[0].endpoint == "https://tee.example.com"
        assert result[0].payment_address == "0xPayment"
        assert result[0].tls_cert_der == b"\x01\x02\x03"

    def test_skips_inactive_tees(self, mock_contract):
        registry, contract = mock_contract

        tee_id = b"\xbb" * 32
        contract.functions.getTEEsByType.return_value.call.return_value = [tee_id]
        contract.functions.getTEE.return_value.call.return_value = _make_tee_info(active=False)

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert len(result) == 0

    def test_skips_tee_with_empty_endpoint(self, mock_contract):
        registry, contract = mock_contract

        tee_id = b"\xcc" * 32
        contract.functions.getTEEsByType.return_value.call.return_value = [tee_id]
        contract.functions.getTEE.return_value.call.return_value = _make_tee_info(endpoint="")

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert len(result) == 0

    def test_skips_tee_with_empty_cert(self, mock_contract):
        registry, contract = mock_contract

        tee_id = b"\xdd" * 32
        contract.functions.getTEEsByType.return_value.call.return_value = [tee_id]
        contract.functions.getTEE.return_value.call.return_value = _make_tee_info(tls_cert_der=b"")

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert len(result) == 0

    def test_returns_empty_on_rpc_failure(self, mock_contract):
        registry, contract = mock_contract

        contract.functions.getTEEsByType.return_value.call.side_effect = Exception("RPC error")

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert result == []

    def test_skips_individual_tee_on_lookup_failure(self, mock_contract):
        registry, contract = mock_contract

        good_id = b"\xaa" * 32
        bad_id = b"\xbb" * 32
        contract.functions.getTEEsByType.return_value.call.return_value = [bad_id, good_id]

        def get_tee_side_effect(tee_id):
            mock = MagicMock()
            if tee_id == bad_id:
                mock.call.side_effect = Exception("lookup failed")
            else:
                mock.call.return_value = _make_tee_info()
            return mock

        contract.functions.getTEE.side_effect = get_tee_side_effect

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert len(result) == 1
        assert result[0].tee_id == good_id.hex()

    def test_multiple_active_tees(self, mock_contract):
        registry, contract = mock_contract

        ids = [b"\x01" * 32, b"\x02" * 32, b"\x03" * 32]
        contract.functions.getTEEsByType.return_value.call.return_value = ids

        def get_tee_side_effect(tee_id):
            mock = MagicMock()
            mock.call.return_value = _make_tee_info(
                endpoint=f"https://tee-{tee_id.hex()[:4]}.example.com"
            )
            return mock

        contract.functions.getTEE.side_effect = get_tee_side_effect

        result = registry.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        assert len(result) == 3

    def test_validator_type_label(self, mock_contract):
        """Ensure validator type queries work the same way."""
        registry, contract = mock_contract

        contract.functions.getTEEsByType.return_value.call.return_value = []

        result = registry.get_active_tees_by_type(TEE_TYPE_VALIDATOR)
        assert result == []
        contract.functions.getTEEsByType.assert_called_once_with(TEE_TYPE_VALIDATOR)


class TestGetLlmTee:
    def test_returns_first_active_tee(self, mock_contract):
        registry, contract = mock_contract

        ids = [b"\x01" * 32, b"\x02" * 32]
        contract.functions.getTEEsByType.return_value.call.return_value = ids
        contract.functions.getTEE.return_value.call.return_value = _make_tee_info()

        result = registry.get_llm_tee()

        assert result is not None
        assert result.tee_id == ids[0].hex()

    def test_returns_none_when_no_tees(self, mock_contract):
        registry, contract = mock_contract

        contract.functions.getTEEsByType.return_value.call.return_value = []

        result = registry.get_llm_tee()
        assert result is None

    def test_queries_llm_proxy_type(self, mock_contract):
        registry, contract = mock_contract

        contract.functions.getTEEsByType.return_value.call.return_value = []
        registry.get_llm_tee()

        contract.functions.getTEEsByType.assert_called_once_with(TEE_TYPE_LLM_PROXY)


# --- build_ssl_context_from_der Tests ---


class TestBuildSslContextFromDer:
    def test_returns_ssl_context(self):
        der_cert = _make_self_signed_der()
        ctx = build_ssl_context_from_der(der_cert)

        assert isinstance(ctx, ssl.SSLContext)

    def test_hostname_check_disabled(self):
        der_cert = _make_self_signed_der()
        ctx = build_ssl_context_from_der(der_cert)

        assert ctx.check_hostname is False

    def test_cert_required(self):
        der_cert = _make_self_signed_der()
        ctx = build_ssl_context_from_der(der_cert)

        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_rejects_invalid_der(self):
        with pytest.raises(Exception):
            build_ssl_context_from_der(b"not-a-valid-cert")
