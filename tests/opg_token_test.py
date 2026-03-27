import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opengradient.client.opg_token import (
    approve_opg,
    ensure_opg_allowance,
    ensure_opg_approval,
)

OWNER_ADDRESS = "0x1234567890abcdef1234567890ABCDEF12345678"
SPENDER_ADDRESS = "0xAABBCCDDEEFF00112233445566778899AABBCCDD"


@pytest.fixture
def mock_wallet():
    wallet = MagicMock()
    wallet.address = OWNER_ADDRESS
    return wallet


@pytest.fixture
def mock_web3(monkeypatch):
    """Patch Web3 and PERMIT2_ADDRESS so no real RPC calls are made."""
    mock_w3 = MagicMock()

    # Make Web3.to_checksum_address pass through
    mock_web3_cls = MagicMock()
    mock_web3_cls.return_value = mock_w3
    mock_web3_cls.to_checksum_address = lambda addr: addr
    mock_web3_cls.HTTPProvider.return_value = MagicMock()

    monkeypatch.setattr("opengradient.client.opg_token.Web3", mock_web3_cls)
    monkeypatch.setattr("opengradient.client.opg_token.PERMIT2_ADDRESS", SPENDER_ADDRESS)

    return mock_w3


def _setup_allowance(mock_w3, allowance_value):
    """Configure the mock contract to return a specific allowance."""
    contract = MagicMock()
    contract.functions.allowance.return_value.call.return_value = allowance_value
    mock_w3.eth.contract.return_value = contract
    return contract


def _setup_approval_mocks(mock_web3, mock_wallet, contract):
    """Set up common mocks for approval transactions."""
    approve_fn = MagicMock()
    contract.functions.approve.return_value = approve_fn
    approve_fn.estimate_gas.return_value = 50_000
    approve_fn.build_transaction.return_value = {"mock": "tx"}

    mock_web3.eth.get_transaction_count.return_value = 7
    mock_web3.eth.gas_price = 1_000_000_000
    mock_web3.eth.chain_id = 84532

    signed = MagicMock()
    signed.raw_transaction = b"\x00"
    mock_wallet.sign_transaction.return_value = signed

    tx_hash = MagicMock()
    tx_hash.hex.return_value = "0xabc123"
    mock_web3.eth.send_raw_transaction.return_value = tx_hash

    receipt = SimpleNamespace(status=1)
    mock_web3.eth.wait_for_transaction_receipt.return_value = receipt

    return approve_fn, tx_hash


# ── approve_opg tests ───────────────────────────────────────────────


class TestApproveOpgSkips:
    """Cases where the existing allowance is sufficient."""

    def test_exact_allowance_skips_tx(self, mock_wallet, mock_web3):
        """When allowance == requested amount, no transaction is sent."""
        amount = 5.0
        amount_base = int(amount * 10**18)
        _setup_allowance(mock_web3, amount_base)

        result = approve_opg(mock_wallet, amount)

        assert result.allowance_before == amount_base
        assert result.allowance_after == amount_base
        assert result.tx_hash is None

    def test_excess_allowance_skips_tx(self, mock_wallet, mock_web3):
        """When allowance > requested amount, no transaction is sent."""
        amount_base = int(5.0 * 10**18)
        _setup_allowance(mock_web3, amount_base * 2)

        result = approve_opg(mock_wallet, 5.0)

        assert result.allowance_before == amount_base * 2
        assert result.tx_hash is None

    def test_zero_amount_with_zero_allowance_skips(self, mock_wallet, mock_web3):
        """Requesting 0 tokens with 0 allowance should skip (0 >= 0)."""
        _setup_allowance(mock_web3, 0)

        result = approve_opg(mock_wallet, 0.0)

        assert result.tx_hash is None


class TestApproveOpgSendsTx:
    """Cases where allowance is insufficient and a transaction is sent."""

    def test_approval_sent_when_allowance_insufficient(self, mock_wallet, mock_web3):
        """When allowance < requested, an approve tx is sent."""
        amount = 5.0
        amount_base = int(amount * 10**18)
        contract = _setup_allowance(mock_web3, 0)
        approve_fn, _ = _setup_approval_mocks(mock_web3, mock_wallet, contract)

        # Side effects: 1st call in approve_opg check, 2nd in _send_approve_tx before,
        # 3rd in post-tx poll
        contract.functions.allowance.return_value.call.side_effect = [0, 0, amount_base]

        result = approve_opg(mock_wallet, amount)

        assert result.allowance_before == 0
        assert result.allowance_after == amount_base
        assert result.tx_hash == "0xabc123"

        # Verify the approve was called with the right amount
        contract.functions.approve.assert_called_once()
        args = contract.functions.approve.call_args[0]
        assert args[1] == amount_base

    def test_gas_estimate_has_20_percent_buffer(self, mock_wallet, mock_web3):
        """Gas limit should be estimatedGas * 1.2."""
        contract = _setup_allowance(mock_web3, 0)
        approve_fn, _ = _setup_approval_mocks(mock_web3, mock_wallet, contract)

        contract.functions.allowance.return_value.call.side_effect = [0, 0, int(1 * 10**18)]

        approve_opg(mock_wallet, 1.0)

        tx_dict = approve_fn.build_transaction.call_args[0][0]
        assert tx_dict["gas"] == int(50_000 * 1.2)

    def test_waits_for_allowance_update_after_receipt(self, mock_wallet, mock_web3, monkeypatch):
        """After a successful receipt, poll allowance until the updated value is visible."""
        monkeypatch.setattr("opengradient.client.opg_token.ALLOWANCE_POLL_INTERVAL", 0)
        contract = _setup_allowance(mock_web3, 0)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        amount_base = int(1.0 * 10**18)
        contract.functions.allowance.return_value.call.side_effect = [0, 0, 0, amount_base]

        result = approve_opg(mock_wallet, 1.0)

        assert result.allowance_before == 0
        assert result.allowance_after == amount_base


class TestApproveOpgErrors:
    """Error handling paths."""

    def test_reverted_tx_raises(self, mock_wallet, mock_web3):
        """A reverted transaction raises RuntimeError."""
        contract = _setup_allowance(mock_web3, 0)

        approve_fn = MagicMock()
        contract.functions.approve.return_value = approve_fn
        approve_fn.estimate_gas.return_value = 50_000

        mock_web3.eth.get_transaction_count.return_value = 0
        mock_web3.eth.gas_price = 1_000_000_000
        mock_web3.eth.chain_id = 84532

        signed = MagicMock()
        signed.raw_transaction = b"\x00"
        mock_wallet.sign_transaction.return_value = signed

        tx_hash = MagicMock()
        tx_hash.hex.return_value = "0xfailed"
        mock_web3.eth.send_raw_transaction.return_value = tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = SimpleNamespace(status=0)

        with pytest.raises(RuntimeError, match="reverted"):
            approve_opg(mock_wallet, 5.0)

    def test_generic_exception_wrapped(self, mock_wallet, mock_web3):
        """Non-RuntimeError exceptions are wrapped in RuntimeError."""
        contract = _setup_allowance(mock_web3, 0)

        approve_fn = MagicMock()
        contract.functions.approve.return_value = approve_fn
        approve_fn.estimate_gas.side_effect = ConnectionError("RPC unavailable")

        mock_web3.eth.get_transaction_count.return_value = 0

        with pytest.raises(RuntimeError, match="Failed to approve Permit2 for OPG"):
            approve_opg(mock_wallet, 5.0)

    def test_runtime_error_not_double_wrapped(self, mock_wallet, mock_web3):
        """RuntimeError raised inside the try block should propagate as-is."""
        contract = _setup_allowance(mock_web3, 0)

        approve_fn = MagicMock()
        contract.functions.approve.return_value = approve_fn
        approve_fn.estimate_gas.return_value = 50_000

        mock_web3.eth.get_transaction_count.return_value = 0
        mock_web3.eth.gas_price = 1_000_000_000
        mock_web3.eth.chain_id = 84532

        signed = MagicMock()
        signed.raw_transaction = b"\x00"
        mock_wallet.sign_transaction.return_value = signed

        tx_hash = MagicMock()
        tx_hash.hex.return_value = "0xfailed"
        mock_web3.eth.send_raw_transaction.return_value = tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = SimpleNamespace(status=0)

        with pytest.raises(RuntimeError, match="reverted") as exc_info:
            approve_opg(mock_wallet, 5.0)

        assert "Failed to approve" not in str(exc_info.value)


# ── ensure_opg_allowance tests ──────────────────────────────────────


class TestEnsureOpgAllowanceSkips:
    """Cases where the existing allowance exceeds the minimum threshold."""

    def test_above_minimum_skips(self, mock_wallet, mock_web3):
        """When allowance >= min_allowance, no transaction is sent."""
        min_base = int(5.0 * 10**18)
        _setup_allowance(mock_web3, min_base * 2)

        result = ensure_opg_allowance(mock_wallet, min_allowance=5.0)

        assert result.tx_hash is None
        assert result.allowance_before == min_base * 2

    def test_exact_minimum_skips(self, mock_wallet, mock_web3):
        """When allowance == min_allowance, no transaction is sent."""
        min_base = int(5.0 * 10**18)
        _setup_allowance(mock_web3, min_base)

        result = ensure_opg_allowance(mock_wallet, min_allowance=5.0)

        assert result.tx_hash is None


class TestEnsureOpgAllowanceSendsTx:
    """Cases where allowance is below the minimum and a tx is needed."""

    def test_approves_default_10x_amount(self, mock_wallet, mock_web3):
        """When no approve_amount given, approves 10x min_allowance."""
        contract = _setup_allowance(mock_web3, 0)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        approve_base = int(50.0 * 10**18)  # 10x of 5.0
        contract.functions.allowance.return_value.call.side_effect = [0, 0, approve_base]

        result = ensure_opg_allowance(mock_wallet, min_allowance=5.0)

        assert result.tx_hash == "0xabc123"
        # Verify approve was called with 10x amount
        args = contract.functions.approve.call_args[0]
        assert args[1] == approve_base

    def test_approves_custom_amount(self, mock_wallet, mock_web3):
        """When approve_amount is specified, uses that exact amount."""
        contract = _setup_allowance(mock_web3, 0)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        approve_base = int(100.0 * 10**18)
        contract.functions.allowance.return_value.call.side_effect = [0, 0, approve_base]

        result = ensure_opg_allowance(mock_wallet, min_allowance=5.0, approve_amount=100.0)

        assert result.tx_hash == "0xabc123"
        args = contract.functions.approve.call_args[0]
        assert args[1] == approve_base

    def test_no_tx_on_restart_when_above_min(self, mock_wallet, mock_web3):
        """Simulates server restart: allowance is above min but below approve_amount."""
        # After first approval of 100 OPG, some was consumed leaving 60 OPG.
        # min_allowance=5.0 so no tx should be sent.
        remaining = int(60.0 * 10**18)
        _setup_allowance(mock_web3, remaining)

        result = ensure_opg_allowance(mock_wallet, min_allowance=5.0, approve_amount=100.0)

        assert result.tx_hash is None
        assert result.allowance_before == remaining


class TestEnsureOpgAllowanceValidation:
    """Input validation."""

    def test_approve_amount_less_than_min_raises(self, mock_wallet, mock_web3):
        """approve_amount < min_allowance should raise ValueError."""
        _setup_allowance(mock_web3, 0)

        with pytest.raises(ValueError, match="approve_amount.*must be >= min_allowance"):
            ensure_opg_allowance(mock_wallet, min_allowance=10.0, approve_amount=5.0)


# ── ensure_opg_approval (deprecated) tests ──────────────────────────


class TestEnsureOpgApprovalDeprecated:
    """The old function still works but emits a deprecation warning."""

    def test_emits_deprecation_warning(self, mock_wallet, mock_web3):
        """ensure_opg_approval should emit a DeprecationWarning."""
        _setup_allowance(mock_web3, int(10 * 10**18))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ensure_opg_approval(mock_wallet, 5.0)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    def test_delegates_to_approve_opg(self, mock_wallet, mock_web3):
        """ensure_opg_approval should produce the same result as approve_opg."""
        amount_base = int(5.0 * 10**18)
        _setup_allowance(mock_web3, amount_base)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = ensure_opg_approval(mock_wallet, 5.0)

        assert result.allowance_before == amount_base
        assert result.tx_hash is None


# ── Amount conversion tests ─────────────────────────────────────────


class TestAmountConversion:
    """Verify float-to-base-unit conversion."""

    def test_fractional_amount(self, mock_wallet, mock_web3):
        """Fractional OPG amounts convert correctly to 18-decimal base units."""
        expected_base = int(0.5 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = approve_opg(mock_wallet, 0.5)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None

    def test_large_amount(self, mock_wallet, mock_web3):
        """Large OPG amounts convert correctly."""
        expected_base = int(1000.0 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = approve_opg(mock_wallet, 1000.0)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None
