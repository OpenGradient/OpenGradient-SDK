from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from opengradient.client.exceptions import OpenGradientError
from opengradient.client.opg_token import (
    Permit2ApprovalResult,
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


class TestEnsureOpgApprovalSkips:
    """Cases where the existing allowance is sufficient."""

    def test_exact_allowance_skips_tx(self, mock_wallet, mock_web3):
        """When allowance == requested amount, no transaction is sent."""
        amount = 5.0
        amount_base = int(amount * 10**18)
        _setup_allowance(mock_web3, amount_base)

        result = ensure_opg_approval(mock_wallet, amount)

        assert result.allowance_before == amount_base
        assert result.allowance_after == amount_base
        assert result.tx_hash is None

    def test_excess_allowance_skips_tx(self, mock_wallet, mock_web3):
        """When allowance > requested amount, no transaction is sent."""
        amount_base = int(5.0 * 10**18)
        _setup_allowance(mock_web3, amount_base * 2)

        result = ensure_opg_approval(mock_wallet, 5.0)

        assert result.allowance_before == amount_base * 2
        assert result.tx_hash is None

    def test_zero_amount_with_zero_allowance_skips(self, mock_wallet, mock_web3):
        """Requesting 0 tokens with 0 allowance should skip (0 >= 0)."""
        _setup_allowance(mock_web3, 0)

        result = ensure_opg_approval(mock_wallet, 0.0)

        assert result.tx_hash is None


class TestEnsureOpgApprovalSendsTx:
    """Cases where allowance is insufficient and a transaction is sent."""

    def test_approval_sent_when_allowance_insufficient(self, mock_wallet, mock_web3):
        """When allowance < requested, an approve tx is sent."""
        amount = 5.0
        amount_base = int(amount * 10**18)
        contract = _setup_allowance(mock_web3, 0)

        # Set up the approval transaction mocks
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

        # After approval the allowance call returns the new value
        contract.functions.allowance.return_value.call.side_effect = [0, amount_base]

        result = ensure_opg_approval(mock_wallet, amount)

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
        tx_hash.hex.return_value = "0x0"
        mock_web3.eth.send_raw_transaction.return_value = tx_hash
        mock_web3.eth.wait_for_transaction_receipt.return_value = SimpleNamespace(status=1)

        contract.functions.allowance.return_value.call.side_effect = [0, int(1 * 10**18)]

        ensure_opg_approval(mock_wallet, 1.0)

        tx_dict = approve_fn.build_transaction.call_args[0][0]
        assert tx_dict["gas"] == int(50_000 * 1.2)


class TestEnsureOpgApprovalErrors:
    """Error handling paths."""

    def test_reverted_tx_raises(self, mock_wallet, mock_web3):
        """A reverted transaction raises OpenGradientError."""
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

        with pytest.raises(OpenGradientError, match="reverted"):
            ensure_opg_approval(mock_wallet, 5.0)

    def test_generic_exception_wrapped(self, mock_wallet, mock_web3):
        """Non-OpenGradientError exceptions are wrapped in OpenGradientError."""
        contract = _setup_allowance(mock_web3, 0)

        approve_fn = MagicMock()
        contract.functions.approve.return_value = approve_fn
        approve_fn.estimate_gas.side_effect = RuntimeError("RPC unavailable")

        mock_web3.eth.get_transaction_count.return_value = 0

        with pytest.raises(OpenGradientError, match="Failed to approve Permit2 for OPG"):
            ensure_opg_approval(mock_wallet, 5.0)

    def test_opengradient_error_not_double_wrapped(self, mock_wallet, mock_web3):
        """OpenGradientError raised inside the try block should propagate as-is."""
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

        with pytest.raises(OpenGradientError, match="reverted") as exc_info:
            ensure_opg_approval(mock_wallet, 5.0)

        # Should be the original error, not wrapped
        assert "Failed to approve" not in str(exc_info.value)


class TestAmountConversion:
    """Verify float-to-base-unit conversion."""

    def test_fractional_amount(self, mock_wallet, mock_web3):
        """Fractional OPG amounts convert correctly to 18-decimal base units."""
        expected_base = int(0.5 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = ensure_opg_approval(mock_wallet, 0.5)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None

    def test_large_amount(self, mock_wallet, mock_web3):
        """Large OPG amounts convert correctly."""
        expected_base = int(1000.0 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = ensure_opg_approval(mock_wallet, 1000.0)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None


class TestPermit2ApprovalResult:
    """Dataclass behavior."""

    def test_default_tx_hash_is_none(self):
        result = Permit2ApprovalResult(allowance_before=100, allowance_after=200)
        assert result.tx_hash is None

    def test_fields(self):
        result = Permit2ApprovalResult(allowance_before=0, allowance_after=500, tx_hash="0xabc")
        assert result.allowance_before == 0
        assert result.allowance_after == 500
        assert result.tx_hash == "0xabc"
