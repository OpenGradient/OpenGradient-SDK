from unittest.mock import MagicMock

import pytest

from opengradient.client.opg_token import (
    ensure_opg_allowance,
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


def _setup_allowance(mock_w3, allowance_value, balance=None):
    """Configure the mock contract to return a specific allowance and balance."""
    contract = MagicMock()
    contract.functions.allowance.return_value.call.return_value = allowance_value
    # Default balance to a large value so existing tests aren't affected
    contract.functions.balanceOf.return_value.call.return_value = balance if balance is not None else int(1_000_000 * 10**18)
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

    def test_approves_default_with_greater_amount(self, mock_wallet, mock_web3):
        """When no approve_amount given, approves 2x min_allowance."""
        contract = _setup_allowance(mock_web3, 0)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        approve_base = int(10.0 * 10**18)  # 2x of 5.0
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


class TestEnsureOpgAllowanceBalanceCheck:
    """Balance-aware approval capping."""

    def test_approve_amount_capped_to_balance(self, mock_wallet, mock_web3):
        """When approve_amount > balance >= min_allowance, cap to balance."""
        balance = int(0.1 * 10**18)
        contract = _setup_allowance(mock_web3, 0, balance=balance)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        # allowance calls: 1st for the check, 2nd in _send_approve_tx before, 3rd in post-tx poll
        contract.functions.allowance.return_value.call.side_effect = [0, 0, balance]

        result = ensure_opg_allowance(mock_wallet, min_allowance=0.1)

        # Default approve_amount would be 0.2, but balance is only 0.1 — capped
        args = contract.functions.approve.call_args[0]
        assert args[1] == balance
        assert result.tx_hash == "0xabc123"

    def test_no_cap_when_balance_sufficient(self, mock_wallet, mock_web3):
        """When balance >= approve_amount, no capping occurs."""
        balance = int(1.0 * 10**18)
        approve_base = int(0.2 * 10**18)
        contract = _setup_allowance(mock_web3, 0, balance=balance)
        _setup_approval_mocks(mock_web3, mock_wallet, contract)

        contract.functions.allowance.return_value.call.side_effect = [0, 0, approve_base]

        result = ensure_opg_allowance(mock_wallet, min_allowance=0.1)

        args = contract.functions.approve.call_args[0]
        assert args[1] == approve_base


class TestEnsureOpgAllowanceValidation:
    """Input validation."""

    def test_approve_amount_less_than_min_raises(self, mock_wallet, mock_web3):
        """approve_amount < min_allowance should raise ValueError."""
        _setup_allowance(mock_web3, 0)

        with pytest.raises(ValueError, match="approve_amount.*must be >= min_allowance"):
            ensure_opg_allowance(mock_wallet, min_allowance=10.0, approve_amount=5.0)


class TestAmountConversion:
    """Verify float-to-base-unit conversion."""

    def test_fractional_amount(self, mock_wallet, mock_web3):
        """Fractional OPG amounts convert correctly to 18-decimal base units."""
        expected_base = int(0.5 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = ensure_opg_allowance(mock_wallet, min_allowance=0.5)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None

    def test_large_amount(self, mock_wallet, mock_web3):
        """Large OPG amounts convert correctly."""
        expected_base = int(1000.0 * 10**18)
        _setup_allowance(mock_web3, expected_base)

        result = ensure_opg_allowance(mock_wallet, min_allowance=1000.0)

        assert result.allowance_before == expected_base
        assert result.tx_hash is None
