"""OPG token Permit2 approval utilities for x402 payments."""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Optional

from eth_account.account import LocalAccount
from web3 import Web3
from x402.mechanisms.evm.constants import PERMIT2_ADDRESS

logger = logging.getLogger(__name__)

BASE_OPG_ADDRESS = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"
BASE_SEPOLIA_RPC = "https://sepolia.base.org"
APPROVAL_TX_TIMEOUT = 120
ALLOWANCE_CONFIRMATION_TIMEOUT = 120
ALLOWANCE_POLL_INTERVAL = 1.0

ERC20_ABI = [
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class Permit2ApprovalResult:
    """Result of a Permit2 allowance check / approval.

    Attributes:
        allowance_before: The Permit2 allowance before the method ran.
        allowance_after: The Permit2 allowance after the method ran.
        tx_hash: Transaction hash of the approval, or None if no transaction was needed.
    """

    allowance_before: int
    allowance_after: int
    tx_hash: Optional[str] = None


def _send_approve_tx(
    wallet_account: LocalAccount,
    w3: Web3,
    token,
    owner: str,
    spender: str,
    amount_base: int,
) -> Permit2ApprovalResult:
    """Send an ERC-20 approve transaction and wait for confirmation.

    Args:
        wallet_account: The wallet to sign the transaction with.
        w3: Web3 instance connected to the RPC.
        token: The ERC-20 contract instance.
        owner: Checksummed owner address.
        spender: Checksummed spender (Permit2) address.
        amount_base: The amount to approve in base units (18 decimals).

    Returns:
        Permit2ApprovalResult with the before/after allowance and tx hash.

    Raises:
        RuntimeError: If the transaction reverts or fails.
    """
    allowance_before = token.functions.allowance(owner, spender).call()

    try:
        approve_fn = token.functions.approve(spender, amount_base)
        nonce = w3.eth.get_transaction_count(owner, "pending")
        estimated_gas = approve_fn.estimate_gas({"from": owner})

        tx = approve_fn.build_transaction(
            {
                "from": owner,
                "nonce": nonce,
                "gas": int(estimated_gas * 1.2),
                "gasPrice": w3.eth.gas_price,
                "chainId": w3.eth.chain_id,
            }
        )

        signed = wallet_account.sign_transaction(tx)  # type: ignore[arg-type]
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=APPROVAL_TX_TIMEOUT)

        if receipt.status != 1:  # type: ignore[attr-defined]
            raise RuntimeError(f"Permit2 approval transaction reverted: {tx_hash.hex()}")

        deadline = time.time() + ALLOWANCE_CONFIRMATION_TIMEOUT
        allowance_after = allowance_before
        while allowance_after < amount_base:
            allowance_after = token.functions.allowance(owner, spender).call()
            if allowance_after >= amount_base:
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    "Permit2 approval transaction was mined, but the updated allowance "
                    f"was not visible within {ALLOWANCE_CONFIRMATION_TIMEOUT} seconds: {tx_hash.hex()}"
                )
            time.sleep(ALLOWANCE_POLL_INTERVAL)

        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_after,
            tx_hash=tx_hash.hex(),
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to approve Permit2 for OPG: {e}")


def _get_web3_and_contract():
    """Create a Web3 instance and OPG token contract."""
    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
    token = w3.eth.contract(address=Web3.to_checksum_address(BASE_OPG_ADDRESS), abi=ERC20_ABI)
    spender = Web3.to_checksum_address(PERMIT2_ADDRESS)
    return w3, token, spender


def approve_opg(wallet_account: LocalAccount, opg_amount: float) -> Permit2ApprovalResult:
    """Approve Permit2 to spend ``opg_amount`` OPG if the current allowance is insufficient.

    Idempotent: if the current allowance is already >= ``opg_amount``, no
    transaction is sent.

    Best for one-off usage — scripts, notebooks, CLI tools::

        result = approve_opg(wallet, 5.0)

    Args:
        wallet_account: The wallet account to check and approve from.
        opg_amount: Number of OPG tokens to approve (e.g. ``5.0`` for 5 OPG).
            Converted to base units (18 decimals) internally.

    Returns:
        Permit2ApprovalResult: Contains ``allowance_before``,
            ``allowance_after``, and ``tx_hash`` (None when no approval
            was needed).

    Raises:
        RuntimeError: If the approval transaction fails.
    """
    amount_base = int(opg_amount * 10**18)

    w3, token, spender = _get_web3_and_contract()
    owner = Web3.to_checksum_address(wallet_account.address)

    allowance_before = token.functions.allowance(owner, spender).call()

    if allowance_before >= amount_base:
        logger.debug("Permit2 allowance already sufficient (%s >= %s), skipping approval", allowance_before, amount_base)
        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_before,
        )

    logger.info("Permit2 allowance insufficient (%s < %s), sending approval tx", allowance_before, amount_base)
    return _send_approve_tx(wallet_account, w3, token, owner, spender, amount_base)


def ensure_opg_allowance(
    wallet_account: LocalAccount,
    min_allowance: float,
    approve_amount: Optional[float] = None,
) -> Permit2ApprovalResult:
    """Ensure the Permit2 allowance stays above a minimum threshold.

    Only sends an approval transaction when the current allowance drops
    below ``min_allowance``. When approval is needed, approves
    ``approve_amount`` (defaults to ``10 * min_allowance``) to create a
    buffer that survives multiple service restarts without re-approving.

    Best for backend servers that call this on startup::

        # On startup — only sends a tx when allowance < 5 OPG,
        # then approves 100 OPG so subsequent restarts are free.
        result = ensure_opg_allowance(wallet, min_allowance=5.0, approve_amount=100.0)

    Args:
        wallet_account: The wallet account to check and approve from.
        min_allowance: The minimum acceptable allowance in OPG. A
            transaction is only sent when the current allowance is
            strictly below this value.
        approve_amount: The amount of OPG to approve when a transaction
            is needed. Defaults to ``10 * min_allowance``. Must be
            >= ``min_allowance``.

    Returns:
        Permit2ApprovalResult: Contains ``allowance_before``,
            ``allowance_after``, and ``tx_hash`` (None when no approval
            was needed).

    Raises:
        ValueError: If ``approve_amount`` is less than ``min_allowance``.
        RuntimeError: If the approval transaction fails.
    """
    if approve_amount is None:
        approve_amount = min_allowance * 10

    if approve_amount < min_allowance:
        raise ValueError(f"approve_amount ({approve_amount}) must be >= min_allowance ({min_allowance})")

    min_base = int(min_allowance * 10**18)
    approve_base = int(approve_amount * 10**18)

    w3, token, spender = _get_web3_and_contract()
    owner = Web3.to_checksum_address(wallet_account.address)

    allowance_before = token.functions.allowance(owner, spender).call()

    if allowance_before >= min_base:
        logger.debug(
            "Permit2 allowance above minimum threshold (%s >= %s), skipping approval",
            allowance_before,
            min_base,
        )
        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_before,
        )

    logger.info(
        "Permit2 allowance below minimum threshold (%s < %s), approving %s base units",
        allowance_before,
        min_base,
        approve_base,
    )
    return _send_approve_tx(wallet_account, w3, token, owner, spender, approve_base)


def ensure_opg_approval(wallet_account: LocalAccount, opg_amount: float) -> Permit2ApprovalResult:
    """Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

    .. deprecated::
        Use ``approve_opg`` for one-off approvals or
        ``ensure_opg_allowance`` for server-startup usage.

    Args:
        wallet_account: The wallet account to check and approve from.
        opg_amount: Minimum number of OPG tokens required (e.g. ``5.0``
            for 5 OPG). Converted to base units (18 decimals) internally.

    Returns:
        Permit2ApprovalResult: Contains ``allowance_before``,
            ``allowance_after``, and ``tx_hash`` (None when no approval
            was needed).

    Raises:
        RuntimeError: If the approval transaction fails.
    """
    warnings.warn(
        "ensure_opg_approval is deprecated. Use approve_opg for one-off approvals "
        "or ensure_opg_allowance for server-startup usage.",
        DeprecationWarning,
        stacklevel=2,
    )
    return approve_opg(wallet_account, opg_amount)
