"""OPG token Permit2 approval utilities for x402 payments."""

import os
import logging
import time
from dataclasses import dataclass
from typing import Optional

from eth_account.account import LocalAccount
from web3 import Web3
from web3.types import ChecksumAddress
from x402.mechanisms.evm.constants import PERMIT2_ADDRESS

logger = logging.getLogger(__name__)

BASE_OPG_ADDRESS = "0xFbC2051AE2265686a469421b2C5A2D5462FbF5eB"
BASE_MAINNET_RPC = os.getenv("BASE_MAINNET_RPC", "https://base-rpc.publicnode.com")
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
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
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
    owner: ChecksumAddress,
    spender: ChecksumAddress,
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

        # Check native (ETH) balance to cover gas before building/sending tx.
        gas_price = w3.eth.gas_price
        gas_limit = int(estimated_gas * 1.2)
        total_cost = gas_limit * gas_price
        native_balance = w3.eth.get_balance(owner)
        if native_balance < int (total_cost * 1.1):  # Add a 10% buffer to avoid underestimating
            needed_eth = Web3.from_wei(total_cost, "ether")
            have_eth = Web3.from_wei(native_balance, "ether")
            raise RuntimeError(
                f"Insufficient native balance for gas on {w3.eth.chain_id}. "
                f"Required: {needed_eth:.6f} ETH, Available: {have_eth:.6f} ETH. "
                f"Please fund your wallet at the Base Sepolia faucet."
            )

        tx = approve_fn.build_transaction(
            {
                "from": owner,
                "nonce": nonce,
                "gas": gas_limit,
                "gasPrice": gas_price,
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
    w3 = Web3(Web3.HTTPProvider(BASE_MAINNET_RPC))
    token = w3.eth.contract(address=Web3.to_checksum_address(BASE_OPG_ADDRESS), abi=ERC20_ABI)
    spender = Web3.to_checksum_address(PERMIT2_ADDRESS)
    return w3, token, spender


def ensure_opg_approval(
    wallet_account: LocalAccount,
    min_allowance: float,
    approve_amount: Optional[float] = None,
) -> Permit2ApprovalResult:
    """Ensure the Permit2 allowance stays above a minimum threshold.

    Only sends an approval transaction when the current allowance drops
    below ``min_allowance``. When approval is needed, approves
    ``approve_amount`` (defaults to ``2 * min_allowance``) to create a
    buffer that survives multiple service restarts without re-approving.

    Best for backend servers that call this on startup::

        # On startup — only sends a tx when allowance < 5 OPG,
        # then approves 100 OPG so subsequent restarts are free.
        result = ensure_opg_approval(wallet, min_allowance=5.0, approve_amount=100.0)

    Args:
        wallet_account: The wallet account to check and approve from.
        min_allowance: The minimum acceptable allowance in OPG. A
            transaction is only sent when the current allowance is
            strictly below this value.
        approve_amount: The amount of OPG to approve when a transaction
            is needed. Defaults to ``2 * min_allowance``. Must be
            >= ``min_allowance``.

    Returns:
        Permit2ApprovalResult: Contains ``allowance_before``,
            ``allowance_after``, and ``tx_hash`` (None when no approval
            was needed).

    Raises:
        ValueError: If ``approve_amount`` is less than ``min_allowance``,
            or if the wallet has zero OPG balance.
        RuntimeError: If the approval transaction fails.
    """
    if approve_amount is None:
        approve_amount = min_allowance * 2
    if approve_amount < min_allowance:
        raise ValueError(f"approve_amount ({approve_amount}) must be >= min_allowance ({min_allowance})")

    w3, token, spender = _get_web3_and_contract()
    owner = Web3.to_checksum_address(wallet_account.address)
    allowance_before = token.functions.allowance(owner, spender).call()

    min_base = int(min_allowance * 10**18)
    approve_base = int(approve_amount * 10**18)

    if allowance_before >= min_base:
        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_before,
        )

    balance = token.functions.balanceOf(owner).call()
    if balance == 0:
        raise ValueError(f"Wallet {owner} has no OPG tokens. Fund the wallet before approving.")
    elif min_base > balance:
        raise ValueError(
            f"Wallet {owner} has insufficient OPG balance: has {balance / 10**18:.6f} OPG, "
            f"but the minimum required is {min_base / 10**18:.6f} OPG. Fund the wallet before approving."
        )
    elif approve_base > balance:
        logger.warning(
            "Requested approve_amount (%.6f OPG) exceeds wallet balance (%.6f OPG), capping approval to wallet balance",
            approve_amount,
            balance / 10**18,
        )
        approve_base = balance

    logger.info(
        "Permit2 allowance below minimum threshold (%s < %s), approving %s base units",
        allowance_before,
        min_base,
        approve_base,
    )
    return _send_approve_tx(wallet_account, w3, token, owner, spender, approve_base)
