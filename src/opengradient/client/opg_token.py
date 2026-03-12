"""OPG token Permit2 approval utilities for x402 payments."""

from dataclasses import dataclass
from typing import Optional

from eth_account.account import LocalAccount
from web3 import Web3
from x402v2.mechanisms.evm.constants import PERMIT2_ADDRESS

BASE_OPG_ADDRESS = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"
BASE_SEPOLIA_RPC = "https://sepolia.base.org"

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


def ensure_opg_approval(wallet_account: LocalAccount, opg_amount: float) -> Permit2ApprovalResult:
    """Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

    Checks the current Permit2 allowance for the wallet. If the allowance
    is already >= the requested amount, returns immediately without sending
    a transaction. Otherwise, sends an ERC-20 approve transaction.

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
    amount_base = int(opg_amount * 10**18)

    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
    token = w3.eth.contract(address=Web3.to_checksum_address(BASE_OPG_ADDRESS), abi=ERC20_ABI)
    owner = Web3.to_checksum_address(wallet_account.address)
    spender = Web3.to_checksum_address(PERMIT2_ADDRESS)

    allowance_before = token.functions.allowance(owner, spender).call()

    # Only approve if the allowance is less than 10% of the requested amount
    if allowance_before >= amount_base * 0.1:
        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_before,
        )

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

        signed = wallet_account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt.status != 1:
            raise RuntimeError(f"Permit2 approval transaction reverted: {tx_hash.hex()}")

        allowance_after = token.functions.allowance(owner, spender).call()

        return Permit2ApprovalResult(
            allowance_before=allowance_before,
            allowance_after=allowance_after,
            tx_hash=tx_hash.hex(),
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to approve Permit2 for OPG: {e}")
