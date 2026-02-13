## Run non-hosted LLMs with x402 transaction processing through OpenGradient routers
# Use opengradient version pip install opengradient==0.5.0a3

# This currently only works with model: gpt-4.1-2025-04-14 but we are quickly working on adding
# more models.

# x402 Settlement Modes:
# - SETTLE: Individual settlement with input/output hashes only (most privacy-preserving)
# - SETTLE_BATCH: Batch settlement for multiple inferences (most cost-efficient, default)
# - SETTLE_METADATA: Full model info, complete input/output data, and all inference metadata

import os

import opengradient as og
from x402v2.mechanisms.evm.constants import PERMIT2_ADDRESS
from web3 import Web3

BASE_OPG_ADDRESS = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"

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
    }
]

BASE_SEPOLIA_RPC = "https://sepolia.base.org"


def check_permit2_approval(client_address: str, network: str):
    """Check if the client address has approved Permit2 for OPG on Base testnet."""
    if network != "base-testnet":
        return

    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC))
    opg = w3.eth.contract(address=Web3.to_checksum_address(BASE_OPG_ADDRESS), abi=ERC20_ABI)

    allowance = opg.functions.allowance(
        Web3.to_checksum_address(client_address),
        Web3.to_checksum_address(PERMIT2_ADDRESS),
    ).call()

    print(f"Current OPG Permit Allowance: {allowance}")

    if allowance == 0:
        raise RuntimeError(
            f"ERROR: No Permit2 approval found for address {client_address}. "
            f"You need to approve the Permit2 contract ({PERMIT2_ADDRESS}) "
            f"to spend OPG ({BASE_OPG_ADDRESS}) on Base Sepolia before using x402 payments."
        )

network = "base-testnet"

client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
)

check_permit2_approval(client.alpha._wallet_account.address, network)

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What makes it good for beginners?"},
]

result = client.llm.chat(
    model=og.TEE_LLM.GPT_4_1_2025_04_14,
    messages=messages,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_METADATA,
    network=network
)
print(f"Response: {result.chat_output['content']}")
print(f"Payment hash: {result.payment_hash}")
