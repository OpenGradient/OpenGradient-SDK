import os
import time

import pytest
from eth_account import Account
from web3 import Web3

import opengradient as og
from opengradient.client.opg_token import BASE_OPG_ADDRESS, BASE_MAINNET_RPC

# Minimal ERC20 ABI for transfer
ERC20_TRANSFER_ABI = [
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "transfer",
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

# Amount of OPG tokens to fund the test account with
OPG_FUND_AMOUNT = 0.1
# Amount of ETH to fund the test account with (for gas)
ETH_FUND_AMOUNT = 0.0001


def _fund_account(funder_key: str, recipient_address: str):
    """Transfer ETH (for gas) and OPG tokens from the funder to the recipient."""
    w3 = Web3(Web3.HTTPProvider(BASE_MAINNET_RPC))
    funder = Account.from_key(funder_key)
    funder_addr = Web3.to_checksum_address(funder.address)
    recipient = Web3.to_checksum_address(recipient_address)

    # --- Transfer ETH for gas ---
    nonce = w3.eth.get_transaction_count(funder_addr, "pending")
    eth_tx = {
        "from": funder_addr,
        "to": recipient,
        "value": Web3.to_wei(ETH_FUND_AMOUNT, "ether"),
        "nonce": nonce,
        "gas": 21000,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    }
    signed_eth = funder.sign_transaction(eth_tx)
    eth_hash = w3.eth.send_raw_transaction(signed_eth.raw_transaction)
    eth_receipt = w3.eth.wait_for_transaction_receipt(eth_hash, timeout=120)
    if eth_receipt.status != 1:
        raise RuntimeError(f"ETH transfer failed: {eth_hash.hex()}")

    # --- Transfer OPG tokens ---
    token = w3.eth.contract(
        address=Web3.to_checksum_address(BASE_OPG_ADDRESS),
        abi=ERC20_TRANSFER_ABI,
    )
    opg_amount_base = int(OPG_FUND_AMOUNT * 10**18)
    transfer_fn = token.functions.transfer(recipient, opg_amount_base)

    estimated_gas = transfer_fn.estimate_gas({"from": funder_addr})
    opg_tx = transfer_fn.build_transaction(
        {
            "from": funder_addr,
            "nonce": nonce + 1,
            "gas": int(estimated_gas * 1.2),
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
        }
    )
    signed_opg = funder.sign_transaction(opg_tx)
    opg_hash = w3.eth.send_raw_transaction(signed_opg.raw_transaction)
    opg_receipt = w3.eth.wait_for_transaction_receipt(opg_hash, timeout=120)
    if opg_receipt.status != 1:
        raise RuntimeError(f"OPG transfer failed: {opg_hash.hex()}")

    # Wait for the recipient balances to be visible on the RPC node
    for _ in range(5):
        if w3.eth.get_balance(recipient) > 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Recipient ETH balance is still 0 after funding")

    for _ in range(10):
        if token.functions.balanceOf(recipient).call() > 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Recipient OPG token balance is still 0 after funding")


@pytest.fixture(scope="module")
def llm_client():
    """Create a fresh account, fund it, and return an initialized LLM client."""
    funder_key = os.environ.get("PRIVATE_KEY")
    if not funder_key:
        pytest.skip("PRIVATE_KEY environment variable is not set")

    account = Account.create()
    print(f"\nTest account: {account.address}")

    _fund_account(funder_key, account.address)
    print("Account funded with ETH and OPG")

    llm = og.LLM(private_key=account.key.hex())
    llm.ensure_opg_approval(min_allowance=OPG_FUND_AMOUNT, approve_amount=OPG_FUND_AMOUNT)
    print("Permit2 approval complete")

    # Wait for the approval to propagate on-chain
    time.sleep(2)

    return llm


@pytest.mark.asyncio(loop_scope="module")
async def test_chat(llm_client):
    messages = [
        {"role": "user", "content": "What is the capital of France? Reply in one word."},
    ]

    result = await llm_client.chat(
        model=og.TEE_LLM.GEMINI_2_5_FLASH,
        messages=messages,
        max_tokens=50,
        x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
    )

    assert result is not None
    assert "Paris" in result.chat_output["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming(llm_client):
    messages = [
        {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
    ]

    stream = await llm_client.chat(
        model=og.TEE_LLM.GEMINI_2_5_FLASH,
        messages=messages,
        max_tokens=50,
        x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    full_response = "".join(chunks)
    assert len(chunks) > 0, "Expected at least one streamed chunk"
    assert "4" in full_response
