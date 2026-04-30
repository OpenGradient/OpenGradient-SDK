import asyncio
import logging
import os

import opengradient as og

logging.basicConfig()
logging.getLogger("opengradient").setLevel(logging.DEBUG)


async def main():
    llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))
    llm.ensure_opg_approval(min_allowance=0.1)

    messages = [
        {"role": "user", "content": "What model are you?"},
    ]

    # Run inference with full public settlement
    result = await llm.chat(
        model=og.TEE_LLM.GPT_5_5,
        messages=messages,
        max_tokens=300,
        x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
    )
    print(result.chat_output["content"])

    # Print inference settlement details
    print("\n" + "=" * 40)
    tx_hash = result.data_settlement_transaction_hash
    if tx_hash:
        print(f"Settlement tx: {tx_hash}")
        print(f"Explorer: https://explorer.opengradient.ai/tx/{tx_hash}?tab=index")
    else:
        print("No settlement tx hash returned")

asyncio.run(main())
