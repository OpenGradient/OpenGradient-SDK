import asyncio
import logging
import os

import opengradient as og

logging.basicConfig()
logging.getLogger("opengradient").setLevel(logging.DEBUG)


async def main():
    llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))
    llm.ensure_opg_approval(opg_amount=1)

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    result = await llm.chat(
        model=og.TEE_LLM.GEMINI_2_5_FLASH,
        messages=messages,
        max_tokens=300,
        x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
    )
    print(result.chat_output["content"])


asyncio.run(main())
