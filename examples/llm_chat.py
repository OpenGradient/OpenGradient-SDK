import os

import opengradient as og

client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
)
client.llm.ensure_opg_approval(opg_amount=2)

messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

result = client.llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=messages,
    max_tokens=300,
    x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
)
print(result.chat_output["content"])
