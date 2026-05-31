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
        {"role": "user", "content": "What are the top technology headlines today? Cite your sources."},
    ]

    # Enable the provider's native web search with web_search=True. Each search is
    # billed per search on top of token usage. Web search is supported by OpenAI,
    # Anthropic, Google, and xAI models; other providers ignore the flag.
    result = await llm.chat(
        model=og.TEE_LLM.CLAUDE_SONNET_4_6,
        messages=messages,
        max_tokens=500,
        web_search=True,
    )
    print(result.chat_output["content"])


asyncio.run(main())
