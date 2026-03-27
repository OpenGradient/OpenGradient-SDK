"""
LLM tool/function calling example.

Usage:
    export OG_PRIVATE_KEY="your_private_key"
    python examples/llm_tool_calling.py
"""

import asyncio
import os

import opengradient as og


async def main():
    llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))

    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'San Francisco'",
                        },
                        "state": {
                            "type": "string",
                            "description": "The two-letter abbreviation for the state, e.g. 'CA'",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit for temperature",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What's the weather like in Dallas, Texas? Give me the temperature in fahrenheit."},
    ]

    # One-time Permit2 approval for OPG spending (idempotent)
    llm.approve_opg(opg_amount=0.1)

    print("Testing Gemini tool calls...")
    print(f"Model: {og.TEE_LLM.GEMINI_2_5_FLASH_LITE}")
    print(f"Messages: {messages}")
    print(f"Tools: {tools}")
    print("-" * 50)

    result = await llm.chat(
        model=og.TEE_LLM.GEMINI_2_5_FLASH_LITE,
        messages=messages,
        tools=tools,
        max_tokens=200,
    )

    print(f"Finish reason: {result.finish_reason}")
    print(f"Chat output: {result.chat_output}")
    print(f"Transaction hash: {result.transaction_hash}")


asyncio.run(main())
