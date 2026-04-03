"""
Multi-turn conversation example with OpenGradient TEE-verified LLM.

Demonstrates how to maintain conversation history across multiple turns,
enabling context-aware responses with full cryptographic verification
of every inference step.

Usage:
    export OG_PRIVATE_KEY="your_private_key"
    python examples/llm_multi_turn_conversation.py
"""

import asyncio
import os

import opengradient as og


def add_user_message(history: list, content: str) -> list:
    """Append a user message to the conversation history."""
    return history + [{"role": "user", "content": content}]


def add_assistant_message(history: list, content: str) -> list:
    """Append an assistant message to the conversation history."""
    return history + [{"role": "assistant", "content": content}]


async def chat_turn(
    llm: og.LLM,
    history: list,
    user_input: str,
    model: og.TEE_LLM = og.TEE_LLM.GEMINI_2_5_FLASH,
) -> tuple[str, list, str]:
    """
    Execute a single conversation turn.

    Args:
        llm: Initialized LLM client.
        history: Current conversation history.
        user_input: The user's message for this turn.
        model: TEE_LLM model to use.

    Returns:
        Tuple of (assistant_reply, updated_history, transaction_hash).
    """
    history = add_user_message(history, user_input)

    result = await llm.chat(
        model=model,
        messages=history,
        max_tokens=500,
        temperature=0.7,
    )
    assert isinstance(result, og.TextGenerationOutput)

    assert result.chat_output is not None
    reply = str(result.chat_output["content"])
    history = add_assistant_message(history, reply)

    return reply, history, result.transaction_hash


async def main():
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise ValueError("OG_PRIVATE_KEY environment variable is not set.")

    llm = og.LLM(private_key=private_key)
    llm.ensure_opg_approval(min_allowance=0.5)

    model = og.TEE_LLM.GEMINI_2_5_FLASH
    print(f"Model : {model.value}")
    print(f"Mode  : Multi-turn conversation with TEE verification")
    print("=" * 60)

    # System prompt sets the assistant persona for the whole conversation
    history = [
        {
            "role": "system",
            "content": ("You are a concise Python tutor. Give short, clear answers with code examples when helpful."),
        }
    ]

    # --- Turn 1 ---
    question_1 = "What is a Python decorator?"
    print(f"\nUser : {question_1}")

    reply_1, history, tx_1 = await chat_turn(llm, history, question_1, model)
    print(f"Assistant : {reply_1}")
    print(f"[tx: {tx_1}]")

    # --- Turn 2 — follow-up referencing Turn 1 ---
    question_2 = "Can you show me a real-world example of one?"
    print(f"\nUser : {question_2}")

    reply_2, history, tx_2 = await chat_turn(llm, history, question_2, model)
    print(f"Assistant : {reply_2}")
    print(f"[tx: {tx_2}]")

    # --- Turn 3 — deeper follow-up ---
    question_3 = "How would I stack two decorators on the same function?"
    print(f"\nUser : {question_3}")

    reply_3, history, tx_3 = await chat_turn(llm, history, question_3, model)
    print(f"Assistant : {reply_3}")
    print(f"[tx: {tx_3}]")

    # Summary
    print("\n" + "=" * 60)
    print(f"Total turns     : {len([m for m in history if m['role'] == 'user'])}")
    print(f"Total messages  : {len(history)}")
    print("Transaction hashes (verifiable on-chain):")
    for i, tx in enumerate([tx_1, tx_2, tx_3], 1):
        print(f"  Turn {i}: {tx}")


if __name__ == "__main__":
    asyncio.run(main())
