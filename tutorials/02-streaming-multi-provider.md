# Streaming Multi-Provider Chat with Settlement Modes

Most LLM APIs lock you into a single provider. If you want to switch from OpenAI to
Anthropic or Google, you rewrite your integration, change authentication, and update
response parsing. OpenGradient gives you a single unified API that wraps every major
provider -- swap one enum value and everything else stays the same.

But the real differentiator is settlement. Every inference call settles on-chain via
the x402 payment protocol, producing a cryptographic receipt you can use for
compliance, billing audits, or dispute resolution. You choose how much data goes
on-chain: just a hash (privacy), a batch digest (cost savings), or full metadata
(complete transparency).

This tutorial walks through the `client.llm.chat()` API, covering non-streaming and
streaming responses, multi-provider switching, settlement modes, and function calling
-- all in one place.

## Prerequisites

```bash
pip install opengradient
```

Export your OpenGradient private key:

```bash
export OG_PRIVATE_KEY="0x..."
```

> **Faucet:** Get free OPG tokens on Base Sepolia at https://faucet.opengradient.ai/
>
> All x402 LLM payments currently settle on Base Sepolia using OPG tokens. If you see
> x402 payment errors, make sure your wallet has sufficient OPG on Base Sepolia.

## Step 1: Basic Non-Streaming Chat

Start with the simplest possible call -- send a message and get a response. Before
making any LLM calls, approve OPG token spending for the x402 payment protocol using
`ensure_opg_approval`. This is idempotent -- it checks the current Permit2 allowance
and only sends a transaction if the allowance is below the requested amount.

```python
import os
import sys
import opengradient as og

private_key = os.environ.get("OG_PRIVATE_KEY")
if not private_key:
    print("Error: set the OG_PRIVATE_KEY environment variable.")
    sys.exit(1)

client = og.init(private_key=private_key)

# Approve OPG spending for x402 payments (one-time, idempotent).
client.llm.ensure_opg_approval(opg_amount=5)

result = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "What is the x402 payment protocol?"}],
    max_tokens=200,
    temperature=0.0,
)

# result is a TextGenerationOutput dataclass
print(result.chat_output.get("content", ""))  # The model's text response
print(result.finish_reason)            # "stop", "length", or "tool_calls"
print(result.payment_hash)             # On-chain x402 receipt
```

The `chat_output` dictionary follows the OpenAI message format: it has `role`,
`content`, and optionally `tool_calls` keys. The `payment_hash` is your on-chain
settlement proof -- every call gets one.

## Step 2: Switch Providers with One Line

The `model` parameter accepts any `og.TEE_LLM` enum value. Swap the model and
everything else -- message format, authentication, response parsing -- stays
identical.

```python
# OpenAI
result_openai = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Hello from OpenAI!"}],
    max_tokens=100,
)

# Anthropic
result_anthropic = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Hello from Anthropic!"}],
    max_tokens=100,
)

# Google
result_google = client.llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=[{"role": "user", "content": "Hello from Google!"}],
    max_tokens=100,
)

# xAI
result_xai = client.llm.chat(
    model=og.TEE_LLM.GROK_4,
    messages=[{"role": "user", "content": "Hello from xAI!"}],
    max_tokens=100,
)

```

This makes A/B testing trivial -- run the same prompt across providers and compare
quality, latency, and cost without changing any infrastructure.

## Step 3: Enable Streaming

For chat UIs and real-time applications, pass `stream=True` to get tokens as they
are generated. The return value changes from a `TextGenerationOutput` to a generator
that yields `StreamChunk` objects.

```python
stream = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "Explain TEEs in one paragraph."},
    ],
    max_tokens=300,
    temperature=0.0,
    stream=True,
)

for chunk in stream:
    # Each chunk has a choices list. The first choice's delta
    # contains the incremental content for this token.
    delta = chunk.choices[0].delta

    if delta.content:
        print(delta.content, end="", flush=True)

    # The final chunk has a finish_reason and optional usage stats.
    if chunk.is_final:
        print(f"\n\nModel: {chunk.model}")
        if chunk.usage:
            print(f"Tokens used: {chunk.usage.total_tokens}")
```

The `StreamChunk` dataclass has these fields:

| Field | Type | Description |
|-------|------|-------------|
| `choices` | `List[StreamChoice]` | Incremental choices (usually one) |
| `model` | `str` | Model identifier |
| `usage` | `StreamUsage` or `None` | Token counts (final chunk only) |
| `is_final` | `bool` | `True` when the stream is ending |

Each `StreamChoice` contains a `StreamDelta` with optional `content`, `role`, and
`tool_calls` fields.

## Step 4: Settlement Modes

Every LLM call settles on-chain. The `x402_settlement_mode` parameter controls the
privacy/cost/transparency trade-off:

| Mode | On-Chain Data | Use Case |
|------|--------------|----------|
| `SETTLE` | Input/output hashes only | **Privacy** -- prove execution without revealing content |
| `SETTLE_BATCH` | Batch digest of multiple calls | **Cost efficiency** -- lower gas per inference (default) |
| `SETTLE_METADATA` | Full model, input, output, metadata | **Transparency** -- complete audit trail |

```python
# Privacy-first: only hashes stored on-chain
result_private = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Sensitive query here."}],
    max_tokens=100,
    x402_settlement_mode=og.x402SettlementMode.SETTLE,
)
print(f"Payment hash (SETTLE): {result_private.payment_hash}")

# Cost-efficient: batched settlement (this is the default)
result_batch = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Regular query."}],
    max_tokens=100,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)
print(f"Payment hash (SETTLE_BATCH): {result_batch.payment_hash}")

# Full transparency: everything on-chain
result_transparent = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Auditable query."}],
    max_tokens=100,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_METADATA,
)
print(f"Payment hash (SETTLE_METADATA): {result_transparent.payment_hash}")
```

All three calls return a `payment_hash` you can look up on-chain. The difference is
how much detail the on-chain record contains. Store these hashes if you need an
audit trail -- they are the on-chain receipts for each inference call.

## Step 5: Function Calling

You can pass tools to `client.llm.chat()` in the standard OpenAI function-calling
format. This works with any model that supports tool use.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_token_price",
            "description": "Get the current USD price of a cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Token ticker symbol, e.g. ETH, BTC.",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]

result = client.llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=[{"role": "user", "content": "What's the current price of ETH?"}],
    max_tokens=200,
    tools=tools,
    tool_choice="auto",
)

if result.chat_output.get("tool_calls"):
    # The model decided to call a tool instead of responding with text.
    # We check for tool_calls in the message rather than relying on finish_reason,
    # since the exact finish_reason string may vary by provider.
    for tc in result.chat_output["tool_calls"]:
        func = tc["function"]
        print(f"Tool: {func['name']}, Args: {func['arguments']}")
else:
    print(result.chat_output.get("content", ""))
```

When the model returns tool calls, execute the requested functions locally,
then send the results back in a follow-up `client.llm.chat()` call with a `"tool"`
role message. See **Tutorial 3** for a complete multi-turn tool-calling loop.

## Complete Code

```python
"""Streaming Multi-Provider Chat -- complete working example."""

import os
import sys
import opengradient as og

# ── Initialize ────────────────────────────────────────────────────────────
private_key = os.environ.get("OG_PRIVATE_KEY")
if not private_key:
    print("Error: set the OG_PRIVATE_KEY environment variable.")
    sys.exit(1)

client = og.init(private_key=private_key)

# Approve OPG spending for x402 payments (idempotent -- skips if already approved).
client.llm.ensure_opg_approval(opg_amount=5)

PROMPT = "Explain what a Trusted Execution Environment is in two sentences."

# ── Multi-provider comparison ─────────────────────────────────────────────
models = [
    ("GPT-5",             og.TEE_LLM.GPT_5),
    ("Claude Sonnet 4.6", og.TEE_LLM.CLAUDE_SONNET_4_6),
    ("Gemini 2.5 Flash",  og.TEE_LLM.GEMINI_2_5_FLASH),
    ("Grok 4",            og.TEE_LLM.GROK_4),
]

for name, model in models:
    try:
        result = client.llm.chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=200,
            temperature=0.0,
        )
        print(f"[{name}] {result.chat_output.get('content', '')}")
        print(f"  Payment hash: {result.payment_hash}\n")
    except Exception as e:
        print(f"[{name}] Error: {e}\n")

# ── Streaming ─────────────────────────────────────────────────────────────
print("--- Streaming from GPT-5 ---")
stream = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "What is x402? Keep it under 50 words."}],
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# ── Settlement modes ──────────────────────────────────────────────────────
for mode_name, mode in [
    ("SETTLE",          og.x402SettlementMode.SETTLE),
    ("SETTLE_BATCH",    og.x402SettlementMode.SETTLE_BATCH),
    ("SETTLE_METADATA", og.x402SettlementMode.SETTLE_METADATA),
]:
    try:
        r = client.llm.chat(
            model=og.TEE_LLM.CLAUDE_SONNET_4_6,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=50,
            x402_settlement_mode=mode,
        )
        print(f"[{mode_name}] payment_hash={r.payment_hash}")
    except Exception as e:
        print(f"[{mode_name}] Error: {e}")

# ── Function calling ──────────────────────────────────────────────────────
tools = [{
    "type": "function",
    "function": {
        "name": "get_token_price",
        "description": "Get the current USD price of a cryptocurrency.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Token ticker, e.g. ETH."}
            },
            "required": ["symbol"],
        },
    },
}]

result = client.llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=[{"role": "user", "content": "What is the price of ETH?"}],
    max_tokens=200,
    tools=tools,
    tool_choice="auto",
)

if result.chat_output.get("tool_calls"):
    for tc in result.chat_output["tool_calls"]:
        func = tc["function"]
        print(f"Tool call: {func['name']}({func['arguments']})")
else:
    print(result.chat_output.get("content", ""))
```

## Next Steps

- **Build a chat UI**: Use the streaming API with a web framework to build a
  real-time chat interface backed by verifiable inference.
- **Add tool calling**: See **Tutorial 3** for a full multi-turn agent loop with
  tool dispatch and result feeding.
- **Build an agent**: See **Tutorial 1** to combine LangChain with on-chain model
  tools for a fully verifiable AI agent.
