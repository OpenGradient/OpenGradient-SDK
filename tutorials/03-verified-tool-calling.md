# Tool-Calling Agent with Verified Reasoning

When an LLM agent decides to call a function, you typically have no way to prove
*why* it made that decision. The model's reasoning is opaque -- it could have been
influenced by a prompt injection, a poisoned context, or a simple hallucination.
You only see the tool call and hope the model made the right choice.

OpenGradient changes this by running every LLM call inside a Trusted Execution
Environment (TEE). The model's reasoning -- including its decision to call a tool,
which tool to call, and what arguments to pass -- is cryptographically attested and
settled on-chain via the x402 payment protocol. The tool executions themselves run
locally, but the AI reasoning that drives them is verifiable.

In this tutorial you will build a personal crypto portfolio assistant that can look
up holdings, check prices, and calculate risk metrics. The agent uses a multi-turn
conversation loop where the LLM decides which tools to call and synthesizes the
results into actionable advice.

## Prerequisites

```bash
pip install opengradient
```

You need an OpenGradient private key funded with test tokens:

```bash
export OG_PRIVATE_KEY="0x..."
```

> **Faucet:** Get free OPG tokens on Base Sepolia at https://faucet.opengradient.ai/
>
> All x402 LLM payments currently settle on Base Sepolia using OPG tokens. If you see
> x402 payment errors, make sure your wallet has sufficient OPG on Base Sepolia.

## Step 1: Initialize the Client

Before making any LLM calls, approve OPG token spending for the x402 payment
protocol. The `ensure_opg_approval` method is idempotent -- it checks the current
Permit2 allowance and only sends a transaction if the allowance is below the
requested amount.

```python
import json
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
```

## Step 2: Define Local Tool Implementations

These are the functions the agent can call. In a real application they would query a
database, exchange API, or on-chain contract. Here we use hardcoded data so the
tutorial runs without external dependencies.

```python
PORTFOLIO = {
    "ETH":  {"amount": 5.0,  "avg_cost": 1950.00},
    "BTC":  {"amount": 0.25, "avg_cost": 42000.00},
    "SOL":  {"amount": 100,  "avg_cost": 95.00},
}

CURRENT_PRICES = {"ETH": 2120.50, "BTC": 67250.00, "SOL": 148.30}
VOLATILITY     = {"ETH": 0.65,    "BTC": 0.55,     "SOL": 0.85}

def get_portfolio() -> str:
    """Return the user's portfolio holdings as a JSON string."""
    rows = [{"token": t, "amount": v["amount"], "avg_cost_usd": v["avg_cost"]}
            for t, v in PORTFOLIO.items()]
    return json.dumps(rows, indent=2)

def get_price(token: str) -> str:
    """Return the current price for a single token."""
    token = token.upper()
    price = CURRENT_PRICES.get(token)
    if price is None:
        return json.dumps({"error": f"Unknown token: {token}"})
    return json.dumps({"token": token, "price_usd": price})

def calculate_risk(token: str) -> str:
    """Return simplified risk metrics for a token."""
    token = token.upper()
    vol = VOLATILITY.get(token)
    if vol is None:
        return json.dumps({"error": f"Unknown token: {token}"})
    holding = PORTFOLIO.get(token)
    price = CURRENT_PRICES.get(token)
    position_value = holding["amount"] * price if holding and price else 0
    daily_vol = vol / (252 ** 0.5)
    var_95 = position_value * daily_vol * 1.645
    return json.dumps({
        "token": token,
        "annualized_volatility": f"{vol:.0%}",
        "position_value_usd": round(position_value, 2),
        "daily_var_95_usd": round(var_95, 2),
    })

# Dispatch table for executing tool calls by name.
TOOL_DISPATCH = {
    "get_portfolio":  lambda **kw: get_portfolio(),
    "get_price":      lambda **kw: get_price(kw["token"]),
    "calculate_risk": lambda **kw: calculate_risk(kw["token"]),
}
```

## Step 3: Define Tools in OpenAI Function-Calling Format

Each tool is described as a JSON object with `type`, `function.name`,
`function.description`, and a `function.parameters` JSON Schema. This format is
the same one used by the OpenAI API and is supported across all OpenGradient
providers.

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Returns the user's current crypto portfolio holdings.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Returns the current USD price for a cryptocurrency token.",
            "parameters": {
                "type": "object",
                "properties": {
                    "token": {"type": "string", "description": "Token ticker, e.g. ETH, BTC, SOL."},
                },
                "required": ["token"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk",
            "description": "Calculates risk metrics: volatility, position value, and daily VaR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "token": {"type": "string", "description": "Token ticker, e.g. ETH, BTC, SOL."},
                },
                "required": ["token"],
            },
        },
    },
]
```

## Step 4: Pass Tools to `client.llm.chat`

Pass the `tools` list and `tool_choice` parameter to any `client.llm.chat()` call.

```python
result = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[
        {"role": "system", "content": "You are a crypto portfolio assistant."},
        {"role": "user", "content": "What's my portfolio worth?"},
    ],
    max_tokens=600,
    temperature=0.0,
    tools=TOOLS,
    # "auto" lets the model decide whether to call a tool or respond with text.
    # "none" forces a text-only response.
    tool_choice="auto",
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)
```

When the model decides to call a tool, `result.finish_reason` will be `"tool_calls"`
(following the OpenAI convention). The tool call details are in
`result.chat_output["tool_calls"]`.

## Step 5: Handle Tool Calls and Feed Results Back

The core pattern for a tool-calling agent is a loop:

1. Send messages + tools to the LLM
2. If `finish_reason == "tool_calls"`, execute each tool locally
3. Append the assistant message AND tool results to the conversation
4. Call the LLM again so it can see the tool output
5. Repeat until the model responds with a regular text message

```python
def run_agent(client: og.Client, user_query: str) -> str:
    """Run a multi-turn tool-calling agent loop."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful crypto portfolio assistant. Use the provided "
                "tools to look up holdings, prices, and risk metrics. Always check "
                "the portfolio and relevant prices before giving advice. Be concise."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    max_iterations = 5  # Safety limit to prevent runaway loops

    for i in range(max_iterations):
        print(f"\n  [Round {i + 1}] Calling LLM...")

        try:
            result = client.llm.chat(
                model=og.TEE_LLM.GPT_5,
                messages=messages,
                max_tokens=600,
                temperature=0.0,
                tools=TOOLS,
                tool_choice="auto",
                x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
            )
        except Exception as e:
            print(f"  LLM call failed: {e}")
            return f"Error: {e}"

        print(f"  Finish reason: {result.finish_reason}")

        # -- The model wants to call one or more tools --
        # "tool_calls" finish reason follows the OpenAI convention and is used
        # consistently across all providers on OpenGradient.
        if result.finish_reason == "tool_calls":
            tool_calls = result.chat_output.get("tool_calls", [])

            # Append the assistant's message (contains tool_calls) to history.
            messages.append(result.chat_output)

            for tc in tool_calls:
                func = tc.get("function", tc)
                tool_name = func["name"]
                tool_args = json.loads(func.get("arguments", "{}"))
                call_id = tc.get("id", "")

                print(f"  -> Tool call: {tool_name}({tool_args})")

                handler = TOOL_DISPATCH.get(tool_name)
                tool_result = handler(**tool_args) if handler else json.dumps({"error": f"Unknown tool: {tool_name}"})

                print(f"  <- Result: {tool_result[:120]}...")

                # Feed the result back as a "tool" role message.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result,
                })
            continue

        # -- The model produced a regular text response --
        content = result.chat_output.get("content", "")
        print(f"\n  [Final answer received]")
        return content

    return "Agent reached maximum iterations without a final answer."
```

## Step 6: Run the Agent

```python
if __name__ == "__main__":
    queries = [
        "What does my portfolio look like right now? What's the total value?",
        "Which of my holdings has the highest risk? Should I rebalance?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"USER: {query}")
        print("=" * 70)
        answer = run_agent(client, query)
        print(f"\nASSISTANT: {answer}")
```

Every LLM call in the loop above was TEE-verified and settled on-chain. The tool
executions ran locally, but the model's reasoning about *when* and *how* to call
tools was cryptographically attested.

## Complete Code

```python
"""Tool-Calling Agent with Verified Reasoning -- complete working example."""

import json
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

# ── Mock data ─────────────────────────────────────────────────────────────
PORTFOLIO      = {"ETH": {"amount": 5.0, "avg_cost": 1950.00},
                  "BTC": {"amount": 0.25, "avg_cost": 42000.00},
                  "SOL": {"amount": 100, "avg_cost": 95.00}}
CURRENT_PRICES = {"ETH": 2120.50, "BTC": 67250.00, "SOL": 148.30}
VOLATILITY     = {"ETH": 0.65,    "BTC": 0.55,     "SOL": 0.85}

def get_portfolio() -> str:
    rows = [{"token": t, "amount": v["amount"], "avg_cost_usd": v["avg_cost"]}
            for t, v in PORTFOLIO.items()]
    return json.dumps(rows, indent=2)

def get_price(token: str) -> str:
    token = token.upper()
    price = CURRENT_PRICES.get(token)
    return json.dumps({"error": f"Unknown token: {token}"}) if price is None else json.dumps({"token": token, "price_usd": price})

def calculate_risk(token: str) -> str:
    token = token.upper()
    vol = VOLATILITY.get(token)
    if vol is None:
        return json.dumps({"error": f"Unknown token: {token}"})
    holding, price = PORTFOLIO.get(token), CURRENT_PRICES.get(token)
    pv = holding["amount"] * price if holding and price else 0
    return json.dumps({"token": token, "annualized_volatility": f"{vol:.0%}",
                       "position_value_usd": round(pv, 2),
                       "daily_var_95_usd": round(pv * (vol / 252**0.5) * 1.645, 2)})

TOOL_DISPATCH = {
    "get_portfolio":  lambda **kw: get_portfolio(),
    "get_price":      lambda **kw: get_price(kw["token"]),
    "calculate_risk": lambda **kw: calculate_risk(kw["token"]),
}

# ── Tool definitions ──────────────────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {"name": "get_portfolio",
        "description": "Returns the user's crypto portfolio holdings.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_price",
        "description": "Returns the current USD price for a cryptocurrency.",
        "parameters": {"type": "object", "properties": {
            "token": {"type": "string", "description": "Token ticker, e.g. ETH."}},
            "required": ["token"]}}},
    {"type": "function", "function": {"name": "calculate_risk",
        "description": "Calculates risk metrics: volatility, position value, and daily VaR.",
        "parameters": {"type": "object", "properties": {
            "token": {"type": "string", "description": "Token ticker, e.g. ETH."}},
            "required": ["token"]}}},
]

# ── Agent loop ────────────────────────────────────────────────────────────
def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a crypto portfolio assistant. Use tools to look up data. Be concise."},
        {"role": "user", "content": user_query},
    ]
    for i in range(5):
        try:
            result = client.llm.chat(
                model=og.TEE_LLM.GPT_5, messages=messages, max_tokens=600,
                temperature=0.0, tools=TOOLS, tool_choice="auto",
                x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
            )
        except Exception as e:
            return f"Error: {e}"

        if result.finish_reason == "tool_calls":
            messages.append(result.chat_output)
            for tc in result.chat_output.get("tool_calls", []):
                func = tc.get("function", tc)
                name, args = func["name"], json.loads(func.get("arguments", "{}"))
                handler = TOOL_DISPATCH.get(name)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": handler(**args) if handler else f'{{"error": "unknown tool"}}'})
            continue
        return result.chat_output.get("content", "")
    return "Max iterations reached."

# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for q in ["What's my portfolio worth?", "Which holding has the highest risk?"]:
        print(f"\nUSER: {q}")
        print(f"ASSISTANT: {run_agent(q)}")
```

## Next Steps

- **Add on-chain model tools**: See **Tutorial 1** for wrapping ONNX models as
  LangChain tools with `create_run_model_tool`, giving the agent access to on-chain
  ML predictions alongside local function calls.
- **Stream tool-calling responses**: Pass `stream=True` to get incremental tokens
  even during multi-turn tool loops. See **Tutorial 2** for streaming basics.
- **Use different providers**: Swap `og.TEE_LLM.GPT_5` for `CLAUDE_SONNET_4_6` or
  `GEMINI_2_5_FLASH` -- tool calling works across all providers.
- **Add settlement transparency**: Switch to `SETTLE_METADATA` to store the full
  tool-calling reasoning chain on-chain for audit purposes.
