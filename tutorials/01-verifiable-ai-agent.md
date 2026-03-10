# Build a Verifiable AI Agent with On-Chain Tools

Traditional AI agents operate as black boxes -- you send a prompt, get a response, and
have no way to prove what model ran, what it saw, or what it actually produced. For
financial applications, compliance workflows, or any context where trust matters, this
opacity is a serious problem.

OpenGradient solves this by running every LLM call inside a **Trusted Execution
Environment (TEE)** and settling every inference on-chain via the **x402 payment
protocol**. This means you get cryptographic proof that a specific model processed
your exact input and produced the exact output you received -- no one, not even the
infrastructure operator, can tamper with it.

In this tutorial you will build a LangChain ReAct agent that combines TEE-verified
LLM reasoning with on-chain ONNX model inference. The agent can look up a crypto
portfolio *and* call a volatility model that executes directly on the OpenGradient
blockchain, giving you a fully verifiable AI financial advisor.

## Prerequisites

```bash
pip install opengradient langgraph
```

You also need an OpenGradient private key funded with test tokens. Any standard
Ethereum private key works -- you can generate one with any Ethereum wallet.

```bash
export OG_PRIVATE_KEY="0x..."
```

> **Faucet:** Get free OPG tokens on Base Sepolia at <https://faucet.opengradient.ai/>
> so your wallet can pay for inference transactions. All x402 LLM payments currently
> settle on Base Sepolia.

## Step 1: Initialize the Client and LangChain Adapter

Before making any LLM calls, you need to approve OPG token spending for the x402
payment protocol. The `ensure_opg_approval` method checks your wallet's current
Permit2 allowance and only sends an on-chain transaction if the allowance is below
the requested amount -- so it's safe to call every time.

```python
import os
import opengradient as og

private_key = os.environ["OG_PRIVATE_KEY"]

# Approve OPG spending for x402 payments (idempotent -- skips if already approved).
client = og.Client(private_key=private_key)
client.llm.ensure_opg_approval(opg_amount=5)

# Create the LangChain chat model backed by OpenGradient TEE.
# Note: the adapter creates its own internal Client, separate from any client
# created via og.init(). This is why private_key is passed here explicitly.
# The approval above applies to the wallet, so it covers the adapter's client too.
llm = og.agents.langchain_adapter(
    private_key=private_key,
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    max_tokens=500,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)
```

Under the hood this creates an `OpenGradientChatModel` that implements LangChain's
`BaseChatModel` interface. It handles message format conversion, tool call parsing,
and x402 payment signing automatically.

## Step 2: Create a Standard Tool

Let's give the agent a simple tool to look up portfolio holdings. This is a regular
LangChain `@tool` -- nothing OpenGradient-specific yet.

```python
import json
from langchain_core.tools import tool

PORTFOLIO = {
    "ETH": {"amount": 10.0, "avg_cost_usd": 1950.00},
    "BTC": {"amount": 0.5, "avg_cost_usd": 42000.00},
}

@tool
def get_portfolio_holdings() -> str:
    """Returns the user's current crypto portfolio holdings including token, amount, and average cost."""
    return json.dumps(PORTFOLIO, indent=2)
```

## Step 3: Create an AlphaSense On-Chain Model Tool

This is where things get interesting. `create_run_model_tool` wraps an ONNX model
that lives on the OpenGradient blockchain as a LangChain `StructuredTool`. When the
agent calls this tool, it triggers an actual on-chain transaction -- the inference
runs on a blockchain node and the result is recorded in a transaction.

You need three pieces:

1. **A Pydantic input schema** -- defines what the LLM agent passes to the tool
2. **A model input provider** -- converts the agent's tool call into model tensors
3. **A model output formatter** -- turns the raw `InferenceResult` into a string

```python
from enum import Enum
from pydantic import BaseModel, Field
from opengradient.alphasense import create_run_model_tool, ToolType

# The model CID for a public ETH volatility model on the Alpha Testnet.
VOLATILITY_MODEL_CID = "hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M"

class Token(str, Enum):
    ETH = "ethereum"
    BTC = "bitcoin"

class VolatilityInput(BaseModel):
    token: Token = Field(
        default=Token.ETH,
        description="The cryptocurrency to measure volatility for.",
    )

# Sample price data. In production, fetch from an exchange API or oracle.
SAMPLE_PRICES = {
    Token.ETH: [2010.1, 2012.3, 2020.1, 2019.2, 2025.0, 2018.7, 2030.5, 2028.1],
    Token.BTC: [67100.0, 67250.0, 67180.0, 67320.0, 67150.0, 67400.0, 67280.0, 67350.0],
}

def provide_model_input(**llm_input) -> dict:
    """Convert the agent's tool call into model input tensors."""
    token = llm_input.get("token", Token.ETH)
    return {"price_series": SAMPLE_PRICES.get(token, SAMPLE_PRICES[Token.ETH])}

def format_model_output(inference_result: og.InferenceResult) -> str:
    """Format the on-chain model's output into a readable string."""
    std = float(inference_result.model_output["std"].item())
    return (
        f"Volatility (std dev of returns): {std:.4f} ({std:.2%} annualized). "
        f"On-chain tx: {inference_result.transaction_hash}"
    )
```

Now create the tool. You need an initialized client to pass `client.alpha` as the
inference backend:

```python
client = og.init(private_key=os.environ["OG_PRIVATE_KEY"])

volatility_tool = create_run_model_tool(
    tool_type=ToolType.LANGCHAIN,
    model_cid=VOLATILITY_MODEL_CID,
    tool_name="crypto_volatility",
    tool_description=(
        "Measures the return volatility (standard deviation of returns) for a "
        "cryptocurrency using an on-chain ONNX model. Use this when the user "
        "asks about risk, volatility, or position sizing."
    ),
    model_input_provider=provide_model_input,
    model_output_formatter=format_model_output,
    inference=client.alpha,
    tool_input_schema=VolatilityInput,
    inference_mode=og.InferenceMode.VANILLA,
)
```

When the agent invokes `crypto_volatility`, the SDK:
1. Calls `provide_model_input()` with the LLM's chosen arguments
2. Submits an on-chain transaction to run the ONNX model
3. Waits for the transaction receipt and parses the model output
4. Calls `format_model_output()` and returns the string to the agent

## Step 4: Wire Up the ReAct Agent

With both tools ready, combine them into a ReAct agent using `langgraph`:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[get_portfolio_holdings, volatility_tool],
)

# Ask a question that requires both tools.
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "I hold 10 ETH. Based on the current volatility from the "
                "on-chain model, should I increase my position? What's the risk?"
            ),
        }
    ],
})

# Print the agent's final answer.
final_message = result["messages"][-1]
print(final_message.content)
```

The agent will:
1. Call `get_portfolio_holdings` to see what you own
2. Call `crypto_volatility` with `token="ethereum"` to get on-chain volatility
3. Reason about the numbers and give you a recommendation

## Step 5: Examine the Output

Every step in the agent's execution leaves a verifiable trail:

- **LLM calls**: Each reasoning step ran in a TEE. The x402 payment hash in the
  response header is your cryptographic receipt. You can look it up on-chain to
  verify the model, input, and output.

- **On-chain model inference**: The volatility tool call produced a blockchain
  transaction hash. You can inspect this on the OpenGradient block explorer to see
  the exact model CID, input tensors, and output tensors that were recorded.

The `format_model_output` function above prints the transaction hash. In a production
app you would store these hashes for audit purposes.

## Understanding Settlement Modes

When calling the LLM, the `x402_settlement_mode` parameter controls how much data
is recorded on-chain:

| Mode | What's Stored | Best For |
|------|--------------|----------|
| `SETTLE` | Hashes of input and output only | **Privacy** -- proves execution happened without revealing content |
| `SETTLE_BATCH` | Batch hash of multiple inferences | **Cost efficiency** -- reduces per-call gas costs (default) |
| `SETTLE_METADATA` | Full model info, input, output, and metadata | **Transparency** -- complete auditability for compliance |

Choose based on your requirements:

```python
# For development and testing -- cheapest option
llm_dev = og.agents.langchain_adapter(
    private_key=os.environ["OG_PRIVATE_KEY"],
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)

# For production financial applications -- full audit trail
llm_prod = og.agents.langchain_adapter(
    private_key=os.environ["OG_PRIVATE_KEY"],
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_METADATA,
)

# For privacy-sensitive applications -- minimal on-chain footprint
llm_private = og.agents.langchain_adapter(
    private_key=os.environ["OG_PRIVATE_KEY"],
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    x402_settlement_mode=og.x402SettlementMode.SETTLE,
)
```

## Complete Code

```python
"""Verifiable AI Financial Agent -- complete working example."""

import json
import os
from enum import Enum

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

import opengradient as og
from opengradient.alphasense import ToolType, create_run_model_tool

# ── Config ────────────────────────────────────────────────────────────────
VOLATILITY_MODEL_CID = "hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M"

PORTFOLIO = {
    "ETH": {"amount": 10.0, "avg_cost_usd": 1950.00},
    "BTC": {"amount": 0.5, "avg_cost_usd": 42000.00},
}

# ── On-chain model tool (Token must be defined before SAMPLE_PRICES) ─────
class Token(str, Enum):
    ETH = "ethereum"
    BTC = "bitcoin"

SAMPLE_PRICES = {
    Token.ETH: [2010.1, 2012.3, 2020.1, 2019.2, 2025.0, 2018.7, 2030.5, 2028.1],
    Token.BTC: [67100.0, 67250.0, 67180.0, 67320.0, 67150.0, 67400.0, 67280.0, 67350.0],
}

# ── Client ────────────────────────────────────────────────────────────────
# og.init() creates the global client (used by AlphaSense tools).
# langchain_adapter() creates its own internal client for LLM calls.
private_key = os.environ["OG_PRIVATE_KEY"]
client = og.init(private_key=private_key)

# Approve OPG spending for x402 payments (idempotent -- skips if already approved).
client.llm.ensure_opg_approval(opg_amount=5)

llm = og.agents.langchain_adapter(
    private_key=private_key,
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    max_tokens=500,
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)

# ── Standard tool ─────────────────────────────────────────────────────────
@tool
def get_portfolio_holdings() -> str:
    """Returns the user's current crypto portfolio holdings."""
    return json.dumps(PORTFOLIO, indent=2)

class VolatilityInput(BaseModel):
    token: Token = Field(default=Token.ETH, description="Cryptocurrency to check.")

def provide_model_input(**llm_input) -> dict:
    token = llm_input.get("token", Token.ETH)
    return {"price_series": SAMPLE_PRICES.get(token, SAMPLE_PRICES[Token.ETH])}

def format_model_output(result: og.InferenceResult) -> str:
    std = float(result.model_output["std"].item())
    return f"Volatility: {std:.4f} ({std:.2%}). Tx: {result.transaction_hash}"

volatility_tool = create_run_model_tool(
    tool_type=ToolType.LANGCHAIN,
    model_cid=VOLATILITY_MODEL_CID,
    tool_name="crypto_volatility",
    tool_description="Measures return volatility for a crypto token using an on-chain model.",
    model_input_provider=provide_model_input,
    model_output_formatter=format_model_output,
    inference=client.alpha,
    tool_input_schema=VolatilityInput,
    inference_mode=og.InferenceMode.VANILLA,
)

# ── Agent ─────────────────────────────────────────────────────────────────
agent = create_react_agent(model=llm, tools=[get_portfolio_holdings, volatility_tool])

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "I hold 10 ETH. Based on the current volatility, "
                "should I increase my position? What's the risk?"
            ),
        }],
    })
    print(result["messages"][-1].content)
```

## Next Steps

- **Swap models**: Replace `GPT_4_1_2025_04_14` with `CLAUDE_SONNET_4_6` or
  `GEMINI_2_5_PRO` -- the rest of your code stays the same.
- **Add more on-chain tools**: Use `create_run_model_tool` with different model CIDs
  to give your agent access to price prediction, sentiment analysis, or other ML
  models deployed on OpenGradient.
- **Read workflow results**: Use `og.alphasense.create_read_workflow_tool` to read
  from scheduled on-chain workflows that run models automatically.
- **Go to production**: Switch settlement mode to `SETTLE_METADATA` and store the
  payment hashes and transaction hashes for your compliance records.
