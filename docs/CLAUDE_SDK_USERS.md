# OpenGradient SDK - Claude Code Guide

> **For SDK Users**: Copy this file to your project's `CLAUDE.md` to help Claude Code assist you with OpenGradient SDK development.

## Overview

OpenGradient SDK provides decentralized AI inference with cryptographic verification via Trusted Execution Environments (TEE). It supports LLM chat/completions, ONNX model inference, and scheduled workflows. All LLM calls are TEE-verified with x402 payments.

## Installation

```bash
pip install opengradient
```

## Quick Start

```python
import opengradient as og
import os

# Initialize client
client = og.Client(
    private_key=os.environ["OG_PRIVATE_KEY"],  # Required: Ethereum private key
)

# LLM Chat (TEE-verified with x402 payments)
result = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(result.chat_output["content"])
```

## Core API Reference

### Client Initialization

```python
# Option 1: Create client instance (recommended)
client = og.Client(
    private_key="0x...",           # Required: Ethereum private key
    email=None,                    # Optional: Model Hub auth
    password=None,                 # Optional: Model Hub auth
)

# Option 2: Global initialization
og.init(private_key="0x...", email="...", password="...")
```

### LLM Chat

```python
result = client.llm.chat(
    model: TEE_LLM,                    # og.TEE_LLM enum value
    messages: List[Dict],              # [{"role": "user", "content": "..."}]
    max_tokens: int = 100,
    temperature: float = 0.0,
    tools: List[Dict] = [],            # Optional: function calling
    tool_choice: str = None,           # "auto", "none", or specific tool
    stop_sequence: List[str] = None,
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.SETTLE_BATCH,
    stream: bool = False,              # Enable streaming responses
)
# Returns: TextGenerationOutput (or TextGenerationStream if stream=True)
#   - chat_output: Dict with role, content, tool_calls
#   - transaction_hash: str
#   - finish_reason: str ("stop", "tool_call")
#   - payment_hash: str
```

### LLM Completion

```python
result = client.llm.completion(
    model: TEE_LLM,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    stop_sequence: List[str] = None,
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.SETTLE_BATCH,
)
# Returns: TextGenerationOutput
#   - completion_output: str (raw text)
#   - transaction_hash: str
```

### ONNX Model Inference

```python
result = client.alpha.infer(
    model_cid: str,                    # IPFS CID of model
    inference_mode: og.InferenceMode,  # VANILLA, TEE, or ZKML
    model_input: Dict[str, Any],       # Input tensors
    max_retries: int = None,
)
# Returns: InferenceResult
#   - model_output: Dict[str, np.ndarray]
#   - transaction_hash: str
```

## Available Models

### TEE-Verified Models (og.TEE_LLM)

All LLM models are TEE-verified. `og.LLM` and `og.TEE_LLM` contain the same models:

```python
# OpenAI
og.TEE_LLM.GPT_4_1_2025_04_14
og.TEE_LLM.O4_MINI
og.TEE_LLM.GPT_5
og.TEE_LLM.GPT_5_MINI
og.TEE_LLM.GPT_5_2

# Anthropic
og.TEE_LLM.CLAUDE_SONNET_4_5
og.TEE_LLM.CLAUDE_SONNET_4_6
og.TEE_LLM.CLAUDE_HAIKU_4_5
og.TEE_LLM.CLAUDE_OPUS_4_5
og.TEE_LLM.CLAUDE_OPUS_4_6

# Google
og.TEE_LLM.GEMINI_2_5_FLASH
og.TEE_LLM.GEMINI_2_5_PRO
og.TEE_LLM.GEMINI_2_5_FLASH_LITE
og.TEE_LLM.GEMINI_3_PRO
og.TEE_LLM.GEMINI_3_FLASH

# xAI
og.TEE_LLM.GROK_4
og.TEE_LLM.GROK_4_FAST
og.TEE_LLM.GROK_4_1_FAST
og.TEE_LLM.GROK_4_1_FAST_NON_REASONING
```

### Using Models

All models are accessed through the OpenGradient TEE infrastructure with x402 payments:

```python
result = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Common Patterns

### Tool/Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

result = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
    tool_choice="auto",
)

if result.chat_output.get("tool_calls"):
    for call in result.chat_output["tool_calls"]:
        print(f"Call {call['name']} with {call['arguments']}")
```

### Streaming

```python
stream = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    for choice in chunk.choices:
        if choice.delta.content:
            print(choice.delta.content, end="")
```

### LangChain Integration

```python
from langgraph.prebuilt import create_react_agent

# Create LangChain-compatible LLM
llm = og.agents.langchain_adapter(
    private_key=os.environ["OG_PRIVATE_KEY"],
    model_cid=og.LLM.CLAUDE_SONNET_4_6,
    max_tokens=300,
)

# Use with LangChain agents
agent = create_react_agent(model=llm, tools=[...])
for chunk in agent.stream({"messages": [("user", "Your question")]}):
    print(chunk)
```

### Scheduled Workflows (Alpha Testnet)

```python
# Define data input
input_query = og.HistoricalInputQuery(
    base="ETH",
    quote="USD",
    total_candles=10,
    candle_duration_in_mins=30,
    order=og.CandleOrder.ASCENDING,
    candle_types=[og.CandleType.OPEN, og.CandleType.HIGH,
                  og.CandleType.LOW, og.CandleType.CLOSE],
)

# Schedule execution
scheduler = og.SchedulerParams(frequency=60, duration_hours=2)

# Deploy
contract = client.alpha.new_workflow(
    model_cid="your-model-cid",
    input_query=input_query,
    input_tensor_name="price_data",
    scheduler_params=scheduler,
)

# Manually trigger execution
result = client.alpha.run_workflow(contract)

# Read results
latest = client.alpha.read_workflow_result(contract)
history = client.alpha.read_workflow_history(contract, num_results=5)
```

### AlphaSense Tool Creation

```python
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    token: str = Field(description="Token symbol")

tool = og.alphasense.create_run_model_tool(
    tool_type=og.alphasense.ToolType.LANGCHAIN,
    model_cid="your-model-cid",
    tool_name="PricePrediction",
    tool_description="Predicts token price movement",
    tool_input_schema=ToolInput,
    model_input_provider=lambda token: {"input": get_price_data(token)},
    model_output_formatter=lambda r: f"Prediction: {r.model_output['pred']}",
    inference_mode=og.InferenceMode.VANILLA,
)
```

## Key Types

```python
# On-chain inference modes (for ONNX models)
og.InferenceMode.VANILLA    # Standard execution
og.InferenceMode.TEE        # Trusted Execution Environment
og.InferenceMode.ZKML       # Zero-knowledge proof

# x402 Payment Settlement Modes (for LLM calls)
og.x402SettlementMode.SETTLE           # Input/output hashes only (most private)
og.x402SettlementMode.SETTLE_BATCH     # Batch hashes (most cost-efficient, default)
og.x402SettlementMode.SETTLE_METADATA  # Full data and metadata on-chain

# Workflow data types
og.CandleType.OPEN, .HIGH, .LOW, .CLOSE, .VOLUME
og.CandleOrder.ASCENDING, .DESCENDING
```

## Error Handling

```python
from opengradient.client.exceptions import (
    OpenGradientError,      # Base exception
    AuthenticationError,
    InferenceError,
    InvalidInputError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    ServerError,
    UnsupportedModelError,
    InsufficientCreditsError,
)

try:
    result = client.llm.chat(...)
except RateLimitError:
    # Retry with backoff
except InferenceError as e:
    print(f"Inference failed: {e.message}")
except OpenGradientError as e:
    print(f"Error {e.status_code}: {e.message}")
```

## Environment Variables

```bash
OG_PRIVATE_KEY=0x...          # Required: Ethereum private key
```

## Resources

- [OpenGradient Documentation](https://docs.opengradient.ai)
- [GitHub Repository](https://github.com/OpenGradient/sdk)
- [Model Hub](https://hub.opengradient.ai)
