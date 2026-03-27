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

# Create an LLM client
llm = og.LLM(private_key=os.environ["OG_PRIVATE_KEY"])

# One-time OPG token approval (idempotent — skips if allowance already sufficient)
llm.approve_opg(opg_amount=0.1)

# LLM Chat (TEE-verified with x402 payments, async)
result = await llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(result.chat_output["content"])
```

## Core API Reference

### Initialization

Each service has its own client class:

```python
# LLM inference (Base Sepolia OPG tokens for x402 payments)
llm = og.LLM(private_key="0x...")

# Connect directly to a known TEE IP instead of using the on-chain registry.
# WARNING: TLS certificate verification is automatically disabled when using
# llm_server_url, as self-hosted TEE servers typically use self-signed certs.
# Only connect to servers you trust over secure network paths.
llm = og.LLM(private_key="0x...", llm_server_url="https://1.2.3.4")

# On-chain model inference (OpenGradient testnet gas tokens)
alpha = og.Alpha(private_key="0x...")

# Model Hub (email/password auth, only needed for model uploads)
hub = og.ModelHub(email="...", password="...")
```

### LLM Chat

```python
result = await llm.chat(
    model: TEE_LLM,                    # og.TEE_LLM enum value
    messages: List[Dict],              # [{"role": "user", "content": "..."}]
    max_tokens: int = 100,
    temperature: float = 0.0,
    tools: List[Dict] = [],            # Optional: function calling
    tool_choice: str = None,           # "auto", "none", or specific tool
    stop_sequence: List[str] = None,
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
    stream: bool = False,              # Enable streaming responses
)
# Returns: TextGenerationOutput (or AsyncGenerator[StreamChunk] if stream=True)
#   - chat_output: Dict with role, content, tool_calls
#   - transaction_hash: str
#   - finish_reason: str ("stop", "tool_call")
#   - payment_hash: str
```

### LLM Completion

```python
result = await llm.completion(
    model: TEE_LLM,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    stop_sequence: List[str] = None,
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
)
# Returns: TextGenerationOutput
#   - completion_output: str (raw text)
#   - transaction_hash: str
```

### ONNX Model Inference

```python
result = alpha.infer(
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
result = await llm.chat(
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

result = await llm.chat(
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
stream = await llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

async for chunk in stream:
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
    model_cid=og.TEE_LLM.CLAUDE_SONNET_4_6,
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
contract = alpha.new_workflow(
    model_cid="your-model-cid",
    input_query=input_query,
    input_tensor_name="price_data",
    scheduler_params=scheduler,
)

# Manually trigger execution
result = alpha.run_workflow(contract)

# Read results
latest = alpha.read_workflow_result(contract)
history = alpha.read_workflow_history(contract, num_results=5)
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
og.x402SettlementMode.PRIVATE           # Input/output hashes only (most private)
og.x402SettlementMode.BATCH_HASHED     # Batch hashes (most cost-efficient, default)
og.x402SettlementMode.INDIVIDUAL_FULL  # Full data and metadata on-chain

# Workflow data types
og.CandleType.OPEN, .HIGH, .LOW, .CLOSE, .VOLUME
og.CandleOrder.ASCENDING, .DESCENDING
```

## Error Handling

LLM methods raise `RuntimeError` on failure and `ValueError` for invalid arguments:

```python
try:
    result = await llm.chat(...)
except RuntimeError as e:
    print(f"Inference failed: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Environment Variables

```bash
OG_PRIVATE_KEY=0x...          # Required: Ethereum private key
```

## Command-Line Interface

The SDK provides a CLI for config and inference without writing Python:

```bash
# Initialize config (interactive wizard)
opengradient config init

# View current config
opengradient config show

# Run inference (ONNX model)
opengradient infer -m <model_cid> --input '{"key": "value"}'

# Chat completion
opengradient chat --model anthropic/claude-haiku-4-5 \
  --messages '[{"role":"user","content":"Hello"}]' --max-tokens 100

# Text completion
opengradient completion --model anthropic/claude-haiku-4-5 --prompt "Hello" --max-tokens 50
```

For all commands and options: `opengradient --help`

## Resources

- [OpenGradient Documentation](https://docs.opengradient.ai)
- [GitHub Repository](https://github.com/OpenGradient/sdk)
- [Model Hub](https://hub.opengradient.ai)
