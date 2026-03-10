---
name: opengradient
description: >
  Use when the user wants to write code using the OpenGradient SDK, including
  LLM inference, chat completions, streaming, tool calling, on-chain model
  inference, LangChain agents, model hub operations, or digital twins.
  Also use when the user asks how to use a specific OpenGradient feature.
argument-hint: "[task description]"
allowed-tools: Read, Grep, Glob
---

You are an expert on the **OpenGradient Python SDK** (`opengradient`). Help the user write correct, idiomatic code using the SDK.

When the user describes what they want to build, generate working code that follows the patterns below. Always prefer the simplest approach that satisfies the requirements.

## Key Reference Files

When you need more detail, read these files from the project:

- **Examples**: `examples/` folder (runnable scripts for every feature)
- **Tutorials**: `tutorials/` folder (step-by-step walkthroughs)
- **Types & Enums**: `src/opengradient/types.py`
- **Client API**: `src/opengradient/client/client.py`
- **LLM API**: `src/opengradient/client/llm.py`
- **Alpha API**: `src/opengradient/client/alpha.py`
- **LangChain adapter**: `src/opengradient/agents/__init__.py`

Also read the detailed API reference bundled with this skill at `api-reference.md` in this skill's directory.

## SDK Overview

OpenGradient is a decentralized AI inference platform. The SDK provides:

- **Verified LLM inference** via TEE (Trusted Execution Environment)
- **x402 payment settlement** on Base Sepolia (on-chain receipts)
- **Multi-provider models** (OpenAI, Anthropic, Google, xAI) through a unified API
- **On-chain ONNX model inference** (alpha features)
- **LangChain integration** for building agents
- **Digital twins** chat

## Initialization

```python
import opengradient as og

client = og.init(
    private_key="0x...",          # Required: Base Sepolia key with OPG tokens
    alpha_private_key="0x...",    # Optional: OpenGradient testnet key
    email="...",                  # Optional: Model Hub auth
    password="...",               # Optional: Model Hub auth
    twins_api_key="...",          # Optional: Digital twins
)
```

Before the first LLM call, approve OPG token spending (idempotent):
```python
client.llm.ensure_opg_approval(opg_amount=5)
```

## Available Models (`og.TEE_LLM`)

| Provider   | Models |
|------------|--------|
| OpenAI     | `GPT_4_1_2025_04_14`, `O4_MINI`, `GPT_5`, `GPT_5_MINI`, `GPT_5_2` |
| Anthropic  | `CLAUDE_SONNET_4_5`, `CLAUDE_SONNET_4_6`, `CLAUDE_HAIKU_4_5`, `CLAUDE_OPUS_4_5`, `CLAUDE_OPUS_4_6` |
| Google     | `GEMINI_2_5_FLASH`, `GEMINI_2_5_PRO`, `GEMINI_2_5_FLASH_LITE`, `GEMINI_3_PRO`, `GEMINI_3_FLASH` |
| xAI        | `GROK_4`, `GROK_4_FAST`, `GROK_4_1_FAST`, `GROK_4_1_FAST_NON_REASONING` |

## Settlement Modes (`og.x402SettlementMode`)

- `SETTLE` — Hashes only (maximum privacy)
- `SETTLE_METADATA` — Full data on-chain (maximum transparency)
- `SETTLE_BATCH` — Aggregated hashes (most cost-efficient, default)

## Core Patterns

### Basic Chat

```python
result = client.llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=300,
    temperature=0.0,
)
print(result.chat_output["content"])
```

### Streaming

```python
stream = client.llm.chat(
    model=og.TEE_LLM.GPT_4_1_2025_04_14,
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

result = client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools,
    max_tokens=200,
)

if result.finish_reason == "tool_calls":
    for tc in result.chat_output["tool_calls"]:
        print(f"Call: {tc['function']['name']}({tc['function']['arguments']})")
```

### Multi-Turn Tool Agent Loop

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_query},
]

for _ in range(max_iterations):
    result = client.llm.chat(
        model=og.TEE_LLM.GPT_5,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    if result.finish_reason == "tool_calls":
        messages.append(result.chat_output)
        for tc in result.chat_output["tool_calls"]:
            tool_result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })
    else:
        final_answer = result.chat_output["content"]
        break
```

### LangChain ReAct Agent

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

llm = og.agents.langchain_adapter(
    private_key="0x...",
    model_cid=og.TEE_LLM.GPT_4_1_2025_04_14,
    max_tokens=300,
)

@tool
def lookup(query: str) -> str:
    """Look up information."""
    return "result"

agent = create_react_agent(llm, [lookup])
result = agent.invoke({"messages": [("user", "Find info about X")]})
print(result["messages"][-1].content)
```

### On-Chain ONNX Inference (Alpha)

```python
result = client.alpha.infer(
    model_cid="QmbUqS93oc4JTLMHwpVxsE39mhNxy6hpf6Py3r9oANr8aZ",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)
print(result.model_output)
print(result.transaction_hash)
```

### Digital Twins

```python
client = og.init(private_key="0x...", twins_api_key="your-key")

result = client.twins.chat(
    twin_id="0x1abd463fd6244be4a1dc0f69e0b70cd5",
    model=og.TEE_LLM.GROK_4_1_FAST_NON_REASONING,
    messages=[{"role": "user", "content": "What do you think about AI?"}],
    max_tokens=1000,
)
print(result.chat_output["content"])
```

### Model Hub: Upload a Model

```python
repo = client.model_hub.create_model(
    model_name="my-model",
    model_desc="A prediction model",
    version="1.0.0",
)
upload = client.model_hub.upload(
    model_name=repo.name,
    version=repo.initialVersion,
    model_path="./model.onnx",
)
print(f"Model CID: {upload.modelCid}")
```

## Return Types

- **`TextGenerationOutput`**: `chat_output` (dict), `finish_reason`, `transaction_hash`, `payment_hash`
- **`TextGenerationStream`**: iterable of `StreamChunk` objects
- **`StreamChunk`**: `choices[0].delta.content`, `choices[0].delta.tool_calls`, `usage` (final only), `is_final`
- **`InferenceResult`**: `model_output` (dict of np.ndarray), `transaction_hash`

## Guidelines

1. Always call `client.llm.ensure_opg_approval()` before the first LLM inference.
2. Handle `finish_reason`: `"stop"` / `"length"` = text response, `"tool_calls"` = function calls.
3. For streaming, check `chunk.choices[0].delta.content` is not None before printing.
4. In tool-calling loops, append `result.chat_output` as the assistant message, then append each tool result with `role: "tool"` and matching `tool_call_id`.
5. Use environment variables or config files for private keys — never hardcode them.
6. If you are unsure about a specific API detail, read the source files listed above.
