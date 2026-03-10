# OpenGradient SDK — API Reference

Detailed reference for the OpenGradient Python SDK. Use this alongside
the main SKILL.md when you need specifics about parameters, return types,
or less common features.

---

## Client Initialization

```python
import opengradient as og

client = og.init(
    private_key: str,              # Required — Base Sepolia wallet (holds OPG tokens)
    alpha_private_key: str = None, # Optional — OpenGradient testnet key
    email: str = None,             # Optional — Model Hub email
    password: str = None,          # Optional — Model Hub password
    twins_api_key: str = None,     # Optional — Digital twins API key
)
```

**Namespaces on `client`:**
- `client.llm` — LLM inference (chat, completions, streaming)
- `client.alpha` — On-chain ONNX model inference, workflows
- `client.model_hub` — Model repository CRUD
- `client.twins` — Digital twins chat (requires `twins_api_key`)

---

## LLM API — `client.llm`

### `ensure_opg_approval(opg_amount: int)`

Approve OPG token spending for x402 payments. Idempotent — safe to call
multiple times. Must be called before the first `chat()` call.

### `chat()`

```python
client.llm.chat(
    model: og.TEE_LLM,
    messages: list[dict],          # OpenAI-style message dicts
    max_tokens: int = 300,
    temperature: float = 0.0,
    stream: bool = False,
    tools: list[dict] = None,      # Function definitions
    tool_choice: str = None,       # "auto", "none", or specific
    x402_settlement_mode: og.x402SettlementMode = og.x402SettlementMode.SETTLE_BATCH,
) -> TextGenerationOutput | TextGenerationStream
```

**Messages format:**
```python
[
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Previous response"},
    {"role": "tool", "tool_call_id": "call_123", "content": "Tool output"},
]
```

---

## Return Types

### `TextGenerationOutput`

Returned by `client.llm.chat()` when `stream=False`.

| Field | Type | Description |
|-------|------|-------------|
| `chat_output` | `dict` | `{"role": "assistant", "content": "...", "tool_calls": [...]}` |
| `completion_output` | `str` | Text (completions API only) |
| `finish_reason` | `str` | `"stop"`, `"length"`, or `"tool_calls"` |
| `transaction_hash` | `str` | Blockchain tx hash (usually `"external"` for TEE) |
| `payment_hash` | `str` | x402 settlement proof |

### `TextGenerationStream`

Returned by `client.llm.chat()` when `stream=True`. Iterable of `StreamChunk`.

### `StreamChunk`

| Field | Type | Description |
|-------|------|-------------|
| `choices` | `list[StreamChoice]` | Delta updates |
| `choices[0].delta.content` | `str or None` | Incremental text |
| `choices[0].delta.tool_calls` | `list or None` | Incremental tool calls |
| `model` | `str` | Model identifier |
| `usage` | `StreamUsage or None` | Token counts (final chunk only) |
| `is_final` | `bool` | `True` on last chunk |

### `InferenceResult`

Returned by `client.alpha.infer()`.

| Field | Type | Description |
|-------|------|-------------|
| `model_output` | `dict[str, np.ndarray]` | Model outputs |
| `transaction_hash` | `str` | On-chain tx hash |

---

## Models — `og.TEE_LLM`

### OpenAI
- `GPT_4_1_2025_04_14`
- `O4_MINI`
- `GPT_5`
- `GPT_5_MINI`
- `GPT_5_2`

### Anthropic
- `CLAUDE_SONNET_4_5`
- `CLAUDE_SONNET_4_6`
- `CLAUDE_HAIKU_4_5`
- `CLAUDE_OPUS_4_5`
- `CLAUDE_OPUS_4_6`

### Google
- `GEMINI_2_5_FLASH`
- `GEMINI_2_5_PRO`
- `GEMINI_2_5_FLASH_LITE`
- `GEMINI_3_PRO`
- `GEMINI_3_FLASH`

### xAI (Grok)
- `GROK_4`
- `GROK_4_FAST`
- `GROK_4_1_FAST`
- `GROK_4_1_FAST_NON_REASONING`

---

## Settlement Modes — `og.x402SettlementMode`

| Mode | Value | Description |
|------|-------|-------------|
| `SETTLE` | `"private"` | Hashes only — maximum privacy |
| `SETTLE_METADATA` | `"individual"` | Full data on-chain — maximum transparency |
| `SETTLE_BATCH` | `"batch"` | Aggregated hashes — most cost-efficient (default) |

---

## Inference Modes — `og.InferenceMode` (Alpha)

| Mode | Description |
|------|-------------|
| `VANILLA` | Standard execution |
| `TEE` | Trusted Execution Environment |
| `ZKML` | Zero-Knowledge ML |

---

## Alpha API — `client.alpha`

### `infer()`

```python
client.alpha.infer(
    model_cid: str,
    inference_mode: og.InferenceMode,
    model_input: dict,
) -> InferenceResult
```

### `new_workflow()`

```python
client.alpha.new_workflow(
    model_cid: str,
    input_query: og.HistoricalInputQuery,
    input_tensor_name: str,
    scheduler_params: og.SchedulerParams,
) -> str  # contract address
```

### `run_workflow(contract_address: str) -> InferenceResult`

### `read_workflow_result(contract_address: str) -> InferenceResult`

### `read_workflow_history(contract_address: str, num_results: int) -> list`

---

## Workflow Input Queries

### `og.HistoricalInputQuery`

```python
og.HistoricalInputQuery(
    base="ETH",
    quote="USD",
    total_candles=10,
    candle_duration_in_mins=60,
    order=og.CandleOrder.DESCENDING,
    candle_types=[og.CandleType.CLOSE],
)
```

### `og.SchedulerParams`

```python
og.SchedulerParams(
    frequency=3600,       # Seconds between runs
    duration_hours=24,    # Total duration
)
```

---

## Model Hub — `client.model_hub`

### `create_model(model_name, model_desc, version) -> ModelRepository`

### `upload(model_name, version, model_path) -> UploadResult`

`UploadResult.modelCid` — the content-addressed identifier for the uploaded model.

---

## Digital Twins — `client.twins`

### `chat()`

```python
client.twins.chat(
    twin_id: str,
    model: og.TEE_LLM,
    messages: list[dict],
    max_tokens: int,
) -> TextGenerationOutput
```

---

## LangChain Integration

```python
llm = og.agents.langchain_adapter(
    private_key: str,
    model_cid: og.TEE_LLM,
    max_tokens: int = 300,
    x402_settlement_mode: og.x402SettlementMode = og.x402SettlementMode.SETTLE_BATCH,
)
```

Returns a LangChain-compatible `BaseChatModel` that can be used with
`create_react_agent()`, chains, or any LangChain component expecting an LLM.

---

## Tools Format (for Tool Calling)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "What this function does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "..."},
                    "param2": {"type": "number"},
                },
                "required": ["param1"],
            },
        },
    }
]
```

---

## CLI Commands

```bash
opengradient config init       # Interactive setup
opengradient config show       # Display config
opengradient config clear      # Reset config
opengradient create-account    # Generate wallet
opengradient infer -m <CID> --input '<json>'
opengradient chat --model <model> --messages '<json>' --max-tokens 100
```
