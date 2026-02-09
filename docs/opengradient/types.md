---
outline: [2,3]
---

[opengradient](./index) / types

# Package opengradient.types

OpenGradient Specific Types

## Classes

### `Abi`

#### Constructor

```python
def __init__(functions: List[`AbiFunction`])
```

#### Static methods

---

#### `from_json()`

```python
static def from_json(abi_json)
```

#### Variables

* static `functions` : List[`AbiFunction`]

### `AbiFunction`

#### Constructor

```python
def __init__(name: str, inputs: List[Union[str, ForwardRef('`AbiFunction`')]], outputs: List[Union[str, ForwardRef('`AbiFunction`')]], state_mutability: str)
```

#### Variables

* static `inputs` : List[Union[str, `AbiFunction`]]
* static `name` : str
* static `outputs` : List[Union[str, `AbiFunction`]]
* static `state_mutability` : str

### `CandleOrder`

Enum where members are also (and must be) ints

#### Variables

* static `ASCENDING`
* static `DESCENDING`

### `CandleType`

Enum where members are also (and must be) ints

#### Variables

* static `CLOSE`
* static `HIGH`
* static `LOW`
* static `OPEN`
* static `VOLUME`

### `FileUploadResult`

#### Constructor

```python
def __init__(modelCid: str, size: int)
```

#### Variables

* static `modelCid` : str
* static `size` : int

### `HistoricalInputQuery`

#### Constructor

```python
def __init__(base: str, quote: str, total_candles: int, candle_duration_in_mins: int, order: `CandleOrder`, candle_types: List[`CandleType`])
```

#### Methods

---

#### `to_abi_format()`

```python
def to_abi_format(self) ‑> tuple
```
Convert to format expected by contract ABI

#### Variables

* static `base` : str
* static `candle_duration_in_mins` : int
* static `candle_types` : List[`CandleType`]
* static `order` : `CandleOrder`
* static `quote` : str
* static `total_candles` : int

### `InferenceMode`

Enum for the different inference modes available for inference (VANILLA, ZKML, TEE)

#### Variables

* static `TEE`
* static `VANILLA`
* static `ZKML`

### `InferenceResult`

Output for ML inference requests.
This class has two fields
    transaction_hash (str): Blockchain hash for the transaction
    model_output (Dict[str, np.ndarray]): Output of the ONNX model

#### Constructor

```python
def __init__(transaction_hash: str, model_output: Dict[str, `ndarray`])
```

#### Variables

* static `model_output` : Dict[str, `ndarray`]
* static `transaction_hash` : str

### `ModelInput`

A collection of tensor inputs required for ONNX model inference.

**Attributes**

* **`numbers`**: Collection of numeric tensors for the model.
* **`strings`**: Collection of string tensors for the model.

#### Constructor

```python
def __init__(numbers: List[`NumberTensor`], strings: List[`StringTensor`])
```

#### Variables

* static `numbers` : List[`NumberTensor`]
* static `strings` : List[`StringTensor`]

### `ModelOutput`

Model output struct based on translations from smart contract.

#### Constructor

```python
def __init__(numbers: Dict[str, `ndarray`], strings: Dict[str, `ndarray`], jsons: Dict[str, `ndarray`], is_simulation_result: bool)
```

#### Variables

* static `is_simulation_result` : bool
* static `jsons` : Dict[str, `ndarray`]
* static `numbers` : Dict[str, `ndarray`]
* static `strings` : Dict[str, `ndarray`]

### `ModelRepository`

#### Constructor

```python
def __init__(name: str, initialVersion: str)
```

#### Variables

* static `initialVersion` : str
* static `name` : str

### `Number`

#### Constructor

```python
def __init__(value: int, decimals: int)
```

#### Variables

* static `decimals` : int
* static `value` : int

### `NumberTensor`

A container for numeric tensor data used as input for ONNX models.

**Attributes**

* **`name`**: Identifier for this tensor in the model.
* **`values`**: List of integer tuples representing the tensor data.

#### Constructor

```python
def __init__(name: str, values: List[Tuple[int, int]])
```

#### Variables

* static `name` : str
* static `values` : List[Tuple[int, int]]

### `PaymentNetwork`

Payment network for x402 payment protocol transactions.

Controls which blockchain network is used for paying for LLM inference.

**Attributes**

* **`OG_EVM`**: Pay on the OpenGradient EVM network (default).
* **`BASE_SEPOLIA`**: Pay on the Base Sepolia testnet using the OG token
        at address ``0x240b09731D96979f50B2C649C9CE10FcF9C7987F``.

#### Variables

* static `BASE_SEPOLIA`
* static `OG_EVM`

### `SchedulerParams`

#### Constructor

```python
def __init__(frequency: int, duration_hours: int)
```

#### Static methods

---

#### `from_dict()`

```python
static def from_dict(data: Optional[Dict[str, int]]) ‑> Optional[`SchedulerParams`]
```

#### Variables

* static `duration_hours` : int
* static `frequency` : int
* `end_time` : int

### `StreamChoice`

Represents a choice in a streaming response.

**Attributes**

* **`delta`**: The incremental changes in this chunk
* **`index`**: Choice index (usually 0)
* **`finish_reason`**: Reason for completion (appears in final chunk)

#### Constructor

```python
def __init__(delta: `StreamDelta`, index: int = 0, finish_reason: Optional[str] = None)
```

#### Variables

* static `delta` : `StreamDelta`
* static `finish_reason` : Optional[str]
* static `index` : int

### `StreamChunk`

Represents a single chunk in a streaming LLM response.

This follows the OpenAI streaming format but is provider-agnostic.
Each chunk contains incremental data, with the final chunk including
usage information.

**Attributes**

* **`choices`**: List of streaming choices (usually contains one choice)
* **`model`**: Model identifier
* **`usage`**: Token usage information (only in final chunk)
* **`is_final`**: Whether this is the final chunk (before [DONE])

#### Constructor

```python
def __init__(choices: List[`StreamChoice`], model: str, usage: Optional[`StreamUsage`] = None, is_final: bool = False)
```

#### Static methods

---

#### `from_sse_data()`

```python
static def from_sse_data(data: Dict) ‑> `StreamChunk`
```
Parse a StreamChunk from SSE data dictionary.

**Arguments**

* **`data`**: Dictionary parsed from SSE data line

**Returns**

StreamChunk instance

#### Variables

* static `choices` : List[`StreamChoice`]
* static `is_final` : bool
* static `model` : str
* static `usage` : Optional[`StreamUsage`]

### `StreamDelta`

Represents a delta (incremental change) in a streaming response.

**Attributes**

* **`content`**: Incremental text content (if any)
* **`role`**: Message role (appears in first chunk)
* **`tool_calls`**: Tool call information (if function calling is used)

#### Constructor

```python
def __init__(content: Optional[str] = None, role: Optional[str] = None, tool_calls: Optional[List[Dict]] = None)
```

#### Variables

* static `content` : Optional[str]
* static `role` : Optional[str]
* static `tool_calls` : Optional[List[Dict]]

### `StreamUsage`

Token usage information for a streaming response.

**Attributes**

* **`prompt_tokens`**: Number of tokens in the prompt
* **`completion_tokens`**: Number of tokens in the completion
* **`total_tokens`**: Total tokens used

#### Constructor

```python
def __init__(prompt_tokens: int, completion_tokens: int, total_tokens: int)
```

#### Variables

* static `completion_tokens` : int
* static `prompt_tokens` : int
* static `total_tokens` : int

### `StringTensor`

A container for string tensor data used as input for ONNX models.

**Attributes**

* **`name`**: Identifier for this tensor in the model.
* **`values`**: List of strings representing the tensor data.

#### Constructor

```python
def __init__(name: str, values: List[str])
```

#### Variables

* static `name` : str
* static `values` : List[str]

### `TEE_LLM`

Enum for LLM models available for TEE (Trusted Execution Environment) execution.

TEE mode provides cryptographic verification that inference was performed
correctly in a secure enclave. Use this for applications requiring
auditability and tamper-proof AI inference.

#### Variables

* static `CLAUDE_3_5_HAIKU`
* static `CLAUDE_3_7_SONNET`
* static `CLAUDE_4_0_SONNET`
* static `GEMINI_2_0_FLASH`
* static `GEMINI_2_5_FLASH`
* static `GEMINI_2_5_FLASH_LITE`
* static `GEMINI_2_5_PRO`
* static `GPT_4O`
* static `GPT_4_1_2025_04_14`
* static `GROK_2_1212`
* static `GROK_2_VISION_LATEST`
* static `GROK_3_BETA`
* static `GROK_3_MINI_BETA`
* static `GROK_4_1_FAST`
* static `GROK_4_1_FAST_NON_REASONING`
* static `O4_MINI`

### `TextGenerationOutput`

Output structure for text generation requests.

#### Constructor

```python
def __init__(transaction_hash: str, finish_reason: Optional[str] = None, chat_output: Optional[Dict] = None, completion_output: Optional[str] = None, payment_hash: Optional[str] = None)
```

#### Variables

* static `chat_output` : Optional[Dict] - Dictionary of chat response containing role, message content, tool call parameters, etc.. Empty dict if not applicable.
* static `completion_output` : Optional[str] - Raw text output from completion-style generation. Empty string if not applicable.
* static `finish_reason` : Optional[str] - Reason for completion (e.g., 'tool_call', 'stop', 'error'). Empty string if not applicable.
* static `payment_hash` : Optional[str] - Payment hash for x402 transaction
* static `transaction_hash` : str - Blockchain hash for the transaction.

### `TextGenerationStream`

Iterator wrapper for streaming text generation responses.

Provides a clean interface for iterating over stream chunks with
automatic parsing of SSE format.

#### Constructor

```python
def __init__(_iterator: Union[Iterator[str], AsyncIterator[str]])
```

### `x402SettlementMode`

Settlement modes for x402 payment protocol transactions.

These modes control how inference data is recorded on-chain for payment settlement
and auditability. Each mode offers different trade-offs between data completeness,
privacy, and transaction costs.

**Attributes**

* **`SETTLE`**: Individual settlement with input/output hashes only.
        Also known as SETTLE_INDIVIDUAL in some documentation.
        Records cryptographic hashes of the inference input and output.
        Most privacy-preserving option - actual data is not stored on-chain.
        Suitable for applications where only proof of execution is needed.
        CLI usage: --settlement-mode settle
* **`SETTLE_METADATA`**: Individual settlement with full metadata.
        Also known as SETTLE_INDIVIDUAL_WITH_METADATA in some documentation.
        Records complete model information, full input and output data,
        and all inference metadata on-chain.
        Provides maximum transparency and auditability.
        Higher gas costs due to larger data storage.
        CLI usage: --settlement-mode settle-metadata
* **`SETTLE_BATCH`**: Batch settlement for multiple inferences.
        Aggregates multiple inference requests into a single settlement transaction
        using batch hashes.
        Most cost-efficient for high-volume applications.
        Reduced per-inference transaction overhead.
        CLI usage: --settlement-mode settle-batch

#### Variables

* static `SETTLE`
* static `SETTLE_BATCH`
* static `SETTLE_INDIVIDUAL`
* static `SETTLE_INDIVIDUAL_WITH_METADATA`
* static `SETTLE_METADATA`