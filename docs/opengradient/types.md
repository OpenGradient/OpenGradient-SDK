---
outline: [2,4]
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
def __init__(
    name: str,
    inputs: List[Union[str, ForwardRef('`AbiFunction`')]],
    outputs: List[Union[str, ForwardRef('`AbiFunction`')]],
    state_mutability: str
)
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
def __init__(
    base: str,
    quote: str,
    total_candles: int,
    candle_duration_in_mins: int,
    order: `CandleOrder`,
    candle_types: List[`CandleType`]
)
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

### `ModelOutput`

Model output struct based on translations from smart contract.

#### Constructor

```python
def __init__(
    numbers: Dict[str, `ndarray`],
    strings: Dict[str, `ndarray`],
    jsons: Dict[str, `ndarray`],
    is_simulation_result: bool
)
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

### `ResponseFormat`

Controls the output format enforced by the TEE gateway.

Use ``type="json_object"`` to receive any valid JSON object (supported by
OpenAI, Gemini, and Grok). Use ``type="json_schema"`` with a ``json_schema``
definition to enforce a specific schema (supported by all providers,
including Anthropic).

**Attributes**

* **`type`**: One of ``"text"``, ``"json_object"``, or ``"json_schema"``.
* **`json_schema`**: Schema definition (required when ``type="json_schema"``).
        Must contain ``name`` (str) and ``schema`` (dict).
        ``strict`` (bool) is optional.

**Raises**

* **`ValueError`**: If ``type`` is not a recognised value, or if
        ``type="json_schema"`` is used without providing ``json_schema``.

Examples::

    # Any valid JSON object — OpenAI, Gemini, Grok only
    ResponseFormat(type="json_object")

    # Strict schema — all providers including Anthropic
    ResponseFormat(
        type="json_schema",
        json_schema={
            "name": "person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    )

#### Constructor

```python
def __init__(type: str, json_schema: Optional[Dict] = None)
```

#### Methods

---

#### `to_dict()`

```python
def to_dict(self) ‑> Dict
```
Serialise to a JSON-compatible dict for the TEE gateway request payload.

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
* **`tee_signature`**: RSA-PSS signature over the response, present on the final chunk.
        Forwarded as-is; verified at on-chain settlement, not by the SDK. Live
        trust comes from the registry-pinned TLS channel — see
        ``TextGenerationOutput`` for the trust model.
* **`tee_timestamp`**: ISO timestamp from the TEE at signing time, present on the final chunk
* **`tee_id`**: On-chain TEE registry ID of the enclave that served this request (final chunk only)
* **`tee_endpoint`**: Endpoint URL of the TEE that served this request (final chunk only)
* **`tee_payment_address`**: Payment address registered for the TEE (final chunk only)
* **`data_settlement_transaction_hash`**: Transaction hash for the data settlement
        transaction, present on the final chunk when available.
* **`data_settlement_blob_id`**: Walrus blob ID for individual data settlement,
        present on the final chunk when available.
* **`images`**: Generated images returned by image-output models, present on the
        final chunk when available. Each entry is a ``data:`` URI.

#### Constructor

```python
def __init__(
    choices: List[`StreamChoice`],
    model: str,
    usage: Optional[`StreamUsage`] = None,
    is_final: bool = False,
    tee_signature: Optional[str] = None,
    tee_timestamp: Optional[str] = None,
    tee_id: Optional[str] = None,
    tee_endpoint: Optional[str] = None,
    tee_payment_address: Optional[str] = None,
    data_settlement_transaction_hash: Optional[str] = None,
    data_settlement_blob_id: Optional[str] = None,
    images: Optional[List[str]] = None
)
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
**`StreamChunk` fields:**

* **`choices`**: List of streaming choices (usually contains one choice)
* **`model`**: Model identifier
* **`usage`**: Token usage information (only in final chunk)
* **`is_final`**: Whether this is the final chunk (before [DONE])
* **`tee_signature`**: RSA-PSS signature over the response, present on the final chunk.
        Forwarded as-is; verified at on-chain settlement, not by the SDK. Live
        trust comes from the registry-pinned TLS channel — see
        ``TextGenerationOutput`` for the trust model.
* **`tee_timestamp`**: ISO timestamp from the TEE at signing time, present on the final chunk
* **`tee_id`**: On-chain TEE registry ID of the enclave that served this request (final chunk only)
* **`tee_endpoint`**: Endpoint URL of the TEE that served this request (final chunk only)
* **`tee_payment_address`**: Payment address registered for the TEE (final chunk only)
* **`data_settlement_transaction_hash`**: Transaction hash for the data settlement
        transaction, present on the final chunk when available.
* **`data_settlement_blob_id`**: Walrus blob ID for individual data settlement,
        present on the final chunk when available.
* **`images`**: Generated images returned by image-output models, present on the
        final chunk when available. Each entry is a ``data:`` URI.

### `StreamDelta`

Represents a delta (incremental change) in a streaming response.

**Attributes**

* **`content`**: Incremental text content (if any)
* **`role`**: Message role (appears in first chunk)
* **`tool_calls`**: Tool call information (if function calling is used)

#### Constructor

```python
def __init__(
    content: Optional[str] = None,
    role: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None
)
```

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

### `StringTensor`

A container for string tensor data used as input for ONNX models.

**Attributes**

* **`name`**: Identifier for this tensor in the model.
* **`values`**: List of strings representing the tensor data.

#### Constructor

```python
def __init__(name: str, values: List[str])
```

### `TEE_LLM`

Enum for LLM models available for TEE (Trusted Execution Environment) execution.

TEE mode provides cryptographic verification that inference was performed
correctly in a secure enclave. Use this for applications requiring
auditability and tamper-proof AI inference.

#### Variables

* static `CLAUDE_HAIKU_4_5`
* static `CLAUDE_OPUS_4_5`
* static `CLAUDE_OPUS_4_6`
* static `CLAUDE_OPUS_4_7`
* static `CLAUDE_OPUS_4_8`
* static `CLAUDE_SONNET_4_5`
* static `CLAUDE_SONNET_4_6`
* static `DEEPSEEK_V4_FLASH`
* static `DEEPSEEK_V4_PRO`
* static `GEMINI_2_5_FLASH`
* static `GEMINI_2_5_FLASH_IMAGE`
* static `GEMINI_2_5_FLASH_LITE`
* static `GEMINI_2_5_PRO`
* static `GEMINI_3_1_FLASH_IMAGE`
* static `GEMINI_3_1_FLASH_LITE_PREVIEW`
* static `GEMINI_3_1_PRO_PREVIEW`
* static `GEMINI_3_5_FLASH`
* static `GEMINI_3_FLASH`
* static `GPT_4_1_2025_04_14`
* static `GPT_4_1_MINI`
* static `GPT_4_1_NANO`
* static `GPT_5`
* static `GPT_5_2`
* static `GPT_5_4`
* static `GPT_5_4_MINI`
* static `GPT_5_4_NANO`
* static `GPT_5_5`
* static `GPT_5_MINI`
* static `GROK_4`
* static `GROK_4_1_FAST`
* static `GROK_4_1_FAST_NON_REASONING`
* static `GROK_4_20_NON_REASONING`
* static `GROK_4_20_REASONING`
* static `GROK_4_FAST`
* static `GROK_CODE_FAST_1`
* static `HERMES_4_405B`
* static `HERMES_4_70B`
* static `O3`
* static `O4_MINI`
* static `SEED_1_6`
* static `SEED_1_8`
* static `SEED_2_0_LITE`

### `TextGenerationOutput`

Output from a non-streaming ``chat()`` or ``completion()`` call.

Returned by ``**`opengradient.client.llm`**.LLM.chat`` (when ``stream=False``)
and ``**`opengradient.client.llm`**.LLM.completion``.

For **chat** requests the response is in ``chat_output``; for
**completion** requests it is in ``completion_output``.  Only the
field that matches the request type will be populated.

Trust model:
    Live trust in the response comes from the **TLS channel** the SDK
    used to obtain it: when the TEE is resolved via the on-chain
    registry, the SDK pins the registry-attested TLS certificate, so
    a successful response is, by construction, from a network-attested
    TEE enclave. See ``opengradient.client.tee_registry`` for the
    pinning logic.

    ``tee_signature`` and ``tee_timestamp`` are durable proof material
    intended for **on-chain settlement verification** and offline /
    auditor use (e.g. when a response is archived and re-checked
    outside the original TLS session). The SDK does not verify the
    signature at return time, and a non-erroring response does not
    imply client-side signature verification has occurred — only that
    the TLS-pinned channel was honored.

**Attributes**

* **`data_settlement_transaction_hash`**: Blockchain transaction hash for
        the data settlement transaction. ``None`` when the provider
        does not return data settlement metadata.
* **`data_settlement_blob_id`**: Walrus blob ID for individual data
        settlement. ``None`` for private/batch settlement or when the
        provider does not return it.
* **`finish_reason`**: Reason the model stopped generating
        (e.g. ``"stop"``, ``"tool_call"``, ``"error"``).
        Only populated for chat requests.
* **`chat_output`**: Dictionary with the assistant message returned by
        a chat request.  Contains ``role``, ``content``, and
        optionally ``tool_calls``.
* **`completion_output`**: Raw text returned by a completion request.
* **`payment_hash`**: Payment hash for the x402 transaction.
* **`tee_signature`**: RSA-PSS signature over the response produced by
        the TEE enclave. Forwarded as-is from the server; verified at
        settlement on-chain, not by the SDK at return time. See the
        class-level "Trust model" note above.
* **`tee_timestamp`**: ISO-8601 timestamp from the TEE at signing
        time. Forwarded as-is alongside ``tee_signature``.

#### Constructor

```python
def __init__(
    data_settlement_transaction_hash: Optional[str] = None,
    data_settlement_blob_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chat_output: Optional[Dict] = None,
    completion_output: Optional[str] = None,
    images: Optional[List[str]] = None,
    usage: Optional[Dict] = None,
    payment_hash: Optional[str] = None,
    tee_signature: Optional[str] = None,
    tee_timestamp: Optional[str] = None,
    tee_id: Optional[str] = None,
    tee_endpoint: Optional[str] = None,
    tee_payment_address: Optional[str] = None
)
```

### `TextGenerationStream`

Iterator over ``StreamChunk`` objects from a streaming chat response.

Returned by ``**`opengradient.client.llm`**.LLM.chat`` when
``stream=True``.  Iterate over the stream to receive incremental
chunks as they arrive from the server.

Each ``StreamChunk`` contains a list of ``StreamChoice`` objects.
Access the incremental text via ``chunk.choices[0].delta.content``.
The final chunk will have ``is_final=True`` and may include
``usage`` and ``tee_signature`` / ``tee_timestamp`` fields.

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

* **`PRIVATE`**: Payment-only settlement.
        Only the payment is settled on-chain — no input or output hashes are posted.
        Your inference data remains completely off-chain, ensuring maximum privacy.
        Suitable when payment settlement is required without any on-chain record of execution.
        CLI usage: --settlement-mode private
* **`BATCH_HASHED`**: Batch settlement with hashes (default).
        Aggregates multiple inferences into a single settlement transaction
        using a Merkle tree containing input hashes, output hashes, and signatures.
        Most cost-efficient for high-volume applications.
        CLI usage: --settlement-mode batch-hashed
* **`INDIVIDUAL_FULL`**: Individual settlement with full metadata.
        Records input data, output data, timestamp, and verification on-chain.
        Provides maximum transparency and auditability.
        Higher gas costs due to larger data storage.
        CLI usage: --settlement-mode individual-full