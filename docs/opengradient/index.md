---
outline: [2,3]
---

opengradient

# Package opengradient

**Version: 0.7.5**

OpenGradient Python SDK for decentralized AI inference with end-to-end verification.

## Overview

The OpenGradient SDK provides programmatic access to decentralized AI infrastructure, including:

- **LLM Inference** -- Chat and completion with major LLM providers (OpenAI, Anthropic, Google, xAI) through TEE-verified execution
- **On-chain Model Inference** -- Run ONNX models via blockchain smart contracts with VANILLA, TEE, or ZKML verification
- **Model Hub** -- Create, version, and upload ML models to the OpenGradient Model Hub

All LLM inference runs inside Trusted Execution Environments (TEEs) and settles on-chain via the x402 payment protocol, giving you cryptographic proof that inference was performed correctly.

## Quick Start

```python
import opengradient as og

# Initialize the client
client = og.init(private_key="0x...")

# One-time approval (idempotent — skips if allowance is already sufficient)
client.llm.ensure_opg_approval(opg_amount=5)

# Chat with an LLM (TEE-verified)
response = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(response.chat_output)

# Stream a response
for chunk in client.llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Explain TEE in one paragraph."}],
    max_tokens=300,
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Run on-chain ONNX model inference
result = client.alpha.infer(
    model_cid="your_model_cid",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)
print(result.model_output)
```

## Private Keys

The SDK operates across two chains. You can use a single key for both, or provide separate keys:

- **``private_key``** -- pays for LLM inference via x402 on **Base Sepolia** (requires OPG tokens)
- **``alpha_private_key``** *(optional)* -- pays gas for Alpha Testnet on-chain inference on the **OpenGradient network** (requires testnet gas tokens). Falls back to ``private_key`` when omitted.

```python
# Separate keys for each chain
client = og.init(private_key="0xBASE_KEY...", alpha_private_key="0xALPHA_KEY...")
```

## Client Namespaces

The [Client](./client/index) object exposes four namespaces:

- **[llm](./client/llm)** -- Verifiable LLM chat and completion via TEE-verified execution with x402 payments (Base Sepolia OPG tokens)
- **[alpha](./client/alpha)** -- On-chain ONNX model inference, workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **[model_hub](./client/model_hub)** -- Model repository management
- **[twins](./client/twins)** -- Digital twins chat via OpenGradient verifiable inference (requires twins API key)

## Model Hub (requires email auth)

```python
client = og.init(
    private_key="0x...",
    email="you@example.com",
    password="...",
)

repo = client.model_hub.create_model("my-model", "A price prediction model")
client.model_hub.upload("model.onnx", repo.name, repo.initialVersion)
```

## Framework Integrations

The SDK includes adapters for popular AI frameworks -- see the `agents` submodule for LangChain and OpenAI integration.

## Submodules

* [**agents**](./agents/index): OpenGradient Agent Framework Adapters
* [**alphasense**](./alphasense/index): OpenGradient AlphaSense Tools
* [**client**](./client/index): OpenGradient Client -- the central entry point to all SDK services.
* [**types**](./types): OpenGradient Specific Types
* [**workflow_models**](./workflow_models/index): OpenGradient Hardcoded Models

## Functions

---

### `init()`

```python
def init(
    private_key: str,
    alpha_private_key: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) ‑> `Client`
```
Initialize the global OpenGradient client.

This is the recommended way to get started. It creates a `Client` instance
and stores it as the global client for convenience.

**Arguments**

* **`private_key`**: Private key whose wallet holds **Base Sepolia OPG tokens**
        for x402 LLM payments.
* **`alpha_private_key`**: Private key whose wallet holds **OpenGradient testnet
        gas tokens** for on-chain inference. Optional -- falls back to
        ``private_key`` for backward compatibility.
* **`email`**: Email for Model Hub authentication. Optional.
* **`password`**: Password for Model Hub authentication. Optional.
    **kwargs: Additional arguments forwarded to `Client`.

**Returns**

The newly created `Client` instance.

## Classes

### `Client`

Main OpenGradient SDK client.

Provides unified access to all OpenGradient services including LLM inference,
on-chain model inference, and the Model Hub.

The client operates across two chains:

- **LLM inference** (``client.llm``) settles via x402 on **Base Sepolia**
  using OPG tokens (funded by ``private_key``).
- **Alpha Testnet** (``client.alpha``) runs on the **OpenGradient network**
  using testnet gas tokens (funded by ``alpha_private_key``, or ``private_key``
  when not provided).

#### Constructor

```python
def __init__(
    private_key: str,
    alpha_private_key: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    twins_api_key: Optional[str] = None,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    api_url: str = 'https://sdk-devnet.opengradient.ai',
    contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE',
    og_llm_server_url: Optional[str] = 'https://3.15.214.21:443',
    og_llm_streaming_server_url: Optional[str] = 'https://3.15.214.21:443'
)
```

**Arguments**

* **`private_key`**: Private key whose wallet holds **Base Sepolia OPG tokens**
        for x402 LLM payments.
* **`alpha_private_key`**: Private key whose wallet holds **OpenGradient testnet
        gas tokens** for on-chain inference. Optional -- falls back to
        ``private_key`` for backward compatibility.
* **`email`**: Email for Model Hub authentication. Optional.
* **`password`**: Password for Model Hub authentication. Optional.
* **`twins_api_key`**: API key for digital twins chat (twin.fun). Optional.
* **`rpc_url`**: RPC URL for the OpenGradient Alpha Testnet.
* **`api_url`**: API URL for the OpenGradient API.
* **`contract_address`**: Inference contract address.
* **`og_llm_server_url`**: OpenGradient LLM server URL.
* **`og_llm_streaming_server_url`**: OpenGradient LLM streaming server URL.

#### Methods

---

#### `close()`

```python
def close(self) ‑> None
```
Close underlying SDK resources.

#### Variables

* [**`alpha`**](./client/alpha): Alpha Testnet features including on-chain inference, workflow management, and ML model execution.
* [**`llm`**](./client/llm): LLM chat and completion via TEE-verified execution.
* [**`model_hub`**](./client/model_hub): Model Hub for creating, versioning, and uploading ML models.
* [**`twins`**](./client/twins): Digital twins chat via OpenGradient verifiable inference. ``None`` when no ``twins_api_key`` is provided.

### `InferenceMode`

Enum for the different inference modes available for inference (VANILLA, ZKML, TEE)

#### Variables

* static `TEE`
* static `VANILLA`
* static `ZKML`

### `TEE_LLM`

Enum for LLM models available for TEE (Trusted Execution Environment) execution.

TEE mode provides cryptographic verification that inference was performed
correctly in a secure enclave. Use this for applications requiring
auditability and tamper-proof AI inference.

#### Variables

* static `CLAUDE_HAIKU_4_5`
* static `CLAUDE_OPUS_4_5`
* static `CLAUDE_OPUS_4_6`
* static `CLAUDE_SONNET_4_5`
* static `CLAUDE_SONNET_4_6`
* static `GEMINI_2_5_FLASH`
* static `GEMINI_2_5_FLASH_LITE`
* static `GEMINI_2_5_PRO`
* static `GEMINI_3_FLASH`
* static `GEMINI_3_PRO`
* static `GPT_4_1_2025_04_14`
* static `GPT_5`
* static `GPT_5_2`
* static `GPT_5_MINI`
* static `GROK_4`
* static `GROK_4_1_FAST`
* static `GROK_4_1_FAST_NON_REASONING`
* static `GROK_4_FAST`
* static `O4_MINI`

### `TextGenerationOutput`

Output from a non-streaming ``chat()`` or ``completion()`` call.

Returned by ``**`opengradient.client.llm`**.LLM.chat`` (when ``stream=False``)
and ``**`opengradient.client.llm`**.LLM.completion``.

For **chat** requests the response is in ``chat_output``; for
**completion** requests it is in ``completion_output``.  Only the
field that matches the request type will be populated.

Every response includes a ``tee_signature`` and ``tee_timestamp``
that can be used to cryptographically verify the inference was
performed inside a TEE enclave.

**Attributes**

* **`transaction_hash`**: Blockchain transaction hash.  Set to
        ``"external"`` for TEE-routed providers.
* **`finish_reason`**: Reason the model stopped generating
        (e.g. ``"stop"``, ``"tool_call"``, ``"error"``).
        Only populated for chat requests.
* **`chat_output`**: Dictionary with the assistant message returned by
        a chat request.  Contains ``role``, ``content``, and
        optionally ``tool_calls``.
* **`completion_output`**: Raw text returned by a completion request.
* **`payment_hash`**: Payment hash for the x402 transaction.
* **`tee_signature`**: RSA-PSS signature over the response produced
        by the TEE enclave.
* **`tee_timestamp`**: ISO-8601 timestamp from the TEE at signing
        time.

#### Constructor

```python
def __init__(
    transaction_hash: str,
    finish_reason: Optional[str] = None,
    chat_output: Optional[Dict] = None,
    completion_output: Optional[str] = None,
    payment_hash: Optional[str] = None,
    tee_signature: Optional[str] = None,
    tee_timestamp: Optional[str] = None
)
```

#### Variables

* static `chat_output` : Optional[Dict] - Dictionary with the assistant message returned by a chat request. Contains ``role``, ``content``, and optionally ``tool_calls``.
* static `completion_output` : Optional[str] - Raw text returned by a completion request.
* static `finish_reason` : Optional[str] - Reason the model stopped generating (e.g. ``"stop"``, ``"tool_call"``, ``"error"``). Only populated for chat requests.
* static `payment_hash` : Optional[str] - Payment hash for the x402 transaction.
* static `tee_signature` : Optional[str] - RSA-PSS signature over the response produced by the TEE enclave.
* static `tee_timestamp` : Optional[str] - ISO-8601 timestamp from the TEE at signing time.
* static `transaction_hash` : str - Blockchain transaction hash. Set to ``"external"`` for TEE-routed providers.

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

* **`SETTLE`**: Most private settlement method.
        Only the payment is settled on-chain — no input or output hashes are posted to the chain.
        Your inference data remains completely off-chain, ensuring maximum privacy.
        Suitable for applications where payment settlement is required without any on-chain record of execution.
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