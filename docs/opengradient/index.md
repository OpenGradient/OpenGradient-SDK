---
outline: [2,3]
---

opengradient

# Package opengradient

**Version: 0.7.6**

OpenGradient Python SDK for decentralized AI inference with end-to-end verification.

## Overview

The OpenGradient SDK provides programmatic access to decentralized AI infrastructure, including:

- **LLM Inference** -- Chat and completion with major LLM providers (OpenAI, Anthropic, Google, xAI) through TEE-verified execution
- **On-chain Model Inference** -- Run ONNX models via blockchain smart contracts with VANILLA, TEE, or ZKML verification
- **Model Hub** -- Create, version, and upload ML models to the OpenGradient Model Hub

All LLM inference runs inside Trusted Execution Environments (TEEs) and settles on-chain via the x402 payment protocol, giving you cryptographic proof that inference was performed correctly.

## Quick Start

```python
import asyncio
import opengradient as og

llm = og.LLM(private_key="0x...")

# One-time approval (idempotent — skips if allowance is already sufficient)
llm.ensure_opg_approval(opg_amount=5)

# Chat with an LLM (TEE-verified)
response = asyncio.run(llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
))
print(response.chat_output)

# Stream a response
async def stream_example():
    stream = await llm.chat(
        model=og.TEE_LLM.GPT_5,
        messages=[{"role": "user", "content": "Explain TEE in one paragraph."}],
        max_tokens=300,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(stream_example())

# Run on-chain ONNX model inference
alpha = og.Alpha(private_key="0x...")
result = alpha.infer(
    model_cid="your_model_cid",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)
print(result.model_output)
```

## Private Keys

The SDK operates across two chains. Use separate keys for each:

- **LLM** (``og.LLM``) -- pays for inference via x402 on **Base Sepolia** (requires OPG tokens)
- **Alpha** (``og.Alpha``) -- pays gas for on-chain inference on the **OpenGradient network** (requires testnet gas tokens)

## Modules

- **[llm](./client/llm)** -- Verifiable LLM chat and completion via TEE-verified execution with x402 payments (Base Sepolia OPG tokens)
- **[alpha](./client/alpha)** -- On-chain ONNX model inference, workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **[model_hub](./client/model_hub)** -- Model repository management
- **[twins](./client/twins)** -- Digital twins chat via OpenGradient verifiable inference (requires twins API key)

## Model Hub (requires email auth)

```python
hub = og.ModelHub(email="you@example.com", password="...")
repo = hub.create_model("my-model", "A price prediction model")
hub.upload("model.onnx", repo.name, repo.initialVersion)
```

## Framework Integrations

The SDK includes adapters for popular AI frameworks -- see the `agents` submodule for LangChain and OpenAI integration.

## Submodules

* [**agents**](./agents/index): OpenGradient Agent Framework Adapters
* [**alphasense**](./alphasense/index): OpenGradient AlphaSense Tools
* [**client**](./client/index): OpenGradient Client -- service modules for the SDK.
* [**types**](./types): OpenGradient Specific Types
* [**workflow_models**](./workflow_models/index): OpenGradient Hardcoded Models

## Classes

### `Alpha`

Alpha Testnet features namespace.

This class provides access to features that are only available on the Alpha Testnet,
including on-chain ONNX model inference, workflow deployment, and execution.

#### Constructor

```python
def __init__(
    private_key: str,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    inference_contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE',
    api_url: str = 'https://sdk-devnet.opengradient.ai'
)
```

#### Methods

---

#### `infer()`

```python
def infer(
    self,
    model_cid: str,
    inference_mode: `InferenceMode`,
    model_input: Dict[str, Union[str, int, float, List, `ndarray`]],
    max_retries: Optional[int] = None
) ‑> `InferenceResult`
```
Perform inference on a model.

**Arguments**

* **`model_cid (str)`**: The unique content identifier for the model from IPFS.
* **`inference_mode (InferenceMode)`**: The inference mode.
* **`model_input (Dict[str, Union[str, int, float, List, np.ndarray]])`**: The input data for the model.
* **`max_retries (int, optional)`**: Maximum number of retry attempts. Defaults to 5.

**Returns**

InferenceResult (InferenceResult): A dataclass object containing the transaction hash and model output.
    transaction_hash (str): Blockchain hash for the transaction
    model_output (Dict[str, np.ndarray]): Output of the ONNX model

**Raises**

* **`RuntimeError`**: If the inference fails.

---

#### `new_workflow()`

```python
def new_workflow(
    self,
    model_cid: str,
    input_query: `HistoricalInputQuery`,
    input_tensor_name: str,
    scheduler_params: Optional[`SchedulerParams`] = None
) ‑> str
```
Deploy a new workflow contract with the specified parameters.

This function deploys a new workflow contract on OpenGradient that connects
an AI model with its required input data. When executed, the workflow will fetch
the specified model, evaluate the input query to get data, and perform inference.

The workflow can be set to execute manually or automatically via a scheduler.

**Arguments**

* **`model_cid (str)`**: CID of the model to be executed from the Model Hub
* **`input_query (HistoricalInputQuery)`**: Input definition for the model inference,
        will be evaluated at runtime for each inference
* **`input_tensor_name (str)`**: Name of the input tensor expected by the model
* **`scheduler_params (Optional[SchedulerParams])`**: Scheduler configuration for automated execution:
        - frequency: Execution frequency in seconds
        - duration_hours: How long the schedule should live for

**Returns**

str: Deployed contract address. If scheduler_params was provided, the workflow
     will be automatically executed according to the specified schedule.

**Raises**

* **`Exception`**: If transaction fails or gas estimation fails

---

#### `read_workflow_history()`

```python
def read_workflow_history(
    self,
    contract_address: str,
    num_results: int
) ‑> List[`ModelOutput`]
```
Gets historical inference results from a workflow contract.

Retrieves the specified number of most recent inference results from the contract's
storage, with the most recent result first.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract
* **`num_results (int)`**: Number of historical results to retrieve

**Returns**

List[ModelOutput]: List of historical inference results

---

#### `read_workflow_result()`

```python
def read_workflow_result(self, contract_address: str) ‑> `ModelOutput`
```
Reads the latest inference result from a deployed workflow contract.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract

**Returns**

ModelOutput: The inference result from the contract

**Raises**

* **`ContractLogicError`**: If the transaction fails
* **`Web3Error`**: If there are issues with the web3 connection or contract interaction

---

#### `run_workflow()`

```python
def run_workflow(self, contract_address: str) ‑> `ModelOutput`
```
Triggers the run() function on a deployed workflow contract and returns the result.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract

**Returns**

ModelOutput: The inference result from the contract

**Raises**

* **`ContractLogicError`**: If the transaction fails
* **`Web3Error`**: If there are issues with the web3 connection or contract interaction

#### Variables

* `inference_abi` : dict
* `precompile_abi` : dict

### `InferenceMode`

Enum for the different inference modes available for inference (VANILLA, ZKML, TEE)

#### Variables

* static `TEE`
* static `VANILLA`
* static `ZKML`

### `LLM`

LLM inference namespace.

Provides access to large language model completions and chat via TEE
(Trusted Execution Environment) with x402 payment protocol support.
Supports both streaming and non-streaming responses.

All request methods (``chat``, ``completion``) are async.

Before making LLM requests, ensure your wallet has approved sufficient
OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.
This only sends an on-chain transaction when the current allowance is
below the requested amount.

#### Constructor

```python
def __init__(
    private_key: str,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    tee_registry_address: str = '0x4e72238852f3c918f4E4e57AeC9280dDB0c80248',
    llm_server_url: Optional[str] = None
)
```

#### Methods

---

#### `chat()`

```python
async def chat(
    self,
    model: `TEE_LLM`,
    messages: List[Dict],
    max_tokens: int = 100,
    stop_sequence: Optional[List[str]] = None,
    temperature: float = 0.0,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    x402_settlement_mode: `x402SettlementMode` = x402SettlementMode.BATCH_HASHED,
    stream: bool = False
) ‑> Union[`TextGenerationOutput`, AsyncGenerator[`StreamChunk`, None]]
```
Perform inference on an LLM model using chat via TEE.

**Arguments**

* **`model (TEE_LLM)`**: The model to use (e.g., TEE_LLM.CLAUDE_HAIKU_4_5).
* **`messages (List[Dict])`**: The messages that will be passed into the chat.
* **`max_tokens (int)`**: Maximum number of tokens for LLM output. Default is 100.
* **`stop_sequence (List[str], optional)`**: List of stop sequences for LLM.
* **`temperature (float)`**: Temperature for LLM inference, between 0 and 1.
* **`tools (List[dict], optional)`**: Set of tools for function calling.
* **`tool_choice (str, optional)`**: Sets a specific tool to choose.
* **`x402_settlement_mode (x402SettlementMode, optional)`**: Settlement mode for x402 payments.
        - PRIVATE: Payment only, no input/output data on-chain (most privacy-preserving).
        - BATCH_HASHED: Aggregates inferences into a Merkle tree with input/output hashes and signatures (default, most cost-efficient).
        - INDIVIDUAL_FULL: Records input, output, timestamp, and verification on-chain (maximum auditability).
        Defaults to BATCH_HASHED.
* **`stream (bool, optional)`**: Whether to stream the response. Default is False.

**Returns**

Union[TextGenerationOutput, AsyncGenerator[StreamChunk, None]]:
    - If stream=False: TextGenerationOutput with chat_output, transaction_hash, finish_reason, and payment_hash
    - If stream=True: Async generator yielding StreamChunk objects

**`TextGenerationOutput` fields:**

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

**Raises**

* **`RuntimeError`**: If the inference fails.

---

#### `close()`

```python
async def close(self) ‑> None
```
Close the underlying HTTP client.

---

#### `completion()`

```python
async def completion(
    self,
    model: `TEE_LLM`,
    prompt: str,
    max_tokens: int = 100,
    stop_sequence: Optional[List[str]] = None,
    temperature: float = 0.0,
    x402_settlement_mode: `x402SettlementMode` = x402SettlementMode.BATCH_HASHED
) ‑> `TextGenerationOutput`
```
Perform inference on an LLM model using completions via TEE.

**Arguments**

* **`model (TEE_LLM)`**: The model to use (e.g., TEE_LLM.CLAUDE_HAIKU_4_5).
* **`prompt (str)`**: The input prompt for the LLM.
* **`max_tokens (int)`**: Maximum number of tokens for LLM output. Default is 100.
* **`stop_sequence (List[str], optional)`**: List of stop sequences for LLM. Default is None.
* **`temperature (float)`**: Temperature for LLM inference, between 0 and 1. Default is 0.0.
* **`x402_settlement_mode (x402SettlementMode, optional)`**: Settlement mode for x402 payments.
        - PRIVATE: Payment only, no input/output data on-chain (most privacy-preserving).
        - BATCH_HASHED: Aggregates inferences into a Merkle tree with input/output hashes and signatures (default, most cost-efficient).
        - INDIVIDUAL_FULL: Records input, output, timestamp, and verification on-chain (maximum auditability).
        Defaults to BATCH_HASHED.

**Returns**

TextGenerationOutput: Generated text results including:
    - Transaction hash ("external" for TEE providers)
    - String of completion output
    - Payment hash for x402 transactions

**`TextGenerationOutput` fields:**

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

**Raises**

* **`RuntimeError`**: If the inference fails.

---

#### `ensure_opg_approval()`

```python
def ensure_opg_approval(self, opg_amount: float) ‑> `Permit2ApprovalResult`
```
Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

Checks the current Permit2 allowance for the wallet. If the allowance
is already >= the requested amount, returns immediately without sending
a transaction. Otherwise, sends an ERC-20 approve transaction.

**Arguments**

* **`opg_amount`**: Minimum number of OPG tokens required (e.g. ``0.05``
        for 0.05 OPG). Must be at least 0.05 OPG.

**Returns**

Permit2ApprovalResult: Contains ``allowance_before``,
    ``allowance_after``, and ``tx_hash`` (None when no approval
    was needed).

**`Permit2ApprovalResult` fields:**

* **`allowance_before`**: The Permit2 allowance before the method ran.
* **`allowance_after`**: The Permit2 allowance after the method ran.
* **`tx_hash`**: Transaction hash of the approval, or None if no transaction was needed.

**Raises**

* **`ValueError`**: If the OPG amount is less than 0.05.
* **`RuntimeError`**: If the approval transaction fails.

### `ModelHub`

Model Hub namespace.

Provides access to the OpenGradient Model Hub for creating, versioning,
and uploading ML models. Requires email/password authentication.

#### Constructor

```python
def __init__(email: Optional[str] = None, password: Optional[str] = None)
```

#### Methods

---

#### `create_model()`

```python
def create_model(
    self,
    model_name: str,
    model_desc: str,
    version: str = '1.00'
) ‑> `ModelRepository`
```
Create a new model with the given model_name and model_desc, and a specified version.

**Arguments**

* **`model_name (str)`**: The name of the model.
* **`model_desc (str)`**: The description of the model.
* **`version (str)`**: The version identifier (default is "1.00").

**Returns**

dict: The server response containing model details.

**Raises**

* **`CreateModelError`**: If the model creation fails.

---

#### `create_version()`

```python
def create_version(
    self,
    model_name: str,
    notes: str = '',
    is_major: bool = False
) ‑> dict
```
Create a new version for the specified model.

**Arguments**

* **`model_name (str)`**: The unique identifier for the model.
* **`notes (str, optional)`**: Notes for the new version.
* **`is_major (bool, optional)`**: Whether this is a major version update. Defaults to False.

**Returns**

dict: The server response containing version details.

**Raises**

* **`Exception`**: If the version creation fails.

---

#### `list_files()`

```python
def list_files(self, model_name: str, version: str) ‑> List[Dict]
```
List files for a specific version of a model.

**Arguments**

* **`model_name (str)`**: The unique identifier for the model.
* **`version (str)`**: The version identifier for the model.

**Returns**

List[Dict]: A list of dictionaries containing file information.

**Raises**

* **`RuntimeError`**: If the file listing fails.

---

#### `upload()`

```python
def upload(
    self,
    model_path: str,
    model_name: str,
    version: str
) ‑> `FileUploadResult`
```
Upload a model file to the server.

**Arguments**

* **`model_path (str)`**: The path to the model file.
* **`model_name (str)`**: The unique identifier for the model.
* **`version (str)`**: The version identifier for the model.

**Returns**

dict: The processed result.

**Raises**

* **`RuntimeError`**: If the upload fails.

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
    tee_timestamp: Optional[str] = None,
    tee_id: Optional[str] = None,
    tee_endpoint: Optional[str] = None,
    tee_payment_address: Optional[str] = None
)
```

#### Variables

* static `chat_output` : Optional[Dict] - Dictionary with the assistant message returned by a chat request. Contains ``role``, ``content``, and optionally ``tool_calls``.
* static `completion_output` : Optional[str] - Raw text returned by a completion request.
* static `finish_reason` : Optional[str] - Reason the model stopped generating (e.g. ``"stop"``, ``"tool_call"``, ``"error"``). Only populated for chat requests.
* static `payment_hash` : Optional[str] - Payment hash for the x402 transaction.
* static `tee_endpoint` : Optional[str] - Endpoint URL of the TEE that served this request, as registered on-chain.
* static `tee_id` : Optional[str] - On-chain TEE registry ID (keccak256 of the enclave's public key) of the TEE that served this request.
* static `tee_payment_address` : Optional[str] - Payment address registered for the TEE that served this request.
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

### `Twins`

Digital twins chat namespace.

Provides access to digital twin conversations backed by OpenGradient
verifiable inference. Browse available twins at https://twin.fun.

#### Constructor

```python
def __init__(api_key: str)
```

#### Methods

---

#### `chat()`

```python
def chat(
    self,
    twin_id: str,
    model: `TEE_LLM`,
    messages: List[Dict],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) ‑> `TextGenerationOutput`
```
Chat with a digital twin.

**Arguments**

* **`twin_id`**: The unique identifier of the digital twin.
* **`model`**: The model to use for inference (e.g., TEE_LLM.GROK_4_1_FAST_NON_REASONING).
* **`messages`**: The conversation messages to send.
* **`temperature`**: Sampling temperature. Optional.
* **`max_tokens`**: Maximum number of tokens for the response. Optional.

**Returns**

TextGenerationOutput: Generated text results including chat_output and finish_reason.

**`TextGenerationOutput` fields:**

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

**Raises**

* **`RuntimeError`**: If the request fails.

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

#### Variables

* static `BATCH_HASHED`
* static `INDIVIDUAL_FULL`
* static `PRIVATE`