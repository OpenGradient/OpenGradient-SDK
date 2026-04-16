---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / llm

# Package opengradient.client.llm

LLM chat and completion via TEE-verified execution with x402 payments.

## Classes

### `LLM`

LLM inference namespace.

Provides access to large language model completions and chat via TEE
(Trusted Execution Environment) with x402 payment protocol support.
Supports both streaming and non-streaming responses.

All request methods (``chat``, ``completion``) are async.

Before making LLM requests, ensure your wallet has approved sufficient
OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.

#### Constructor

```python
def __init__(
    private_key: str,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    tee_registry_address: str = '0x4e72238852f3c918f4E4e57AeC9280dDB0c80248'
)
```

#### Static methods

---

#### `from_url()`

```python
static def from_url(private_key: str, llm_server_url: str) ‑> `LLM`
```
**[Dev]** Create an LLM client with a hardcoded TEE endpoint URL.

Intended for development and self-hosted TEE servers. TLS certificate
verification is disabled because these servers typically use self-signed
certificates. For production use, prefer the default constructor which
resolves TEEs from the on-chain registry.

**Arguments**

* **`private_key`**: Ethereum private key for signing x402 payments.
* **`llm_server_url`**: The TEE endpoint URL (e.g. ``"https://1.2.3.4"``).

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
    response_format: Optional[`ResponseFormat`] = None,
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
* **`response_format (ResponseFormat, optional)`**: Enforce a specific output format.
        Use ``ResponseFormat(type="json_object")`` for any valid JSON (not supported
        by Anthropic models). Use ``ResponseFormat(type="json_schema", json_schema={...})``
        to enforce a strict schema (supported by all providers including Anthropic).
        Defaults to None (plain text).
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

* **`ValueError`**: If ``response_format="json_object"`` is used with an Anthropic model.
* **`RuntimeError`**: If the inference fails.

---

#### `close()`

```python
async def close(self) ‑> None
```
Cancel the background refresh loop and close the HTTP client.

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
def ensure_opg_approval(
    self,
    min_allowance: float,
    approve_amount: Optional[float] = None
) ‑> [Permit2ApprovalResult](./opg_token)
```
Ensure the Permit2 allowance stays above a minimum threshold.

Only sends a transaction when the current allowance drops below
``min_allowance``. When approval is needed, approves ``approve_amount``
(defaults to ``2 * min_allowance``) to create a buffer that survives
multiple service restarts without re-approving.

Best for backend servers that call this on startup::

    llm.ensure_opg_approval(min_allowance=5.0, approve_amount=100.0)

**Arguments**

* **`min_allowance`**: The minimum acceptable allowance in OPG. Must be
        at least 0.1 OPG.
* **`approve_amount`**: The amount of OPG to approve when a transaction
        is needed. Defaults to ``2 * min_allowance``. Must be
        >= ``min_allowance``.

**Returns**

Permit2ApprovalResult: Contains ``allowance_before``,
    ``allowance_after``, and ``tx_hash`` (None when no approval
    was needed).

**Raises**

* **`ValueError`**: If ``min_allowance`` is less than 0.1 or
        ``approve_amount`` is less than ``min_allowance``.
* **`RuntimeError`**: If the approval transaction fails.