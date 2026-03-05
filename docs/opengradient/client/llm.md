---
outline: [2,3]
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

Before making LLM requests, ensure your wallet has approved sufficient
OPG tokens for Permit2 spending by calling ``ensure_opg_approval``.
This only sends an on-chain transaction when the current allowance is
below the requested amount.

#### Constructor

```python
def __init__(
    wallet_account: `LocalAccount`,
    og_llm_server_url: str,
    og_llm_streaming_server_url: str
)
```

#### Methods

---

#### `chat()`

```python
def chat(
    self,
    model: `TEE_LLM`,
    messages: List[Dict],
    max_tokens: int = 100,
    stop_sequence: Optional[List[str]] = None,
    temperature: float = 0.0,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    x402_settlement_mode: Optional[`x402SettlementMode`] = x402SettlementMode.SETTLE_BATCH,
    stream: bool = False
) ‑> Union[`TextGenerationOutput`, `TextGenerationStream`]
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
        - SETTLE: Records input/output hashes only (most privacy-preserving).
        - SETTLE_BATCH: Aggregates multiple inferences into batch hashes (most cost-efficient).
        - SETTLE_METADATA: Records full model info, complete input/output data, and all metadata.
        Defaults to SETTLE_BATCH.
* **`stream (bool, optional)`**: Whether to stream the response. Default is False.

**Returns**

Union[TextGenerationOutput, TextGenerationStream]:
    - If stream=False: TextGenerationOutput with chat_output, transaction_hash, finish_reason, and payment_hash
    - If stream=True: TextGenerationStream yielding StreamChunk objects with typed deltas (true streaming via threading)

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

* **`OpenGradientError`**: If the inference fails.

---

#### `close()`

```python
def close(self) ‑> None
```

---

#### `completion()`

```python
def completion(
    self,
    model: `TEE_LLM`,
    prompt: str,
    max_tokens: int = 100,
    stop_sequence: Optional[List[str]] = None,
    temperature: float = 0.0,
    x402_settlement_mode: Optional[`x402SettlementMode`] = x402SettlementMode.SETTLE_BATCH
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
        - SETTLE: Records input/output hashes only (most privacy-preserving).
        - SETTLE_BATCH: Aggregates multiple inferences into batch hashes (most cost-efficient).
        - SETTLE_METADATA: Records full model info, complete input/output data, and all metadata.
        Defaults to SETTLE_BATCH.

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

* **`OpenGradientError`**: If the inference fails.

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
* **`OpenGradientError`**: If the approval transaction fails.