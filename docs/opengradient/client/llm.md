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

#### Constructor

```python
def __init__(wallet_account: `LocalAccount`, og_llm_server_url: str, og_llm_streaming_server_url: str, network_filter: str = 'og-evm')
```

#### Methods

---

#### `chat()`

```python
def chat(self, model: `TEE_LLM`, messages: List[Dict], max_tokens: int = 100, stop_sequence: Optional[List[str]] = None, temperature: float = 0.0, tools: Optional[List[Dict]] = [], tool_choice: Optional[str] = None, x402_settlement_mode: Optional[`x402SettlementMode`] = x402SettlementMode.SETTLE_BATCH, stream: bool = False) ‑> Union[`TextGenerationOutput`, `TextGenerationStream`]
```
Perform inference on an LLM model using chat via TEE.

**Arguments**

* **`model (TEE_LLM)`**: The model to use (e.g., TEE_LLM.CLAUDE_3_5_HAIKU).
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

**Raises**

* **`OpenGradientError`**: If the inference fails.

---

#### `completion()`

```python
def completion(self, model: `TEE_LLM`, prompt: str, max_tokens: int = 100, stop_sequence: Optional[List[str]] = None, temperature: float = 0.0, x402_settlement_mode: Optional[`x402SettlementMode`] = x402SettlementMode.SETTLE_BATCH) ‑> `TextGenerationOutput`
```
Perform inference on an LLM model using completions via TEE.

**Arguments**

* **`model (TEE_LLM)`**: The model to use (e.g., TEE_LLM.CLAUDE_3_5_HAIKU).
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

**Raises**

* **`OpenGradientError`**: If the inference fails.