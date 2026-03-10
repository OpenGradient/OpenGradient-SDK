---
outline: [2,3]
---

[opengradient](../index) / [client](./index) / twins

# Package opengradient.client.twins

Digital twins chat via OpenGradient verifiable inference.

## Classes

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

* **`OpenGradientError`**: If the request fails.