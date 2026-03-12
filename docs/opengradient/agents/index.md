---
outline: [2,4]
---

[opengradient](../index) / agents

# Package opengradient.agents

OpenGradient Agent Framework Adapters

This module provides adapter interfaces to use OpenGradient LLMs with popular AI frameworks
like LangChain. These adapters allow seamless integration of OpenGradient models
into existing applications and agent frameworks.

## Functions

---

### `langchain_adapter()`

```python
def langchain_adapter(
    private_key: str,
    model_cid: `TEE_LLM`,
    max_tokens: int = 300,
    x402_settlement_mode: `x402SettlementMode` = x402SettlementMode.BATCH_HASHED
) ‑> [OpenGradientChatModel](./og_langchain)
```
Returns an OpenGradient LLM that implements LangChain's LLM interface
and can be plugged into LangChain agents.