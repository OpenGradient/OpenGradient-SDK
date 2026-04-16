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
    private_key: Optional[str] = None,
    model_cid: Union[`TEE_LLM`, str, ForwardRef(None)] = None,
    model: Union[`TEE_LLM`, str, ForwardRef(None)] = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
    x402_settlement_mode: `x402SettlementMode` = x402SettlementMode.BATCH_HASHED,
    client: Optional[`LLM`] = None,
    rpc_url: Optional[str] = None,
    tee_registry_address: Optional[str] = None,
    llm_server_url: Optional[str] = None
) ‑> [OpenGradientChatModel](./og_langchain)
```
Returns an OpenGradient LLM that implements LangChain's LLM interface
and can be plugged into LangChain agents.