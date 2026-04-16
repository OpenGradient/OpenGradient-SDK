---
outline: [2,4]
---

opengradient

# Package opengradient

**Version: 1.0.0**

OpenGradient Python SDK for decentralized AI inference with end-to-end verification.

## Overview

The OpenGradient SDK provides programmatic access to decentralized AI infrastructure.
All LLM inference runs inside Trusted Execution Environments (TEEs) and settles
on-chain via the x402 payment protocol, giving you cryptographic proof that
inference was performed correctly.

The SDK operates across two chains with separate private keys:

- **[llm](./client/llm)** (``og.LLM``) -- LLM chat and completion with TEE-verified execution. Pays via x402 on **Base** (requires OPG tokens).
- **[alpha](./client/alpha)** (``og.Alpha``) -- On-chain ONNX model inference with VANILLA, TEE, or ZKML verification. Pays gas on the **OpenGradient alpha testnet**.
- **[model_hub](./client/model_hub)** (``og.ModelHub``) -- Model repository management: create, version, and upload ML models. Requires email/password auth.
- **[twins](./client/twins)** (``og.Twins``) -- Digital twins chat via verifiable inference. Requires a twins API key.

See **`opengradient.types`** for shared data types (``TEE_LLM``, ``InferenceMode``, ``TextGenerationOutput``, ``x402SettlementMode``, etc.).

## LLM Chat

```python
import asyncio
import opengradient as og

llm = og.LLM(private_key="0x...")

# One-time OPG token approval (idempotent -- skips if allowance is sufficient)
llm.ensure_opg_approval(min_allowance=5)

# Chat with an LLM (TEE-verified)
response = asyncio.run(llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
))
print(response.chat_output)
```

## Streaming

```python
async def stream_example():
    llm = og.LLM(private_key="0x...")
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
```

## On-chain Model Inference

```python
alpha = og.Alpha(private_key="0x...")
result = alpha.infer(
    model_cid="your_model_cid",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)
print(result.model_output)
```

## Model Hub

```python
hub = og.ModelHub(email="you@example.com", password="...")
repo = hub.create_model("my-model", "A price prediction model")
hub.upload("model.onnx", repo.name, repo.initialVersion)
```

## Classes

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