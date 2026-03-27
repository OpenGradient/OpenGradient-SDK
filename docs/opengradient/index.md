---
outline: [2,4]
---

opengradient

# Package opengradient

**Version: 0.9.3**

OpenGradient Python SDK for decentralized AI inference with end-to-end verification.

## Overview

The OpenGradient SDK provides programmatic access to decentralized AI infrastructure.
All LLM inference runs inside Trusted Execution Environments (TEEs) and settles
on-chain via the x402 payment protocol, giving you cryptographic proof that
inference was performed correctly.

The SDK operates across two chains with separate private keys:

- **[llm](./client/llm)** (``og.LLM``) -- LLM chat and completion with TEE-verified execution. Pays via x402 on **Base Sepolia** (requires OPG tokens).
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