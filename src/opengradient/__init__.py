"""
OpenGradient Python SDK for decentralized AI inference with end-to-end verification.

## Overview

The OpenGradient SDK provides programmatic access to decentralized AI infrastructure.
All LLM inference runs inside Trusted Execution Environments (TEEs) and settles
on-chain via the x402 payment protocol, giving you cryptographic proof that
inference was performed correctly.

The SDK operates across two chains with separate private keys:

- **`opengradient.client.llm`** (``og.LLM``) -- LLM chat and completion with TEE-verified execution. Pays via x402 on **Base Sepolia** (requires OPG tokens).
- **`opengradient.client.alpha`** (``og.Alpha``) -- On-chain ONNX model inference with VANILLA, TEE, or ZKML verification. Pays gas on the **OpenGradient testnet**.
- **`opengradient.client.model_hub`** (``og.ModelHub``) -- Model repository management: create, version, and upload ML models. Requires email/password auth.
- **`opengradient.client.twins`** (``og.Twins``) -- Digital twins chat via verifiable inference. Requires a twins API key.

## LLM Chat

```python
import asyncio
import opengradient as og

llm = og.LLM(private_key="0x...")

# One-time OPG token approval (idempotent -- skips if allowance is sufficient)
llm.ensure_opg_approval(opg_amount=5)

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
"""

from . import agents, alphasense
from .client import LLM, Alpha, ModelHub, Twins
from .types import (
    TEE_LLM,
    CandleOrder,
    CandleType,
    FileUploadResult,
    HistoricalInputQuery,
    InferenceMode,
    InferenceResult,
    ModelOutput,
    ModelRepository,
    SchedulerParams,
    TextGenerationOutput,
    TextGenerationStream,
    x402SettlementMode,
)

__all__ = [
    "LLM",
    "Alpha",
    "ModelHub",
    "Twins",
    "TEE_LLM",
    "InferenceMode",
    "HistoricalInputQuery",
    "SchedulerParams",
    "CandleType",
    "CandleOrder",
    "TextGenerationOutput",
    "TextGenerationStream",
    "x402SettlementMode",
    "agents",
    "alphasense",
]

__pdoc__ = {
    "account": False,
    "cli": False,
    "client": True,
    "defaults": False,
    "agents": True,
    "alphasense": True,
    "types": True,
    # Hide re-exported classes from the top-level page -- they are documented on their own submodule pages
    "LLM": False,
    "Alpha": False,
    "ModelHub": False,
    "Twins": False,
    "TEE_LLM": False,
    "InferenceMode": False,
    "TextGenerationOutput": False,
    "TextGenerationStream": False,
    "x402SettlementMode": False,
    "InferenceResult": False,
    "ModelOutput": False,
    "FileUploadResult": False,
    "ModelRepository": False,
    "CandleOrder": False,
    "CandleType": False,
    "HistoricalInputQuery": False,
    "SchedulerParams": False,
}
