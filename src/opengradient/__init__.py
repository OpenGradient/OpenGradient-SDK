"""
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

- **`opengradient.client.llm`** -- Verifiable LLM chat and completion via TEE-verified execution with x402 payments (Base Sepolia OPG tokens)
- **`opengradient.client.alpha`** -- On-chain ONNX model inference, workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **`opengradient.client.model_hub`** -- Model repository management
- **`opengradient.client.twins`** -- Digital twins chat via OpenGradient verifiable inference (requires twins API key)

## Model Hub (requires email auth)

```python
hub = og.ModelHub(email="you@example.com", password="...")
repo = hub.create_model("my-model", "A price prediction model")
hub.upload("model.onnx", repo.name, repo.initialVersion)
```

## Framework Integrations

The SDK includes adapters for popular AI frameworks -- see the `agents` submodule for LangChain and OpenAI integration.
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
    # Hide niche types from the top-level page -- they are documented under the types submodule
    "CandleOrder": False,
    "CandleType": False,
    "HistoricalInputQuery": False,
    "SchedulerParams": False,
}
