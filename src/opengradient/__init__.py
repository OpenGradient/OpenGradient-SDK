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
import opengradient as og

# Initialize the client
client = og.init(private_key="0x...")

# One-time approval (idempotent — skips if allowance is already sufficient)
client.llm.ensure_opg_approval(opg_amount=5)

# Chat with an LLM (TEE-verified)
response = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_3_5_HAIKU,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
)
print(response.chat_output)

# Stream a response
for chunk in client.llm.chat(
    model=og.TEE_LLM.GPT_4O,
    messages=[{"role": "user", "content": "Explain TEE in one paragraph."}],
    max_tokens=300,
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Run on-chain ONNX model inference
result = client.alpha.infer(
    model_cid="your_model_cid",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)
print(result.model_output)
```

## Private Keys

The SDK operates across two chains. You can use a single key for both, or provide separate keys:

- **``private_key``** -- pays for LLM inference via x402 on **Base Sepolia** (requires OPG tokens)
- **``alpha_private_key``** *(optional)* -- pays gas for Alpha Testnet on-chain inference on the **OpenGradient network** (requires testnet gas tokens). Falls back to ``private_key`` when omitted.

```python
# Separate keys for each chain
client = og.init(private_key="0xBASE_KEY...", alpha_private_key="0xALPHA_KEY...")
```

## Client Namespaces

The `opengradient.client.Client` object exposes four namespaces:

- **`opengradient.client.llm`** -- Verifiable LLM chat and completion via TEE-verified execution with x402 payments (Base Sepolia OPG tokens)
- **`opengradient.client.alpha`** -- On-chain ONNX model inference, workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **`opengradient.client.model_hub`** -- Model repository management
- **`opengradient.client.twins`** -- Digital twins chat via OpenGradient verifiable inference (requires twins API key)

## Model Hub (requires email auth)

```python
client = og.init(
    private_key="0x...",
    email="you@example.com",
    password="...",
)

repo = client.model_hub.create_model("my-model", "A price prediction model")
client.model_hub.upload("model.onnx", repo.name, repo.initialVersion)
```

## Framework Integrations

The SDK includes adapters for popular AI frameworks -- see the `agents` submodule for LangChain and OpenAI integration.
"""

from typing import Optional

from . import agents, alphasense
from .client import Client
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
    ResponseFormat,
    SchedulerParams,
    TextGenerationOutput,
    TextGenerationStream,
    x402SettlementMode,
)

global_client: Optional[Client] = None
"""Global client instance. Set by calling `init()`."""


def init(
    private_key: str,
    alpha_private_key: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs,
) -> Client:
    """Initialize the global OpenGradient client.

    This is the recommended way to get started. It creates a `Client` instance
    and stores it as the global client for convenience.

    Args:
        private_key: Private key whose wallet holds **Base Sepolia OPG tokens**
            for x402 LLM payments.
        alpha_private_key: Private key whose wallet holds **OpenGradient testnet
            gas tokens** for on-chain inference. Optional -- falls back to
            ``private_key`` for backward compatibility.
        email: Email for Model Hub authentication. Optional.
        password: Password for Model Hub authentication. Optional.
        **kwargs: Additional arguments forwarded to `Client`.

    Returns:
        The newly created `Client` instance.

    Usage:
        import opengradient as og
        client = og.init(private_key="0x...")
        client.llm.ensure_opg_approval(opg_amount=5)
        response = client.llm.chat(model=og.TEE_LLM.GPT_4O, messages=[...])
    """
    global global_client
    global_client = Client(
        private_key=private_key,
        alpha_private_key=alpha_private_key,
        email=email,
        password=password,
        **kwargs,
    )
    return global_client


__all__ = [
    "Client",
    "global_client",
    "init",
    "TEE_LLM",
    "InferenceMode",
    "HistoricalInputQuery",
    "SchedulerParams",
    "CandleType",
    "CandleOrder",
    "ResponseFormat",
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
    "global_client": False,
}
