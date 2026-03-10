"""
OpenGradient Client -- the central entry point to all SDK services.

## Overview

The `opengradient.client.client.Client` class provides unified access to four service namespaces:

- **`opengradient.client.llm`** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement (Base Sepolia OPG tokens)
- **`opengradient.client.model_hub`** -- Model repository management: create, version, and upload ML models
- **`opengradient.client.alpha`** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **`opengradient.client.twins`** -- Digital twins chat via OpenGradient verifiable inference

## Private Keys

The SDK operates across two chains:

- **`private_key`** -- used for LLM inference (``client.llm``). Pays via x402 on **Base Sepolia** with OPG tokens.
- **`alpha_private_key`** *(optional)* -- used for Alpha Testnet features (``client.alpha``). Pays gas on the **OpenGradient network** with testnet tokens. Falls back to ``private_key`` when omitted.

## Usage

```python
import opengradient as og

# Single key for both chains (backward compatible)
client = og.init(private_key="0x...")

# Separate keys: Base Sepolia OPG for LLM, OpenGradient testnet gas for Alpha
client = og.init(private_key="0xLLM_KEY...", alpha_private_key="0xALPHA_KEY...")

# One-time approval (idempotent — skips if allowance is already sufficient)
client.llm.ensure_opg_approval(opg_amount=5)

# LLM chat (TEE-verified, streamed)
for chunk in client.llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# On-chain model inference
result = client.alpha.infer(
    model_cid="your_model_cid",
    inference_mode=og.InferenceMode.VANILLA,
    model_input={"input": [1.0, 2.0, 3.0]},
)

# Model Hub (requires email auth)
client = og.init(private_key="0x...", email="you@example.com", password="...")
repo = client.model_hub.create_model("my-model", "A price prediction model")
```
"""

from .client import Client

__all__ = ["Client"]

__pdoc__ = {}
