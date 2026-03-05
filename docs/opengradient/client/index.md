---
outline: [2,3]
---

[opengradient](../index) / client

# Package opengradient.client

OpenGradient Client -- the central entry point to all SDK services.

## Overview

The [Client](./client) class provides unified access to four service namespaces:

- **[llm](./llm)** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement (Base Sepolia OPG tokens)
- **[model_hub](./model_hub)** -- Model repository management: create, version, and upload ML models
- **[alpha](./alpha)** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **[twins](./twins)** -- Digital twins chat via OpenGradient verifiable inference

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

## Submodules

* [alpha](./alpha): Alpha Testnet features for OpenGradient SDK.
* [client](./client): Main Client class that unifies all OpenGradient service namespaces.
* [exceptions](./exceptions): Exception types for OpenGradient SDK errors.
* [llm](./llm): LLM chat and completion via TEE-verified execution with x402 payments.
* [model_hub](./model_hub): Model Hub for creating, versioning, and uploading ML models.
* [opg_token](./opg_token): OPG token Permit2 approval utilities for x402 payments.
* [twins](./twins): Digital twins chat via OpenGradient verifiable inference.

## Classes

### `Client`

Main OpenGradient SDK client.

Provides unified access to all OpenGradient services including LLM inference,
on-chain model inference, and the Model Hub.

The client operates across two chains:

- **LLM inference** (``client.llm``) settles via x402 on **Base Sepolia**
  using OPG tokens (funded by ``private_key``).
- **Alpha Testnet** (``client.alpha``) runs on the **OpenGradient network**
  using testnet gas tokens (funded by ``alpha_private_key``, or ``private_key``
  when not provided).

#### Constructor

```python
def __init__(
    private_key: str,
    alpha_private_key: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    twins_api_key: Optional[str] = None,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    api_url: str = 'https://sdk-devnet.opengradient.ai',
    contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE',
    og_llm_server_url: Optional[str] = 'https://3.15.214.21:443',
    og_llm_streaming_server_url: Optional[str] = 'https://3.15.214.21:443'
)
```

**Arguments**

* **`private_key`**: Private key whose wallet holds **Base Sepolia OPG tokens**
        for x402 LLM payments.
* **`alpha_private_key`**: Private key whose wallet holds **OpenGradient testnet
        gas tokens** for on-chain inference. Optional -- falls back to
        ``private_key`` for backward compatibility.
* **`email`**: Email for Model Hub authentication. Optional.
* **`password`**: Password for Model Hub authentication. Optional.
* **`twins_api_key`**: API key for digital twins chat (twin.fun). Optional.
* **`rpc_url`**: RPC URL for the OpenGradient Alpha Testnet.
* **`api_url`**: API URL for the OpenGradient API.
* **`contract_address`**: Inference contract address.
* **`og_llm_server_url`**: OpenGradient LLM server URL.
* **`og_llm_streaming_server_url`**: OpenGradient LLM streaming server URL.

#### Methods

---

#### `close()`

```python
def close(self) ‑> None
```
Close underlying SDK resources.

#### Variables

* [**`alpha`**](./alpha): Alpha Testnet features including on-chain inference, workflow management, and ML model execution.
* [**`llm`**](./llm): LLM chat and completion via TEE-verified execution.
* [**`model_hub`**](./model_hub): Model Hub for creating, versioning, and uploading ML models.
* [**`twins`**](./twins): Digital twins chat via OpenGradient verifiable inference. ``None`` when no ``twins_api_key`` is provided.