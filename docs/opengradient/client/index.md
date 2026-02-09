---
outline: [2,3]
---

[opengradient](../index) / client

# Package opengradient.client

OpenGradient Client -- the central entry point to all SDK services.

## Overview

The [Client](./client) class provides unified access to three service namespaces:

- **[llm](./llm)** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement
- **[model_hub](./model_hub)** -- Model repository management: create, version, and upload ML models
- **[alpha](./alpha)** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution

## Usage

```python
import opengradient as og

client = og.init(private_key="0x...")

# LLM chat (TEE-verified, streamed)
for chunk in client.llm.chat(
    model=og.TEE_LLM.CLAUDE_3_5_HAIKU,
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

## Classes

### `Client`

Main OpenGradient SDK client.

Provides unified access to all OpenGradient services including LLM inference,
on-chain model inference, and the Model Hub. Handles authentication via
blockchain private key and optional Model Hub credentials.

#### Constructor

```python
def __init__(private_key: str, email: Optional[str] = None, password: Optional[str] = None, rpc_url: str = 'https://ogevmdevnet.opengradient.ai', api_url: str = 'https://sdk-devnet.opengradient.ai', contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE', og_llm_server_url: Optional[str] = 'https://llmogevm.opengradient.ai', og_llm_streaming_server_url: Optional[str] = 'https://llmogevm.opengradient.ai', payment_network: `PaymentNetwork` = PaymentNetwork.OG_EVM)
```

**Arguments**

* **`private_key`**: Private key for OpenGradient transactions.
* **`email`**: Email for Model Hub authentication. Optional.
* **`password`**: Password for Model Hub authentication. Optional.
* **`rpc_url`**: RPC URL for the blockchain network.
* **`api_url`**: API URL for the OpenGradient API.
* **`contract_address`**: Inference contract address.
* **`og_llm_server_url`**: OpenGradient LLM server URL.
* **`og_llm_streaming_server_url`**: OpenGradient LLM streaming server URL.
* **`payment_network`**: Payment network for x402 transactions. Defaults to OG_EVM.

#### Variables

* [**`alpha`**](./alpha): Alpha Testnet features including on-chain inference, workflow management, and ML model execution.
* [**`llm`**](./llm): LLM chat and completion via TEE-verified execution.
* [**`model_hub`**](./model_hub): Model Hub for creating, versioning, and uploading ML models.