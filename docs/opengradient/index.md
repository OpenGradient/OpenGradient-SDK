---
outline: [2,3]
---

opengradient

# Package opengradient

**Version: 0.6.0**

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

## Client Namespaces

The [Client](./client/index) object exposes three namespaces:

- **[llm](./client/llm)** -- Verifiable LLM chat and completion via TEE-verified execution with x402 payments
- **[alpha](./client/alpha)** -- On-chain ONNX model inference, workflow deployment, and scheduled ML model execution (only available on the Alpha Testnet)
- **[model_hub](./client/model_hub)** -- Model repository management

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

## Submodules

* [**agents**](./agents/index): OpenGradient Agent Framework Adapters
* [**alphasense**](./alphasense/index): OpenGradient AlphaSense Tools
* [**client**](./client/index): OpenGradient Client -- the central entry point to all SDK services.
* [**types**](./types): OpenGradient Specific Types
* [**workflow_models**](./workflow_models/index): OpenGradient Hardcoded Models

## Functions

---

### `init()`

```python
def init(private_key: str, email: Optional[str] = None, password: Optional[str] = None, **kwargs) ‑> `Client`
```
Initialize the global OpenGradient client.

This is the recommended way to get started. It creates a `Client` instance
and stores it as the global client for convenience.

**Arguments**

* **`private_key`**: Private key for OpenGradient transactions.
* **`email`**: Email for Model Hub authentication. Optional.
* **`password`**: Password for Model Hub authentication. Optional.
    **kwargs: Additional arguments forwarded to `Client`.

**Returns**

The newly created `Client` instance.

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

* [**`alpha`**](./client/alpha): Alpha Testnet features including on-chain inference, workflow management, and ML model execution.
* [**`llm`**](./client/llm): LLM chat and completion via TEE-verified execution.
* [**`model_hub`**](./client/model_hub): Model Hub for creating, versioning, and uploading ML models.

### `InferenceMode`

Enum for the different inference modes available for inference (VANILLA, ZKML, TEE)

#### Variables

* static `TEE`
* static `VANILLA`
* static `ZKML`

### `PaymentNetwork`

Payment network for x402 payment protocol transactions.

Controls which blockchain network is used for paying for LLM inference.

**Attributes**

* **`OG_EVM`**: Pay on the OpenGradient EVM network (default).
* **`BASE_SEPOLIA`**: Pay on the Base Sepolia testnet using the OG token
        at address ``0x240b09731D96979f50B2C649C9CE10FcF9C7987F``.

#### Variables

* static `BASE_SEPOLIA`
* static `OG_EVM`

### `TEE_LLM`

Enum for LLM models available for TEE (Trusted Execution Environment) execution.

TEE mode provides cryptographic verification that inference was performed
correctly in a secure enclave. Use this for applications requiring
auditability and tamper-proof AI inference.

#### Variables

* static `CLAUDE_3_5_HAIKU`
* static `CLAUDE_3_7_SONNET`
* static `CLAUDE_4_0_SONNET`
* static `GEMINI_2_0_FLASH`
* static `GEMINI_2_5_FLASH`
* static `GEMINI_2_5_FLASH_LITE`
* static `GEMINI_2_5_PRO`
* static `GPT_4O`
* static `GPT_4_1_2025_04_14`
* static `GROK_2_1212`
* static `GROK_2_VISION_LATEST`
* static `GROK_3_BETA`
* static `GROK_3_MINI_BETA`
* static `GROK_4_1_FAST`
* static `GROK_4_1_FAST_NON_REASONING`
* static `O4_MINI`