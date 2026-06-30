# OpenGradient Python SDK

[![Checks](https://github.com/OpenGradient/sdk/actions/workflows/test.yml/badge.svg)](https://github.com/OpenGradient/sdk/actions/workflows/test.yml)
[![E2E Tests](https://github.com/OpenGradient/sdk/actions/workflows/e2e.yml/badge.svg)](https://github.com/OpenGradient/sdk/actions/workflows/e2e.yml)
[![PyPI](https://img.shields.io/pypi/v/opengradient)](https://pypi.org/project/opengradient/)

A Python SDK for decentralized model management and inference services on the OpenGradient platform. The SDK provides programmatic access to distributed AI infrastructure with cryptographic verification capabilities.

## Quick Summary for Developers

> **New to OpenGradient?** Start here.

| Question | Answer |
|---|---|
| **What is it?** | A decentralized network that runs AI inference inside TEEs and settles every request on-chain |
| **What problem does it solve?** | Centralized AI is a black box. OpenGradient gives cryptographic proof for every inference |
| **How do I use it?** | Install the SDK, get a private key, call llm.chat() like OpenAI but with transaction_hash and tee_signature in every response |
| **What is Model Hub?** | A decentralized registry to upload, discover, and run custom ONNX models on-chain |
| **What is MemSync?** | A long-term memory layer for AI agents with persistent context across sessions |

### 30-Second Quickstart

Install, set up a private key with OPG tokens, and run:

    import asyncio, os, opengradient as og

    async def main():
        llm = og.LLM(private_key=os.environ["OG_PRIVATE_KEY"])
        llm.ensure_opg_approval(min_allowance=0.1)
        result = await llm.chat(
            model=og.TEE_LLM.GEMINI_2_5_FLASH,
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(result.chat_output["content"])  # AI response
        print(result.transaction_hash)         # on-chain proof

    asyncio.run(main())

---

## Overview

OpenGradient enables developers to build AI applications with verifiable execution guarantees through Trusted Execution Environments (TEE) and blockchain-based settlement. The SDK supports standard LLM inference patterns while adding cryptographic attestation for applications requiring auditability and tamper-proof AI execution.

### Key Features

- **Verifiable LLM Inference**: Drop-in replacement for OpenAI and Anthropic APIs with cryptographic attestation
- **Multi-Provider Support**: Access models from OpenAI, Anthropic, Google, and xAI through a unified interface
- **Native Web Search**: Opt-in `web_search` flag enables each provider's built-in web search, billed per search
- **Image Generation**: Native image-output models ("nano banana") return generated images directly on the response
- **TEE Execution**: Trusted Execution Environment inference with cryptographic verification
- **Model Hub Integration**: Registry for model discovery, versioning, and deployment
- **Consensus-Based Verification**: End-to-end verified AI execution through the OpenGradient network
- **Command-Line Interface**: Direct access to SDK functionality via CLI

## Installation
```bash
pip install opengradient
```

> **Note for Windows users:** See the [Windows Installation Guide](./WINDOWS_INSTALL.md) for step-by-step setup instructions.

### Claude Code Integration

If you use [Claude Code](https://claude.ai/code), you can enhance your development experience with OpenGradient:

- **CLAUDE.md context file**: Copy [docs/CLAUDE_SDK_USERS.md](docs/CLAUDE_SDK_USERS.md) to your project's `CLAUDE.md` to enable context-aware assistance with OpenGradient SDK development.

- **Claude Code Plugin**: Install the [OpenGradient Claude Plugin](https://github.com/OpenGradient/claude-plugins) for skills, commands, and agents tailored to OpenGradient development. To install, run:
  ```bash
  claude plugin marketplace add https://github.com/OpenGradient/claude-plugins
  ```

## Network Architecture

OpenGradient operates two networks:

- **Testnet**: Primary public testnet for general development and testing
- **Alpha Testnet**: Experimental features including atomic AI execution from smart contracts and scheduled ML workflow execution

For current network RPC endpoints, contract addresses, and deployment information, refer to the [Network Deployment Documentation](https://docs.opengradient.ai/learn/network/deployment.html).

## Trust Model

OpenGradient's verifiable inference relies on three distinct layers. Understanding which layer enforces what makes it easier to reason about what an SDK response does and does not guarantee.

1. **TEE attestation (network-side, at registration).** Each TEE proves its identity and code measurement (PCR hash) to the OpenGradient network when it registers in the on-chain `TEERegistry` contract. The network verifies the attestation and stores the TEE's public key, PCR hash, and TLS certificate on-chain. SDK clients do not re-run attestation themselves.

2. **TLS certificate pinning (SDK-side, at request time).** When the SDK resolves a TEE through `RegistryTEEConnection` (the production path), it reads the registered TLS certificate directly from the registry and pins it as the *only* trust anchor for the connection (`CERT_REQUIRED`, no system-CA fallback). A successful response is therefore, by TLS construction, from a TEE the network has already attested. This is the live trust binding for SDK responses.

3. **Signature verification (settlement-side, on-chain).** The `tee_signature` and `tee_timestamp` returned with each response are durable proof material. Signature verification happens during on-chain settlement, not in the SDK at return time. These fields are also useful for offline / auditor verification when a response leaves the original TLS session.

> **Note:** The SDK does **not** verify `tee_signature` client-side at return time. A non-erroring response means the TLS-pinned channel was honored; signature checking lives at the settlement layer. If you archive a response and want to re-verify it offline, use the registry's stored public key for that TEE.
>
> The `StaticTEEConnection` mode (used for self-hosted dev with `verify=False`) bypasses both the registry and TLS pinning, and is **not a production trust path**.

## Getting Started

### Prerequisites

Before using the SDK, you will need:

1. **Private Key**: An Ethereum-compatible wallet private key funded with **Base OPG tokens** for x402 LLM payments
2. **Alpha Testnet Key** (Optional): A private key funded with **OpenGradient testnet gas tokens** for Alpha Testnet on-chain inference (can be the same or a different key)
3. **Model Hub Account** (Optional): Required only for model uploads. Register at [hub.opengradient.ai/signup](https://hub.opengradient.ai/signup)

### Configuration

Initialize your configuration using the interactive wizard:
```bash
opengradient config init
```

### Environment Variables

The SDK accepts configuration through environment variables, though most parameters (like `private_key`) are passed directly to the client.

The following Firebase configuration variables are **optional** and only needed for Model Hub operations (uploading/managing models):

- `FIREBASE_API_KEY`
- `FIREBASE_AUTH_DOMAIN`
- `FIREBASE_PROJECT_ID`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_APP_ID`
- `FIREBASE_DATABASE_URL`

**Note**: If you're only using the SDK for LLM inference, you don't need to configure any environment variables.

### Initialization

The SDK provides separate clients for each service. Create only the ones you need:

```python
import os
import opengradient as og

# LLM inference — settles via x402 on Base using OPG tokens
llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))

# Alpha Testnet — on-chain inference on the OpenGradient network using testnet gas tokens
alpha = og.Alpha(private_key=os.environ.get("OG_PRIVATE_KEY"))

# Model Hub — requires email/password, only needed for model uploads
hub = og.ModelHub(email="you@example.com", password="...")
```

### OPG Token Approval

Before making LLM requests, your wallet must approve OPG token spending via the [Permit2](https://github.com/Uniswap/permit2) protocol. This only sends an on-chain transaction when the current allowance drops below the threshold:

```python
llm.ensure_opg_approval(min_allowance=5)
```

See [Payment Settlement](#payment-settlement) for details on settlement modes.

## Core Functionality

### TEE-Secured LLM Chat

OpenGradient provides secure, verifiable inference through Trusted Execution Environments. All supported models run in TEEs whose attestation is verified by the OpenGradient network at registration; the SDK binds each request to an attested TEE via registry-pinned TLS, and signed proof material is verified at on-chain settlement (see [Trust Model](#trust-model)). LLM methods are async:
```python
completion = await llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(f"Response: {completion.chat_output['content']}")
print(f"Transaction hash: {completion.transaction_hash}")
```

### Streaming Responses

For real-time generation, enable streaming:
```python
stream = await llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500,
    stream=True,
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Native Web Search

Set `web_search=True` to let the model search the web while answering. Each search is billed per search on top of token usage, at the provider's list price. Supported by OpenAI, Anthropic, Google, and xAI models; other providers ignore the flag.
```python
completion = await llm.chat(
    model=og.TEE_LLM.CLAUDE_SONNET_4_6,
    messages=[{"role": "user", "content": "What are today's top tech headlines?"}],
    max_tokens=500,
    web_search=True,
)
print(completion.chat_output["content"])
```

### Image Generation

Native image-output models ("nano banana") return generated images on the response. The generated images are available in `result.images` as `data:` URIs, while any text caption is in `chat_output["content"]`. Images travel out-of-band and are not part of the signed output hash.
```python
import base64

result = await llm.chat(
    model=og.TEE_LLM.GEMINI_3_1_FLASH_IMAGE,
    messages=[{"role": "user", "content": "A friendly robot reading under a tree"}],
    max_tokens=1024,
)

for i, image in enumerate(result.images or []):
    payload = image.split(",", 1)[1]  # strip the "data:image/png;base64," prefix
    with open(f"image_{i}.png", "wb") as f:
        f.write(base64.b64decode(payload))
```

### Verifiable LangChain Integration

Use OpenGradient as a drop-in LLM provider for LangChain agents with network-verified execution:
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import opengradient as og

llm = og.agents.langchain_adapter(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
    model_cid=og.TEE_LLM.GPT_5,
)

@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"Sunny, 72°F in {city}"

agent = create_react_agent(llm, [get_weather])
result = agent.invoke({
    "messages": [("user", "What's the weather in San Francisco?")]
})
print(result["messages"][-1].content)
```

### Available Models

The SDK provides access to models from multiple providers via the `og.TEE_LLM` enum:

#### OpenAI
- GPT-4.1 (2025-04-14)
- o4-mini
- GPT-5
- GPT-5 Mini
- GPT-5.2
- GPT Image 2 (native image generation)

#### Anthropic
- Claude Sonnet 4.5
- Claude Sonnet 4.6
- Claude Haiku 4.5
- Claude Opus 4.5
- Claude Opus 4.6
- Claude Opus 4.7
- Claude Opus 4.8

#### Google
- Gemini 2.5 Flash
- Gemini 2.5 Pro
- Gemini 2.5 Flash Lite
- Gemini 3 Pro
- Gemini 3 Flash
- Gemini 3.5 Flash
- Gemini 2.5 Flash Image (native image generation, "nano banana")
- Gemini 3.1 Flash Image (native image generation, "nano banana 2")

#### xAI
- Grok 4
- Grok 4 Fast
- Grok 4.1 Fast (reasoning and non-reasoning)

#### Nous Research
- Hermes 4 405B
- Hermes 4 70B

For a complete list, reference the `og.TEE_LLM` enum or consult the [API documentation](https://docs.opengradient.ai/api_reference/python_sdk/).

## Alpha Testnet Features

The Alpha Testnet provides access to experimental capabilities including custom ML model inference and workflow orchestration. These features enable on-chain AI pipelines that connect models with data sources and support scheduled automated execution.

**Note**: Alpha features require connecting to the Alpha Testnet. See [Network Architecture](#network-architecture) for details.

### Custom Model Inference

Browse models on the [Model Hub](https://hub.opengradient.ai/) or deploy your own:
```python
result = alpha.infer(
    model_cid="your-model-cid",
    model_input={"input": [1.0, 2.0, 3.0]},
    inference_mode=og.InferenceMode.VANILLA,
)
print(f"Output: {result.model_output}")
```

### Workflow Deployment

Deploy on-chain AI workflows with optional scheduling:
```python
import opengradient as og

alpha = og.Alpha(private_key="your-private-key")

# Define input query for historical price data
input_query = og.HistoricalInputQuery(
    base="ETH",
    quote="USD",
    total_candles=10,
    candle_duration_in_mins=60,
    order=og.CandleOrder.DESCENDING,
    candle_types=[og.CandleType.CLOSE],
)

# Deploy workflow with optional scheduling
contract_address = alpha.new_workflow(
    model_cid="your-model-cid",
    input_query=input_query,
    input_tensor_name="input",
    scheduler_params=og.SchedulerParams(
        frequency=3600,
        duration_hours=24
    ),  # Optional
)
print(f"Workflow deployed at: {contract_address}")
```

### Workflow Execution and Monitoring
```python
# Manually trigger workflow execution
result = alpha.run_workflow(contract_address)
print(f"Inference output: {result}")

# Read the latest result
latest = alpha.read_workflow_result(contract_address)

# Retrieve historical results
history = alpha.read_workflow_history(
    contract_address,
    num_results=5
)
```

## Command-Line Interface

The SDK includes a comprehensive CLI for direct operations. Verify your configuration:
```bash
opengradient config show
```

Execute a test inference:
```bash
opengradient infer -m QmbUqS93oc4JTLMHwpVxsE39mhNxy6hpf6Py3r9oANr8aZ \
    --input '{"num_input1":[1.0, 2.0, 3.0], "num_input2":10}'
```

Run a chat completion:
```bash
opengradient chat --model anthropic/claude-haiku-4-5 \
    --messages '[{"role":"user","content":"Hello"}]' \
    --max-tokens 100
```

For a complete list of CLI commands:
```bash
opengradient --help
```

## Use Cases

### Decentralized AI Applications
Use OpenGradient as a decentralized alternative to centralized AI providers, eliminating single points of failure and vendor lock-in.

### Verifiable AI Execution
Leverage TEE inference for cryptographically attested AI outputs, enabling trustless AI applications where execution integrity must be proven.

### Auditability and Compliance
Build applications requiring complete audit trails of AI decisions with cryptographic verification of model inputs, outputs, and execution environments.

### Model Hosting and Distribution
Manage, host, and execute models through the Model Hub with direct integration into development workflows.

## Payment Settlement

OpenGradient supports multiple settlement modes through the x402 payment protocol:

- **PRIVATE**: Payment only, no input/output data on-chain (maximum privacy)
- **BATCH_HASHED**: Aggregates inferences into a Merkle tree with input/output hashes and signatures (most cost-efficient, default)
- **INDIVIDUAL_FULL**: Records input, output, timestamp, and verification on-chain (maximum auditability)

Specify settlement mode in your requests:
```python
result = await llm.chat(
    model=og.TEE_LLM.GPT_5,
    messages=[{"role": "user", "content": "Hello"}],
    x402_settlement_mode=og.x402SettlementMode.BATCH_HASHED,
)
```

## Examples

Additional code examples are available in the [examples](./examples) directory.

## Tutorials

Step-by-step guides for building with OpenGradient are available in the [tutorials](./tutorials) directory:

1. **[Build a Verifiable AI Agent with On-Chain Tools](./tutorials/01-verifiable-ai-agent.md)** — Create an AI agent with cryptographically attested execution and on-chain tool integration
2. **[Streaming Multi-Provider Chat with Settlement Modes](./tutorials/02-streaming-multi-provider.md)** — Use a unified API across OpenAI, Anthropic, and Google with real-time streaming and configurable settlement
3. **[Tool-Calling Agent with Verified Reasoning](./tutorials/03-verified-tool-calling.md)** — Build a tool-calling agent where every reasoning step is cryptographically verifiable

## Documentation

For comprehensive documentation, API reference, and guides:

- [OpenGradient Documentation](https://docs.opengradient.ai/)
- [API Reference](https://docs.opengradient.ai/api_reference/python_sdk/)
- [Network Deployment](https://docs.opengradient.ai/learn/network/deployment.html)

## Model Hub

Browse and discover AI models on the [OpenGradient Model Hub](https://hub.opengradient.ai/). The Hub provides:

- Comprehensive model registry with versioning
- Model discovery and deployment tools
- Direct SDK integration for seamless workflows

## Support

- Visit our [documentation](https://docs.opengradient.ai/) for detailed guides
- Join our [community](https://discord.gg/axammqTRDz) for support and discussions
