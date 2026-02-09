# OpenGradient Python SDK

A Python SDK for decentralized model management and inference services on the OpenGradient platform. The SDK provides programmatic access to distributed AI infrastructure with cryptographic verification capabilities.

## Overview

OpenGradient enables developers to build AI applications with verifiable execution guarantees through Trusted Execution Environments (TEE) and blockchain-based settlement. The SDK supports standard LLM inference patterns while adding cryptographic attestation for applications requiring auditability and tamper-proof AI execution.

### Key Features

- **Verifiable LLM Inference**: Drop-in replacement for OpenAI and Anthropic APIs with cryptographic attestation
- **Multi-Provider Support**: Access models from OpenAI, Anthropic, Google, and xAI through a unified interface
- **TEE Execution**: Trusted Execution Environment inference with cryptographic verification
- **Model Hub Integration**: Registry for model discovery, versioning, and deployment
- **Consensus-Based Verification**: End-to-end verified AI execution through the OpenGradient network
- **Command-Line Interface**: Direct access to SDK functionality via CLI

## Installation
```bash
pip install opengradient
```

**Note**: Windows users should temporarily enable WSL during installation (fix in progress).

## Network Architecture

OpenGradient operates two networks:

- **Testnet**: Primary public testnet for general development and testing
- **Alpha Testnet**: Experimental features including atomic AI execution from smart contracts and scheduled ML workflow execution

For current network RPC endpoints, contract addresses, and deployment information, refer to the [Network Deployment Documentation](https://docs.opengradient.ai/learn/network/deployment.html).

## Getting Started

### Prerequisites

Before using the SDK, you will need:

1. **Private Key**: An Ethereum-compatible wallet private key for OpenGradient transactions
2. **Test Tokens**: Obtain free test tokens from the [OpenGradient Faucet](https://faucet.opengradient.ai) for testnet LLM inference
3. **Model Hub Account** (Optional): Required only for model uploads. Register at [hub.opengradient.ai/signup](https://hub.opengradient.ai/signup)

### Configuration

Initialize your configuration using the interactive wizard:
```bash
opengradient config init
```

### Client Initialization
```python
import os
import opengradient as og

client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
    email=None,  # Optional: required only for model uploads
    password=None,
)
```

## Core Functionality

### TEE-Secured LLM Chat

OpenGradient provides secure, verifiable inference through Trusted Execution Environments. All supported models include cryptographic attestation verified by the OpenGradient network:
```python
completion = client.llm.chat(
    model=og.TEE_LLM.GPT_4O,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(f"Response: {completion.chat_output['content']}")
print(f"Transaction hash: {completion.transaction_hash}")
```

### Streaming Responses

For real-time generation, enable streaming:
```python
stream = client.llm.chat(
    model=og.TEE_LLM.CLAUDE_3_7_SONNET,
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Verifiable LangChain Integration

Use OpenGradient as a drop-in LLM provider for LangChain agents with network-verified execution:
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import opengradient as og

llm = og.agents.langchain_adapter(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
    model_cid=og.TEE_LLM.GPT_4O,
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
- GPT-4o
- o4-mini

#### Anthropic
- Claude 3.7 Sonnet
- Claude 3.5 Haiku
- Claude 4.0 Sonnet

#### Google
- Gemini 2.5 Flash
- Gemini 2.5 Pro
- Gemini 2.0 Flash
- Gemini 2.5 Flash Lite

#### xAI
- Grok 3 Beta
- Grok 3 Mini Beta
- Grok 2 (1212)
- Grok 2 Vision
- Grok 4.1 Fast (reasoning and non-reasoning)

For a complete list, reference the `og.TEE_LLM` enum or consult the [API documentation](https://docs.opengradient.ai/api_reference/python_sdk/).

## Alpha Testnet Features

The Alpha Testnet provides access to experimental capabilities including custom ML model inference and workflow orchestration. These features enable on-chain AI pipelines that connect models with data sources and support scheduled automated execution.

**Note**: Alpha features require connecting to the Alpha Testnet. See [Network Architecture](#network-architecture) for details.

### Custom Model Inference

Browse models on the [Model Hub](https://hub.opengradient.ai/) or deploy your own:
```python
result = client.alpha.infer(
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

client = og.Client(
    private_key="your-private-key",
    email="your-email",
    password="your-password",
)

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
contract_address = client.alpha.new_workflow(
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
result = client.alpha.run_workflow(contract_address)
print(f"Inference output: {result}")

# Read the latest result
latest = client.alpha.read_workflow_result(contract_address)

# Retrieve historical results
history = client.alpha.read_workflow_history(
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
opengradient chat --model anthropic/claude-3.5-haiku \
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

- **SETTLE**: Records cryptographic hashes only (maximum privacy)
- **SETTLE_METADATA**: Records complete input/output data (maximum transparency)
- **SETTLE_BATCH**: Aggregates multiple inferences (most cost-efficient)

Specify settlement mode in your requests:
```python
result = client.llm.chat(
    model=og.TEE_LLM.GPT_4O,
    messages=[{"role": "user", "content": "Hello"}],
    x402_settlement_mode=og.x402SettlementMode.SETTLE_BATCH,
)
```

### Payment Network Selection

By default, payments settle on the OG EVM network. To pay via Base Sepolia instead, set the `payment_network` parameter at client initialization. The Base Sepolia payment network uses the OG token at address `0x240b09731D96979f50B2C649C9CE10FcF9C7987F`.
```python
client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
    payment_network=og.PaymentNetwork.BASE_SEPOLIA,
)
```

## Examples

Additional code examples are available in the [examples](./examples) directory.

## Documentation

For comprehensive documentation, API reference, and guides:

- [OpenGradient Documentation](https://docs.opengradient.ai/)
- [API Reference](https://docs.opengradient.ai/api_reference/python_sdk/)
- [Network Deployment](https://docs.opengradient.ai/learn/network/deployment.html)

### Claude Code Integration

If you use [Claude Code](https://claude.ai/code), copy [docs/CLAUDE_SDK_USERS.md](docs/CLAUDE_SDK_USERS.md) to your project's `CLAUDE.md` to enable context-aware assistance with OpenGradient SDK development.

## Model Hub

Browse and discover AI models on the [OpenGradient Model Hub](https://hub.opengradient.ai/). The Hub provides:

- Comprehensive model registry with versioning
- Model discovery and deployment tools
- Direct SDK integration for seamless workflows

## Support

- Execute `opengradient --help` for CLI command reference
- Visit our [documentation](https://docs.opengradient.ai/) for detailed guides
- Join our [community](https://opengradient.ai/) for support and discussions
