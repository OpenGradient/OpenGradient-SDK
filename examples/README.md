# OpenGradient SDK Examples

This directory contains example scripts demonstrating how to use the OpenGradient Python SDK for various use cases, from basic model operations to advanced agent integrations.

## Prerequisites

Before running any examples, ensure you have:

1. **Installed the SDK**: `pip install opengradient`
2. **Set up your credentials**: Configure your OpenGradient account using environment variables:
   - `OG_PRIVATE_KEY`: Private key funded with **Base Sepolia OPG tokens** for x402 LLM payments (can be obtained from our [faucet](https://faucet.opengradient.ai)). Also used for Alpha Testnet on-chain inference (requires **OpenGradient testnet gas tokens**).
   - `OG_MODEL_HUB_EMAIL`: (Optional) Your Model Hub email for model uploads
   - `OG_MODEL_HUB_PASSWORD`: (Optional) Your Model Hub password for model uploads

You can also use the configuration wizard:
```bash
opengradient config init
```

## Basic Usage Examples

### Model Management

#### `create_model_repo.py`
Creates a new model repository on the Model Hub.

```bash
python examples/create_model_repo.py
```

**What it does:**
- Creates a new model repository with a name and description
- Returns a model repository object with version information

#### `upload_model_to_hub.py`
Uploads a model file to an existing model repository.

```bash
python examples/upload_model_to_hub.py
```

**What it does:**
- Creates a model repository (or uses an existing one)
- Uploads an ONNX model file to the repository
- Returns the model CID (Content Identifier) for use in inference

**Note:** Requires Model Hub credentials (`OG_MODEL_HUB_EMAIL` and `OG_MODEL_HUB_PASSWORD`).

## LLM Examples

#### `llm_chat.py`
Runs a basic LLM chat completion.

```bash
python examples/llm_chat.py
```

**What it does:**
- Sends a multi-turn conversation to an LLM
- Uses x402 protocol for payment processing
- Returns the model's response

#### `llm_chat_streaming.py`
Runs a streaming LLM chat completion.

```bash
python examples/llm_chat_streaming.py
```

**What it does:**
- Sends a multi-turn conversation to an LLM with streaming enabled
- Demonstrates real-time token streaming
- Returns chunks as they arrive from the model

#### `llm_tool_calling.py`
Demonstrates LLM tool/function calling.

```bash
python examples/llm_tool_calling.py
```

**What it does:**
- Defines a tool (weather lookup) and passes it to the LLM
- The model decides when to invoke tools based on the user's query
- Uses x402 protocol for payment processing

## Alpha Testnet Examples

Examples for features only available on the **Alpha Testnet** are located in the [`alpha/`](./alpha/) folder. These include:

- Model inference (`run_inference.py`)
- Embeddings models (`run_embeddings_model.py`)
- Workflow creation and usage (`create_workflow.py`, `use_workflow.py`)

See [`alpha/README.md`](./alpha/README.md) for details.

## LangChain Agent Example

#### `langchain_react_agent.py`
Creates a basic LangChain ReAct agent powered by an OpenGradient LLM.

```bash
python examples/langchain_react_agent.py
```

**What it does:**
- Uses `og.agents.langchain_adapter` to create a LangChain-compatible LLM
- Sets up a LangGraph ReAct agent with a custom tool
- Demonstrates tool calling via x402 payment processing

**Example use case:** Building conversational agents with tool access, task automation.

## Twins Chat Example

#### `twins_chat.py`
Chat with digital twins from [twin.fun](https://twin.fun) via OpenGradient verifiable inference.

```bash
python examples/twins_chat.py
```

**What it does:**
- Connects to the Twins API to chat with digital twins of public figures
- Demonstrates multi-turn conversations with different twin personas
- Uses TEE-based verifiable inference for trustworthy responses

**Required environment variables:**
- `OG_PRIVATE_KEY`: Your Ethereum private key
- `TWINS_API_KEY`: Your Twins API key

**Example use case:** Building applications that interact with digital twin personas through verified AI inference.

## Common Patterns

### Initialization

Each sub-client is created independently with the credentials it needs:

```python
import os
import opengradient as og

# LLM inference (Base Sepolia OPG tokens for x402 payments)
llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))

# On-chain model inference (OpenGradient testnet gas tokens)
alpha = og.Alpha(private_key=os.environ.get("OG_PRIVATE_KEY"))

# Model Hub (email/password auth)
hub = og.ModelHub(
    email=os.environ.get("OG_MODEL_HUB_EMAIL"),
    password=os.environ.get("OG_MODEL_HUB_PASSWORD"),
)
```

### Running Inference

Basic inference pattern:

```python
alpha = og.Alpha(private_key=os.environ.get("OG_PRIVATE_KEY"))

result = alpha.infer(
    model_cid="your-model-cid",
    model_input={"input_key": "input_value"},
    inference_mode=og.InferenceMode.VANILLA
)
print(f"Output: {result.model_output}")
print(f"Tx hash: {result.transaction_hash}")
```

### LLM Chat

LLM chat methods are async. The default client auto-discovers active TEE endpoints from the on-chain registry:

```python
llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))

completion = await llm.chat(
    model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
    messages=[{"role": "user", "content": "Your message"}],
)
print(f"Response: {completion.chat_output['content']}")
```

For development or self-hosted TEE servers, use a custom endpoint:

```python
llm = og.LLM.from_url(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
    llm_server_url="https://your-tee-server.local",
)
```

## Finding Model CIDs

Browse available models on the [OpenGradient Model Hub](https://hub.opengradient.ai/). Each model has a CID that you can use in your code.

## Additional Resources

- [OpenGradient Documentation](https://docs.opengradient.ai/)
- [API Reference](https://docs.opengradient.ai/api_reference/python_sdk/)
- [Model Hub](https://hub.opengradient.ai/)

## Getting Help

- Run `opengradient --help` for CLI command reference
- Visit our [documentation](https://docs.opengradient.ai/) for detailed guides
- Check the main [README](../README.md) for SDK overview
