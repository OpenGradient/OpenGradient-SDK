# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenGradient Python SDK - A decentralized model management and inference platform SDK. The SDK enables programmatic access to model repositories and decentralized AI infrastructure, including end-to-end verified AI execution. Use virtualenv for dependency management locally (in `venv` folder).

## Development Commands

### Build & Installation
```bash
# Install in development mode
pip install -e .

# Build distribution
make build

# Publish to PyPI (requires credentials)
make publish
```

### Testing
```bash
# Run all tests
make test

# Run unit tests only
make utils_test

# Run integration tests
make integrationtest
```

### Code Quality
```bash
# Run formatting (ruff, line-length=140) and type checking (mypy)
make check

# Generate documentation
make docs
```

### CLI Testing
```bash
# Test model inference
make infer

# Test LLM completion
make completion

# Test chat functionality
make chat

# Test streaming chat
make chat-stream

# Test tool calling
make chat-tool
```

## Architecture Overview

### Core Components

1. **Client (`src/opengradient/client/`)**: Client package with submodules
   - `client.py` — Main Client class, authentication, initialization
   - `llm.py` — LLM chat/completion with TEE and x402 payment support
   - `model_hub.py` — Model repository management (CRUD, upload)
   - `twins.py` — Digital twins chat integration (twin.fun)
   - `alpha.py` — Alpha Testnet on-chain inference and workflows
   - `opg_token.py` — OPG token Permit2 approval
   - `exceptions.py` — Custom exception classes

2. **CLI (`src/opengradient/cli.py`)**: Command-line interface using Click
   - Commands: `config`, `infer`, `completion`, `chat`
   - Supports file-based input for messages and tools

3. **Agents (`src/opengradient/agents/`)**: Framework integrations
   - `og_langchain.py` — LangChain ChatModel adapter

4. **Blockchain Integration**: 
   - Smart contract ABIs in `src/opengradient/abi/`
   - Web3 integration for decentralized inference
   - Support for TEE (Trusted Execution Environment) mode

5. **Protocol Buffers (`src/opengradient/proto/`)**: gRPC service definitions for inference

6. **AlphaSense (`src/opengradient/alphasense/`)**: LangChain-compatible tools for AI agents
   - `run_model_tool.py` — Tool for running on-chain model inference
   - `read_workflow_tool.py` — Tool for reading workflow results

7. **Workflow Models (`src/opengradient/workflow_models/`)**: Hardcoded price/volatility forecast models for on-chain execution

### Key Concepts

- **Inference Modes**: VANILLA, TEE (end-to-end verified), ZKML
- **Model CID**: Content-addressed model identifiers
- **Dual Authentication**: Model Hub (email/password) + blockchain (private key)
- **x402 Payments**: Streamed micropayment protocol for LLM inference
- **Digital Twins**: Chat integration with twin.fun personas
- **Dual Chain**: Base Sepolia (LLM) + OpenGradient testnet (Alpha on-chain inference)

## Configuration

The SDK uses environment-based configuration with defaults:
- `DEFAULT_RPC_URL`: Blockchain RPC endpoint
- `DEFAULT_API_URL`: Model Hub API endpoint
- `DEFAULT_INFERENCE_CONTRACT_ADDRESS`: Smart contract address

User configuration stored via `opengradient config init` wizard.

## Testing Strategy

- Unit tests in `tests/` using pytest: `utils_test.py`, `client_test.py`, `langchain_adapter_test.py`, `opg_token_test.py`
- Integration tests: `integrationtest/` for agent and workflow models
- Stress tests: `stresstest/` for load testing LLM and inference endpoints
- CLI command testing via Makefile targets
- Run individual suites: `make client_test`, `make langchain_adapter_test`, `make opg_token_test`

## Documentation (pdoc)

Docs are generated with `pdoc3` using a custom Mako template at `templates/text.mako`. Run `make docs` to regenerate into `docs/`. Do not edit generated documentation files in `docs/` by hand.

There are concrete example scripts using the SDK in the `examples/` folder that highlight how to use the SDK and provides a starting point for developers. Make sure all example files are referenced in the readme in the folder.

There are more detailed walkthroughs and tutorials in the `tutorials/` folder.

### Cross-referencing in docstrings

To link to another module or class from a docstring, use fully qualified names in backticks:

```python
"""
- **`opengradient.client.llm`** -- description here
- **`opengradient.client.onchain_inference`** -- description here
"""
```

The template's `linkify` function automatically converts `` `opengradient.x.y.z` `` references into relative markdown links. Always use fully qualified names starting with `opengradient.` — never hardcode relative URLs in docstrings.

### Docstring style

- Use Google-style docstrings (Args, Returns, Raises, Note, Attributes sections)
- The template parses these sections and renders them as structured markdown
- For classes, Args in the `__init__` docstring are rendered under the Constructor heading

## Dependencies

Key dependencies (Python >=3.11):
- `web3` & `eth-account`: Blockchain interaction
- `langchain` & `openai`: LLM integrations
- `click`: CLI framework
- `firebase-rest-api`: Backend services
- `pydantic`: Data validation
- `og-x402`: x402 payment protocol
- `numpy`: Array handling