# Alpha Testnet Examples

These examples demonstrate features that are currently only available on the **OpenGradient Alpha Testnet**. They are not yet supported on the official testnet.

## Prerequisites

1. **Alpha Testnet Access**: You must be connected to the OpenGradient Alpha Testnet
2. **SDK Installation**: `pip install opengradient`
3. **Credentials**: Set up your environment variables:
   - `OG_PRIVATE_KEY`: Private key funded with **OpenGradient testnet gas tokens** for on-chain inference

## Examples

### `run_inference.py`
Runs inference on a custom model using the OpenGradient network.

```bash
python examples/alpha/run_inference.py
```

**What it does:**
- Executes inference on a model using its CID
- Demonstrates passing structured input data (e.g., OHLC price data)
- Returns model predictions along with the transaction hash

### `run_embeddings_model.py`
Runs inference on an embeddings model for semantic search.

```bash
python examples/alpha/run_embeddings_model.py
```

**What it does:**
- Generates embeddings for queries and passages
- Demonstrates multilingual embeddings models
- Useful for semantic search, retrieval-augmented generation (RAG), etc.

### `create_workflow.py`
Creates a new scheduled workflow for automated model inference.

```bash
python examples/alpha/create_workflow.py
```

**What it does:**
- Defines a workflow that runs a model on a schedule
- Configures historical data queries (e.g., cryptocurrency price data)
- Deploys the workflow as a smart contract
- Returns the contract address for the workflow

### `use_workflow.py`
Reads results from a deployed workflow.

```bash
python examples/alpha/use_workflow.py
```

**What it does:**
- Retrieves the latest prediction from a workflow contract
- Fetches historical predictions from the workflow
- Demonstrates how to consume workflow outputs
