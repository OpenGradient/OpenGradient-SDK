---
outline: [2,3]
---

[opengradient](../index) / [client](./index) / client

# Package opengradient.client.client

Main Client class that unifies all OpenGradient service namespaces.

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