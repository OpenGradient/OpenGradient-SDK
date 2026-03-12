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
    inference_contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE',
    llm_server_url: Optional[str] = None,
    tee_registry_address: str = '0x4e72238852f3c918f4E4e57AeC9280dDB0c80248'
)
```

**Arguments**

* **`private_key`**: Private key whose wallet holds **Base Sepolia OPG tokens**
        for x402 LLM payments.
* **`alpha_private_key`**: Private key whose wallet holds **OpenGradient testnet
        gas tokens** for on-chain inference. Optional -- falls back to
        ``private_key`` for backward compatibility.
* **`email`**: Email for Model Hub authentication. Must be provided together
        with ``password``.
* **`password`**: Password for Model Hub authentication. Must be provided
        together with ``email``.
* **`twins_api_key`**: API key for digital twins chat (twin.fun). Optional.
* **`rpc_url`**: RPC URL for the OpenGradient Alpha Testnet.
* **`api_url`**: API URL for the OpenGradient API.
* **`inference_contract_address`**: Inference contract address on the
        OpenGradient Alpha Testnet.
* **`llm_server_url`**: Override the LLM server URL instead of using the
        registry-discovered endpoint. When set, the TLS certificate is
        validated against the system CA bundle rather than the registry.
* **`tee_registry_address`**: Address of the TEERegistry contract used to
        discover active LLM proxy endpoints and their verified TLS certs.

#### Methods

---

#### `close()`

```python
async def close(self) ‑> None
```
Close underlying SDK resources.

#### Variables

* [**`alpha`**](./alpha): Alpha Testnet features including on-chain inference, workflow management, and ML model execution.
* [**`llm`**](./llm): LLM chat and completion via TEE-verified execution.
* [**`model_hub`**](./model_hub): Model Hub for creating, versioning, and uploading ML models.
* [**`twins`**](./twins): Digital twins chat via OpenGradient verifiable inference. ``None`` when no ``twins_api_key`` is provided.