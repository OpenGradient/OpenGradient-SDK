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