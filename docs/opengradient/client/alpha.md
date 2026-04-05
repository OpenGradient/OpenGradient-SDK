---
outline: [2,4]
---

B[opengradient](../index) / [client](./index) / alpha

# Package opengradient.client.alpha

Alpha Testnet features for OpenGradient SDK.

This module contains features that are only available on the Alpha Testnet,
including on-chain ONNX model inference, workflow management, and ML model execution.

## Classes

### `Alpha`

Alpha Testnet features namespace.

This class provides access to features that are only available on the Alpha Testnet,
including on-chain ONNX model inference, workflow deployment, and execution.

#### Constructor

```python
def __init__(
    private_key: str,
    rpc_url: str = 'https://ogevmdevnet.opengradient.ai',
    inference_contract_address: str = '0x8383C9bD7462F12Eb996DD02F78234C0421A6FaE',
    api_url: str = 'https://sdk-devnet.opengradient.ai'
)
```

#### Methods

---

#### `infer()`

```python
def infer(
    self,
    model_cid: str,
    inference_mode: `InferenceMode`,
    model_input: Dict[str, Union[str, int, float, List, `ndarray`]],
    max_retries: Optional[int] = None
) ‑> `InferenceResult`
```
Perform inference on a model.

**Arguments**

* **`model_cid (str)`**: The unique content identifier for the model from IPFS.
* **`inference_mode (InferenceMode)`**: The inference mode.
* **`model_input (Dict[str, Union[str, int, float, List, np.ndarray]])`**: The input data for the model.
* **`max_retries (int, optional)`**: Maximum number of retry attempts. Defaults to 5.

**Returns**

InferenceResult (InferenceResult): A dataclass object containing the transaction hash and model output.
    transaction_hash (str): Blockchain hash for the transaction
    model_output (Dict[str, np.ndarray]): Output of the ONNX model

**Raises**

* **`RuntimeError`**: If the inference fails.

---

#### `new_workflow()`

```python
def new_workflow(
    self,
    model_cid: str,
    input_query: `HistoricalInputQuery`,
    input_tensor_name: str,
    scheduler_params: Optional[`SchedulerParams`] = None
) ‑> str
```
Deploy a new workflow contract with the specified parameters.

This function deploys a new workflow contract on OpenGradient that connects
an AI model with its required input data. When executed, the workflow will fetch
the specified model, evaluate the input query to get data, and perform inference.

The workflow can be set to execute manually or automatically via a scheduler.

**Arguments**

* **`model_cid (str)`**: CID of the model to be executed from the Model Hub
* **`input_query (HistoricalInputQuery)`**: Input definition for the model inference,
        will be evaluated at runtime for each inference
* **`input_tensor_name (str)`**: Name of the input tensor expected by the model
* **`scheduler_params (Optional[SchedulerParams])`**: Scheduler configuration for automated execution:
        - frequency: Execution frequency in seconds
        - duration_hours: How long the schedule should live for

**Returns**

str: Deployed contract address. If scheduler_params was provided, the workflow
     will be automatically executed according to the specified schedule.

**Raises**

* **`Exception`**: If transaction fails or gas estimation fails

---

#### `read_workflow_history()`

```python
def read_workflow_history(
    self,
    contract_address: str,
    num_results: int
) ‑> List[`ModelOutput`]
```
Gets historical inference results from a workflow contract.

Retrieves the specified number of most recent inference results from the contract's
storage, with the most recent result first.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract
* **`num_results (int)`**: Number of historical results to retrieve

**Returns**

List[ModelOutput]: List of historical inference results

---

#### `read_workflow_result()`

```python
def read_workflow_result(self, contract_address: str) ‑> `ModelOutput`
```
Reads the latest inference result from a deployed workflow contract.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract

**Returns**

ModelOutput: The inference result from the contract

**Raises**

* **`ContractLogicError`**: If the transaction fails
* **`Web3Error`**: If there are issues with the web3 connection or contract interaction

---

#### `run_workflow()`

```python
def run_workflow(self, contract_address: str) ‑> `ModelOutput`
```
Triggers the run() function on a deployed workflow contract and returns the result.

**Arguments**

* **`contract_address (str)`**: Address of the deployed workflow contract

**Returns**

ModelOutput: The inference result from the contract

**Raises**

* **`ContractLogicError`**: If the transaction fails
* **`Web3Error`**: If there are issues with the web3 connection or contract interaction

#### Variables

* `inference_abi` : dict
* `precompile_abi` : dict





---

## Error Handling

When calling `infer()`, several errors can occur. Here's how to handle them properly:

### Basic Error Handling

```python
from opengradient import Alpha

alpha = Alpha(private_key='your_private_key')

try:
    result = alpha.infer(
        model_cid='QmbUqS93...',
        model_input={"data": "input_value"},
        inference_mode=InferenceMode.VANILLA
    )
    print(f"Inference succeeded: {result.model_output}")
    
except ValueError as e:
    # Invalid model CID or model input
    print(f"Invalid input: {e}")
    
except RuntimeError as e:
    # Inference execution failed
    print(f"Inference failed: {e}")
    
except Exception as e:
    # Network error, timeout, insufficient balance, etc.
    print(f"Unexpected error: {e}")


try:
    result = alpha.infer(
        model_cid='QmbUqS93...',
        model_input={...},
        inference_mode=InferenceMode.VANILLA,
        max_retries=3
    )
    
except ValueError as e:
    print("Invalid model CID or input format")
    # Check that model_cid is a valid IPFS CID
    
except RuntimeError as e:
    if "Insufficient balance" in str(e):
        print("Insufficient wallet balance for inference")
        # Fund your wallet with OPG tokens
    elif "Timeout" in str(e):
        print("Inference timed out - trying again with max_retries")
        # Function will auto-retry up to max_retries times
    else:
        print(f"Inference failed: {e}")
        
except Exception as e:
    print(f"Unexpected error: {e}")


Common Errors and Solutions
Error	Cause	Solution
ValueError: Invalid model CID	Model CID format incorrect	Verify CID is valid IPFS format
RuntimeError: Insufficient balance	Not enough OPG tokens	Fund wallet at opengradient.ai
RuntimeError: Timeout	Model taking too long	Increase max_retries or check TEE status
RuntimeError: Network error	Connection issue	Check internet, retry with backoff
RuntimeError: Inference failed	Generic inference error	Check model_input matches model requirements
