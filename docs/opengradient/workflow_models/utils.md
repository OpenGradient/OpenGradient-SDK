---
outline: [2,4]
---

[opengradient](../index) / [workflow_models](./index) / utils

# Package opengradient.workflow_models.utils

Utility functions for the models module.

## Functions

---

### `create_block_explorer_link_smart_contract()`

```python
def create_block_explorer_link_smart_contract(transaction_hash: str) ‑> str
```
Create block explorer link for smart contract.

---

### `create_block_explorer_link_transaction()`

```python
def create_block_explorer_link_transaction(transaction_hash: str) ‑> str
```
Create block explorer link for transaction.

---

### `read_workflow_wrapper()`

```python
def read_workflow_wrapper(
    alpha: `Alpha`,
    contract_address: str,
    format_function: Callable[..., str]
) ‑> `WorkflowModelOutput`
```
Wrapper function for reading from models through workflows.

**Arguments**

* **`alpha (Alpha)`**: The alpha namespace from an initialized OpenGradient client (client.alpha).
* **`contract_address (str)`**: Smart contract address of the workflow
* **`format_function (Callable)`**: Function for formatting the result returned by read_workflow