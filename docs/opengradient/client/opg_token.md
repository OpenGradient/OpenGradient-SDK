---
outline: [2,3]
---

[opengradient](../index) / [client](./index) / opg_token

# Package opengradient.client.opg_token

OPG token Permit2 approval utilities for x402 payments.

## Functions

---

### `ensure_opg_approval()`

```python
def ensure_opg_approval(
    wallet_account: `LocalAccount`,
    opg_amount: float
) ‑> `Permit2ApprovalResult`
```
Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

Checks the current Permit2 allowance for the wallet. If the allowance
is already >= the requested amount, returns immediately without sending
a transaction. Otherwise, sends an ERC-20 approve transaction.

**Arguments**

* **`wallet_account`**: The wallet account to check and approve from.
* **`opg_amount`**: Minimum number of OPG tokens required (e.g. ``5.0``
        for 5 OPG). Converted to base units (18 decimals) internally.

**Returns**

Permit2ApprovalResult: Contains ``allowance_before``,
    ``allowance_after``, and ``tx_hash`` (None when no approval
    was needed).

**`Permit2ApprovalResult` fields:**

* **`allowance_before`**: The Permit2 allowance before the method ran.
* **`allowance_after`**: The Permit2 allowance after the method ran.
* **`tx_hash`**: Transaction hash of the approval, or None if no transaction was needed.

**Raises**

* **`OpenGradientError`**: If the approval transaction fails.

## Classes

### `Permit2ApprovalResult`

Result of a Permit2 allowance check / approval.

**Attributes**

* **`allowance_before`**: The Permit2 allowance before the method ran.
* **`allowance_after`**: The Permit2 allowance after the method ran.
* **`tx_hash`**: Transaction hash of the approval, or None if no transaction was needed.

#### Constructor

```python
def __init__(allowance_before: int, allowance_after: int, tx_hash: Optional[str] = None)
```

#### Variables

* static `allowance_after` : int
* static `allowance_before` : int
* static `tx_hash` : Optional[str]