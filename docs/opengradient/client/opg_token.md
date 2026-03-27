---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / opg_token

# Package opengradient.client.opg_token

OPG token Permit2 approval utilities for x402 payments.

## Functions

---

### `approve_opg()`

```python
def approve_opg(
    wallet_account: `LocalAccount`,
    opg_amount: float
) ‑> `Permit2ApprovalResult`
```
Approve Permit2 to spend ``opg_amount`` OPG if the current allowance is insufficient.

Idempotent: if the current allowance is already >= ``opg_amount``, no
transaction is sent.

Best for one-off usage — scripts, notebooks, CLI tools::

    result = approve_opg(wallet, 5.0)

**Arguments**

* **`wallet_account`**: The wallet account to check and approve from.
* **`opg_amount`**: Number of OPG tokens to approve (e.g. ``5.0`` for 5 OPG).
        Converted to base units (18 decimals) internally.

**Returns**

Permit2ApprovalResult: Contains ``allowance_before``,
    ``allowance_after``, and ``tx_hash`` (None when no approval
    was needed).

**`Permit2ApprovalResult` fields:**

* **`allowance_before`**: The Permit2 allowance before the method ran.
* **`allowance_after`**: The Permit2 allowance after the method ran.
* **`tx_hash`**: Transaction hash of the approval, or None if no transaction was needed.

**Raises**

* **`RuntimeError`**: If the approval transaction fails.

---

### `ensure_opg_allowance()`

```python
def ensure_opg_allowance(
    wallet_account: `LocalAccount`,
    min_allowance: float,
    approve_amount: Optional[float] = None
) ‑> `Permit2ApprovalResult`
```
Ensure the Permit2 allowance stays above a minimum threshold.

Only sends an approval transaction when the current allowance drops
below ``min_allowance``. When approval is needed, approves
``approve_amount`` (defaults to ``10 * min_allowance``) to create a
buffer that survives multiple service restarts without re-approving.

Best for backend servers that call this on startup::

    # On startup — only sends a tx when allowance < 5 OPG,
    # then approves 100 OPG so subsequent restarts are free.
    result = ensure_opg_allowance(wallet, min_allowance=5.0, approve_amount=100.0)

**Arguments**

* **`wallet_account`**: The wallet account to check and approve from.
* **`min_allowance`**: The minimum acceptable allowance in OPG. A
        transaction is only sent when the current allowance is
        strictly below this value.
* **`approve_amount`**: The amount of OPG to approve when a transaction
        is needed. Defaults to ``10 * min_allowance``. Must be
        >= ``min_allowance``.

**Returns**

Permit2ApprovalResult: Contains ``allowance_before``,
    ``allowance_after``, and ``tx_hash`` (None when no approval
    was needed).

**`Permit2ApprovalResult` fields:**

* **`allowance_before`**: The Permit2 allowance before the method ran.
* **`allowance_after`**: The Permit2 allowance after the method ran.
* **`tx_hash`**: Transaction hash of the approval, or None if no transaction was needed.

**Raises**

* **`ValueError`**: If ``approve_amount`` is less than ``min_allowance``.
* **`RuntimeError`**: If the approval transaction fails.

---

### `ensure_opg_approval()`

```python
def ensure_opg_approval(
    wallet_account: `LocalAccount`,
    opg_amount: float
) ‑> `Permit2ApprovalResult`
```
Ensure the Permit2 allowance for OPG is at least ``opg_amount``.

.. deprecated::
    Use ``approve_opg`` for one-off approvals or
    ``ensure_opg_allowance`` for server-startup usage.

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

* **`RuntimeError`**: If the approval transaction fails.

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