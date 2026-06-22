# Tutorial 04: Building a Verifiable Audit Trail

OpenGradient's core guarantee is that every LLM inference runs inside a
Trusted Execution Environment (TEE) and is settled on-chain. This means
you can prove — cryptographically — what prompt was sent, which model
processed it, and what it returned.

This tutorial shows how to turn that guarantee into a practical audit
trail: a tamper-proof log you can store, inspect, and verify at any time.

## Prerequisites

```bash
pip install opengradient
export OG_PRIVATE_KEY="0x..."
```

Get free OPG test tokens at https://faucet.opengradient.ai

## What makes an inference verifiable?

Every response from `llm.chat()` includes:

| Field | What it is |
|---|---|
| `transaction_hash` | On-chain record of the inference payment |
| `tee_signature` | RSA-PSS signature from the TEE enclave |
| `tee_timestamp` | ISO timestamp from inside the enclave |
| `tee_id` | On-chain registry ID of the enclave |
| `tee_payment_address` | Payment address of the TEE node |

Verify any transaction at https://explorer.opengradient.ai

## Step 1: Choose INDIVIDUAL_FULL settlement

For audit trail use cases, use `INDIVIDUAL_FULL` — it records the
complete input and output on-chain, not just a hash:

```python
import opengradient as og

result = await llm.chat(
    model=og.TEE_LLM.GEMINI_2_5_FLASH,
    messages=[{"role": "user", "content": "Analyze this contract..."}],
    x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
)

print(result.transaction_hash)  # verify on explorer
print(result.tee_signature)     # cryptographic proof
print(result.tee_id)            # which enclave served this
```

## Step 2: Build an audit entry

Capture everything needed to prove the inference happened:

```python
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AuditEntry:
    turn: int
    timestamp_utc: str
    model: str
    prompt: str
    response: str
    transaction_hash: str
    tee_id: Optional[str]
    tee_signature: Optional[str]
    tee_timestamp: Optional[str]
    content_hash: str


def compute_content_hash(prompt: str, response: str) -> str:
    raw = (prompt + "||" + response).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
```

The `content_hash` is a SHA-256 digest of prompt and response together.
If anyone modifies the log after the fact, the hash will no longer match.

## Step 3: Persist and verify

Save to JSON and verify integrity at any time:

```python
import json
from dataclasses import asdict


def save_audit_log(entries, path="audit_log.json"):
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)


def verify_audit_log(path="audit_log.json"):
    with open(path) as f:
        entries = json.load(f)
    all_ok = True
    for e in entries:
        expected = compute_content_hash(e["prompt"], e["response"])
        ok = expected == e["content_hash"]
        print("Turn " + str(e["turn"]) + ": " + ("OK" if ok else "TAMPERED"))
        if not ok:
            all_ok = False
    return all_ok
```

## Step 4: Verify on-chain

Every `transaction_hash` can be looked up on the block explorer:

```
https://explorer.opengradient.ai/tx/<transaction_hash>
```

This gives you the full on-chain record: model, timestamp, and
cryptographic proof that the TEE enclave processed your request.

## Settlement mode comparison

| Mode | On-chain data | Best for |
|---|---|---|
| `PRIVATE` | Payment only | Maximum privacy |
| `BATCH_HASHED` | Merkle hash batch | High volume, cost efficiency |
| `INDIVIDUAL_FULL` | Full input + output | Compliance, audit trails |

## Full example

See `examples/llm_audit_trail.py` for a complete runnable script
that combines all the steps above.

## Summary

- Use `INDIVIDUAL_FULL` settlement for maximum on-chain transparency
- Store `transaction_hash` and `tee_signature` for every inference
- Add a `content_hash` to detect local log tampering
- Verify transactions at https://explorer.opengradient.ai
