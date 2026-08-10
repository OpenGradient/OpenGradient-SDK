"""
TEE Signature Audit Trail example.

Usage:
    export OG_PRIVATE_KEY="your_private_key"
    python examples/llm_audit_trail.py
"""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import opengradient as og


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
    tee_payment_address: Optional[str]
    content_hash: str


def compute_content_hash(prompt: str, response: str) -> str:
    raw = (prompt + "||" + response).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_audit_entry(turn, model, prompt, result):
    response = result.chat_output.get("content", "") if result.chat_output else ""
    return AuditEntry(
        turn=turn,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model=model.value,
        prompt=prompt,
        response=response,
        transaction_hash=result.transaction_hash,
        tee_id=result.tee_id,
        tee_signature=result.tee_signature,
        tee_timestamp=result.tee_timestamp,
        tee_payment_address=result.tee_payment_address,
        content_hash=compute_content_hash(prompt, response),
    )


def print_audit_entry(entry):
    print("-" * 60)
    print("  Turn             : " + str(entry.turn))
    print("  Model            : " + entry.model)
    print("  Prompt           : " + entry.prompt)
    preview = entry.response[:120] + ("..." if len(entry.response) > 120 else "")
    print("  Response         : " + preview)
    print("  Content hash     : " + entry.content_hash)
    print("  Transaction hash : " + entry.transaction_hash)
    print("  TEE ID           : " + (entry.tee_id or "n/a"))
    print("  TEE timestamp    : " + (entry.tee_timestamp or "n/a"))
    print("  TEE signature    : " + ("present" if entry.tee_signature else "not returned"))


def save_audit_log(entries, path="audit_log.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    print("Audit log saved -> " + path)


def verify_audit_log(path="audit_log.json"):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    print("Verifying " + str(len(entries)) + " entries...")
    all_ok = True
    for e in entries:
        expected = compute_content_hash(e["prompt"], e["response"])
        ok = expected == e["content_hash"]
        print("  Turn " + str(e["turn"]) + ": " + ("OK" if ok else "TAMPERED"))
        if not ok:
            all_ok = False
    return all_ok


async def main():
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise ValueError("OG_PRIVATE_KEY environment variable is not set.")

    llm = og.LLM(private_key=private_key)
    llm.ensure_opg_approval(min_allowance=0.5)

    model = og.TEE_LLM.GEMINI_2_5_FLASH
    audit_log = []

    prompts = [
        "What is a Trusted Execution Environment?",
        "How does cryptographic attestation work?",
        "Why does verifiable AI matter for DeFi?",
    ]

    print("Model : " + model.value)
    print("Turns : " + str(len(prompts)))

    for i, prompt in enumerate(prompts, 1):
        print("\n[Turn " + str(i) + "] " + prompt)
        result = await llm.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
            x402_settlement_mode=og.x402SettlementMode.INDIVIDUAL_FULL,
        )
        entry = build_audit_entry(i, model, prompt, result)
        audit_log.append(entry)
        print_audit_entry(entry)

    save_audit_log(audit_log)
    intact = verify_audit_log()
    result_str = "PASS" if intact else "FAIL"
    print("Audit complete - integrity: " + result_str)
    print("Verify at: https://explorer.opengradient.ai")


if __name__ == "__main__":
    asyncio.run(main())
