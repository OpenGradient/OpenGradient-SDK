"""Confidential (Oblivious HTTP) chat.

Sends a chat completion end-to-end encrypted to a TEE through an untrusted
relay, then verifies the enclave's signature before printing the answer — the
same OHTTP path the OpenGradient chat app uses in the browser. No wallet is
needed on this side: the relay holds the x402 account and only ever sees
ciphertext.

Set OG_RELAY_URL to your relay's base URL (the /api/v1/chat/ohttp path is
appended automatically). Set OG_RELAY_TOKEN if the relay requires a bearer
token.
"""

import logging
import os

import opengradient as og

logging.basicConfig()
logging.getLogger("opengradient").setLevel(logging.INFO)


def main():
    relay_url = os.environ.get("OG_RELAY_URL", "https://chat-api.opengradient.ai")
    token = os.environ.get("OG_RELAY_TOKEN")

    # Resolves an OHTTP-capable TEE from the on-chain registry and targets the
    # relay's confidential-inference path automatically.
    client = og.ConfidentialLLM(
        relay_url=relay_url,
        auth_headers=(lambda: {"Authorization": f"Bearer {token}"}) if token else None,
    )

    result = client.chat(
        model=og.TEE_LLM.CLAUDE_HAIKU_4_5,
        messages=[{"role": "user", "content": "In one sentence, what is a TEE?"}],
        max_tokens=200,
    )

    print(result.content)

    # The response is cryptographically verified: the proof ties this exact
    # output to the attested enclave that produced it.
    print("\n" + "=" * 40)
    print(f"Verified TEE: {result.proof.tee_id}")
    print(f"Signed at:    {result.proof.timestamp}")


main()
