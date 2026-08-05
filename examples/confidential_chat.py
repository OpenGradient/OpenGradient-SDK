"""Confidential (Oblivious HTTP) chat.

Sends a chat completion end-to-end encrypted to a TEE through an untrusted
relay, then verifies the enclave's signature before printing the answer — the
same OHTTP path the OpenGradient chat app uses in the browser. No wallet is
needed on this side: the relay holds the x402 account and only ever sees
ciphertext.

By default this signs you in through the browser (og.login_chat_account) and
uses the account's own relay URL and token — the same CLI-auth flow other
OpenGradient tools use. Set OG_RELAY_URL / OG_RELAY_TOKEN to skip the browser
and point at a relay directly instead.
"""

import logging
import os

import opengradient as og

logging.basicConfig()
logging.getLogger("opengradient").setLevel(logging.INFO)


def main():
    relay_url = os.environ.get("OG_RELAY_URL")
    token = os.environ.get("OG_RELAY_TOKEN")

    if relay_url:
        # Direct: point at a relay yourself (the /api/v1/chat/ohttp path is
        # appended automatically).
        auth_headers = (lambda: {"Authorization": f"Bearer {token}"}) if token else None
    else:
        # Browser login: opens chat.opengradient.ai/cli-auth, waits for sign-in,
        # and hands back the access token plus the relay URL to use.
        auth = og.login_chat_account()
        relay_url = auth.chat_api_base_url
        auth_headers = auth.auth_headers
        if not relay_url:
            raise SystemExit("The chat account did not return a relay URL; set OG_RELAY_URL instead.")

    # Resolves an OHTTP-capable TEE from the on-chain registry and targets the
    # relay's confidential-inference path automatically.
    client = og.ConfidentialLLM(
        relay_url=relay_url,
        auth_headers=auth_headers,
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
