"""
OpenGradient Client -- service modules for the SDK.

## Modules

- **`opengradient.client.llm`** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement (Base OPG tokens)
- **`opengradient.client.confidential_llm`** -- One-call confidential (Oblivious HTTP) chat: auto-resolves an OHTTP-capable TEE and verifies the response, no wallet needed on the caller
- **`opengradient.client.chat_auth`** -- Browser-based login to an OpenGradient Chat account (the CLI-auth flow): returns the access token and client config for the confidential-inference relay
- **`opengradient.client.model_hub`** -- Model repository management: create, version, and upload ML models
- **`opengradient.client.alpha`** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **`opengradient.client.twins`** -- Digital twins chat via OpenGradient verifiable inference
- **`opengradient.client.opg_token`** -- OPG token Permit2 approval utilities for x402 payments
- **`opengradient.client.tee_registry`** -- TEE registry client for verified endpoints and TLS certificates

## Usage

```python
import opengradient as og

# LLM inference (Base OPG tokens)
llm = og.LLM(private_key="0x...")
llm.ensure_opg_approval(min_allowance=5)
result = await llm.chat(model=og.TEE_LLM.CLAUDE_HAIKU_4_5, messages=[...])

# On-chain model inference (OpenGradient testnet gas tokens)
alpha = og.Alpha(private_key="0x...")
result = alpha.infer(model_cid, og.InferenceMode.VANILLA, model_input)

# Model Hub (requires email auth)
hub = og.ModelHub(email="you@example.com", password="...")
repo = hub.create_model("my-model", "A price prediction model")
```
"""

from .alpha import Alpha
from .chat_auth import ChatAccountAuth, login_chat_account
from .confidential_llm import ConfidentialLLM
from .llm import LLM
from .model_hub import ModelHub
from .tee_ohttp_client import OhttpRelayClient, RelayError, VerifiedChatResponse
from .tee_registry import TEEEndpoint, TEERegistry
from .tee_verify import TeeProof, VerificationError, build_inner_request, verify_response
from .twins import Twins

__all__ = [
    "LLM",
    "Alpha",
    "ModelHub",
    "Twins",
    # Confidential inference: one-call verified, private chat over Oblivious HTTP
    # (auto-resolves an OHTTP-capable TEE from the registry, like the chat app).
    "ConfidentialLLM",
    # Browser-based login to an OpenGradient Chat account (the CLI-auth flow):
    # returns the access token + client config, ready for ConfidentialLLM.
    "login_chat_account",
    "ChatAccountAuth",
    # Verified-inference building blocks: route an OpenAI-style request to a TEE
    # through an untrusted relay, then cryptographically verify the response.
    "TEERegistry",
    "TEEEndpoint",
    "OhttpRelayClient",
    "VerifiedChatResponse",
    "RelayError",
    "TeeProof",
    "VerificationError",
    "build_inner_request",
    "verify_response",
]

__pdoc__ = {
    "Alpha": False,
    "LLM": False,
    "ModelHub": False,
    "Twins": False,
    "ConfidentialLLM": False,
    "client": False,
    "exceptions": False,
    "opg_token": False,
    "tee_registry": True,
    "tee_ohttp": True,
    "tee_verify": True,
    "tee_ohttp_client": True,
    "confidential_llm": True,
    "chat_auth": True,
}
