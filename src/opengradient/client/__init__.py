"""
OpenGradient Client -- service modules for the SDK.

## Modules

- **`opengradient.client.llm`** -- LLM chat and text completion with TEE-verified execution and x402 payment settlement (Base Sepolia OPG tokens)
- **`opengradient.client.model_hub`** -- Model repository management: create, version, and upload ML models
- **`opengradient.client.alpha`** -- Alpha Testnet features: on-chain ONNX model inference (VANILLA, TEE, ZKML modes), workflow deployment, and scheduled ML model execution (OpenGradient testnet gas tokens)
- **`opengradient.client.twins`** -- Digital twins chat via OpenGradient verifiable inference

## Usage

```python
import opengradient as og

# LLM inference (Base Sepolia OPG tokens)
llm = og.LLM(private_key="0x...")
llm.ensure_opg_approval(opg_amount=5)
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
from .llm import LLM
from .model_hub import ModelHub
from .twins import Twins

__all__ = ["LLM", "Alpha", "ModelHub", "Twins"]

__pdoc__ = {
    "Alpha": False,
    "LLM": False,
    "ModelHub": False,
    "Twins": False,
    "client": False,
    "exceptions": False,
    "opg_token": False,
    "tee_registry": False,
}
