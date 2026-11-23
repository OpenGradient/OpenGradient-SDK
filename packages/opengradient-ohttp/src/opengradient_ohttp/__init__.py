"""End-to-end encrypted, TEE-verified chat completions for the OpenAI Python SDK.

```python
import httpx
from openai import OpenAI
from opengradient_ohttp import OHTTPTransport

client = OpenAI(
    base_url="https://ohttp.invalid/v1",
    api_key="df_sk_...",
    http_client=httpx.Client(transport=OHTTPTransport()),
)
```

See the README for streaming, async, errors and reading the TEE proof.
"""

from ._errors import (
    OHTTPError,
    RegistryError,
    StreamIncompleteError,
    TeeVerificationError,
    UnsupportedRequestError,
)
from ._protocol import DEFAULT_RELAY_URL
from ._registry import DEFAULT_REGISTRY_ADDRESS, DEFAULT_RPC_URL, TeeTarget
from ._transport import AsyncOHTTPTransport, OHTTPTransport
from ._verify import TeeProof
from ._version import __version__

__all__ = [
    "AsyncOHTTPTransport",
    "DEFAULT_REGISTRY_ADDRESS",
    "DEFAULT_RELAY_URL",
    "DEFAULT_RPC_URL",
    "OHTTPError",
    "OHTTPTransport",
    "RegistryError",
    "StreamIncompleteError",
    "TeeProof",
    "TeeTarget",
    "TeeVerificationError",
    "UnsupportedRequestError",
    "__version__",
]
