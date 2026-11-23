"""Pick a TEE gateway, and its keys, from the on-chain TEE registry.

The HPKE public key a request is sealed to and the RSA key its response is
checked against both come from the registry, never from the relay (the relay
is untrusted). This is one ``eth_call`` to ``getActiveTEEs(uint8)`` over plain
JSON-RPC, so the package needs ``eth-abi`` rather than all of ``web3``.

Selection matches the DreamFort dashboard (``src/lib/ohttp-chat.ts``): active
LLM-proxy TEEs only, minus the apps' excluded gateways, chosen at random and
cached for five minutes. A TEE whose OHTTP config uses a suite this client
doesn't implement is skipped rather than failing every request.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import httpx
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_hash.auto import keccak

from ._errors import RegistryError
from ._hpke import SUPPORTED_SUITE
from ._verify import pem_from_der

DEFAULT_RPC_URL = "https://ogevmdevnet.opengradient.ai"
DEFAULT_REGISTRY_ADDRESS = "0x703cB174AEadB35D611858369B4b1111dC9Abda6"
TEE_TYPE_LLM_PROXY = 0

# Gateways the chat-app and the dashboard never route to; keep in step with them.
DEFAULT_EXCLUDED_TEE_IDS: frozenset[str] = frozenset(
    {
        "0x381a4c1d64f4b00791881b58ffaec635ce7a1b2d1af6af49ef1c162282e46edf",
    }
)

DEFAULT_TEE_CACHE_TTL = 300.0
_RPC_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

# TEERegistry.getActiveTEEs(uint8) returns TEEInfo[]; see opengradient/abi/TEERegistry.abi.
_OHTTP_CONFIG_TYPE = "(uint8,uint16,uint16,uint16,bytes,bytes,uint256)"
TEE_INFO_ARRAY_TYPE = f"(address,address,string,bytes,bytes,bytes32,uint8,bool,uint256,uint256,{_OHTTP_CONFIG_TYPE})[]"
_GET_ACTIVE_TEES_SELECTOR = keccak(b"getActiveTEEs(uint8)")[:4]


@dataclass(frozen=True)
class TeeTarget:
    """One TEE gateway a request can be sealed to.

    Attributes:
        tee_id: ``0x`` + keccak256 of the TEE's signing-key DER (lowercase hex).
        endpoint: The gateway URL, for display only (the relay picks the route).
        signing_key_pem: The RSA key responses must be signed with.
        ohttp_public_key: The 32-byte X25519 key requests are sealed to.
        key_id: The OHTTP key-config id that goes in the request header.
    """

    tee_id: str
    endpoint: str
    signing_key_pem: str
    ohttp_public_key: bytes
    key_id: int = 0x01


def get_active_tees_call(registry_address: str, tee_type: int = TEE_TYPE_LLM_PROXY) -> dict[str, Any]:
    """The JSON-RPC ``eth_call`` payload for ``getActiveTEEs(tee_type)``."""
    data = _GET_ACTIVE_TEES_SELECTOR + abi_encode(["uint8"], [tee_type])
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": registry_address, "data": "0x" + data.hex()}, "latest"],
    }


def parse_active_tees(rpc_response: Any, excluded_tee_ids: Iterable[str] = DEFAULT_EXCLUDED_TEE_IDS) -> list[TeeTarget]:
    """Decode an ``eth_call`` response into the TEEs usable for OHTTP.

    Raises:
        RegistryError: If the RPC returned an error or undecodable data.
    """
    if not isinstance(rpc_response, dict):
        raise RegistryError("TEE registry RPC returned a non-object response")
    if rpc_response.get("error"):
        raise RegistryError(f"TEE registry RPC error: {rpc_response['error']}")
    result = rpc_response.get("result")
    if not isinstance(result, str) or not result.startswith("0x"):
        raise RegistryError("TEE registry RPC returned no result")
    try:
        (tees,) = abi_decode([TEE_INFO_ARRAY_TYPE], bytes.fromhex(result[2:]))
    except Exception as exc:  # eth-abi raises several decoding error types
        raise RegistryError(f"could not decode getActiveTEEs result: {exc}") from exc

    excluded = {tee_id.lower() for tee_id in excluded_tee_ids}
    targets: list[TeeTarget] = []
    for tee in tees:
        endpoint, signing_key_der, ohttp_config = tee[2], bytes(tee[3]), tee[10]
        key_id, kem_id, kdf_id, aead_id, hpke_public_key = (
            int(ohttp_config[0]),
            int(ohttp_config[1]),
            int(ohttp_config[2]),
            int(ohttp_config[3]),
            bytes(ohttp_config[4]),
        )
        if not signing_key_der or len(hpke_public_key) != 32:
            continue
        if (kem_id, kdf_id, aead_id) != SUPPORTED_SUITE:
            continue
        tee_id = "0x" + keccak(signing_key_der).hex()
        if tee_id in excluded:
            continue
        try:
            signing_key_pem = pem_from_der(signing_key_der)
        except Exception:  # not a parseable SPKI key; skip this TEE
            continue
        targets.append(
            TeeTarget(
                tee_id=tee_id,
                endpoint=str(endpoint),
                signing_key_pem=signing_key_pem,
                ohttp_public_key=hpke_public_key,
                key_id=key_id,
            )
        )
    return targets


def _pick(targets: list[TeeTarget]) -> TeeTarget:
    if not targets:
        raise RegistryError("the TEE registry lists no active TEE with a usable OHTTP config")
    return random.choice(targets)


class _Cache:
    def __init__(self, ttl: float):
        self._ttl = ttl
        self._target: Optional[TeeTarget] = None
        self._expires_at = 0.0

    def get(self) -> Optional[TeeTarget]:
        if self._target is not None and time.monotonic() < self._expires_at:
            return self._target
        return None

    def put(self, target: TeeTarget) -> None:
        self._target = target
        self._expires_at = time.monotonic() + self._ttl

    def invalidate(self, tee_id: Optional[str]) -> None:
        if self._target is not None and (tee_id is None or self._target.tee_id == tee_id):
            self._target = None


class TeeResolver:
    """Sync registry lookup with a TTL cache. A ``pinned`` target skips the registry."""

    def __init__(
        self,
        *,
        rpc_url: str = DEFAULT_RPC_URL,
        registry_address: str = DEFAULT_REGISTRY_ADDRESS,
        ttl: float = DEFAULT_TEE_CACHE_TTL,
        excluded_tee_ids: Iterable[str] = DEFAULT_EXCLUDED_TEE_IDS,
        pinned: Optional[TeeTarget] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self._rpc_url = rpc_url
        self._payload = get_active_tees_call(registry_address)
        self._excluded = frozenset(excluded_tee_ids)
        self._pinned = pinned
        self._cache = _Cache(ttl)
        self._lock = threading.Lock()
        self._client: Optional[httpx.Client] = None
        self._transport = transport

    @property
    def is_pinned(self) -> bool:
        return self._pinned is not None

    def get(self) -> TeeTarget:
        if self._pinned is not None:
            return self._pinned
        with self._lock:
            cached = self._cache.get()
            if cached is not None:
                return cached
            target = _pick(parse_active_tees(self._fetch(), self._excluded))
            self._cache.put(target)
            return target

    def invalidate(self, tee_id: Optional[str] = None) -> None:
        with self._lock:
            self._cache.invalidate(tee_id)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _fetch(self) -> Any:
        if self._client is None:
            self._client = httpx.Client(transport=self._transport, timeout=_RPC_TIMEOUT)
        try:
            response = self._client.post(self._rpc_url, json=self._payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RegistryError(f"could not read the TEE registry at {self._rpc_url}: {exc}") from exc


class AsyncTeeResolver:
    """Async twin of :class:`TeeResolver`."""

    def __init__(
        self,
        *,
        rpc_url: str = DEFAULT_RPC_URL,
        registry_address: str = DEFAULT_REGISTRY_ADDRESS,
        ttl: float = DEFAULT_TEE_CACHE_TTL,
        excluded_tee_ids: Iterable[str] = DEFAULT_EXCLUDED_TEE_IDS,
        pinned: Optional[TeeTarget] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._rpc_url = rpc_url
        self._payload = get_active_tees_call(registry_address)
        self._excluded = frozenset(excluded_tee_ids)
        self._pinned = pinned
        self._cache = _Cache(ttl)
        self._lock = asyncio.Lock()
        self._client: Optional[httpx.AsyncClient] = None
        self._transport = transport

    @property
    def is_pinned(self) -> bool:
        return self._pinned is not None

    async def get(self) -> TeeTarget:
        if self._pinned is not None:
            return self._pinned
        async with self._lock:
            cached = self._cache.get()
            if cached is not None:
                return cached
            target = _pick(parse_active_tees(await self._fetch(), self._excluded))
            self._cache.put(target)
            return target

    def invalidate(self, tee_id: Optional[str] = None) -> None:
        # Plain attribute writes: safe without the lock on a single event loop.
        self._cache.invalidate(tee_id)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _fetch(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=_RPC_TIMEOUT)
        try:
            response = await self._client.post(self._rpc_url, json=self._payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RegistryError(f"could not read the TEE registry at {self._rpc_url}: {exc}") from exc
