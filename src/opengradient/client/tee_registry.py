"""TEE Registry client for fetching verified TEE endpoints and TLS certificates."""

import logging
import random
import ssl
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Sequence

from web3 import Web3

from ._utils import get_abi

logger = logging.getLogger(__name__)

# TEE types as defined in the registry contract
TEE_TYPE_LLM_PROXY = 0
TEE_TYPE_VALIDATOR = 1


class OhttpConfig(NamedTuple):
    """Mirrors the on-chain TEERegistry.OhttpConfig struct.

    The HPKE key material a client needs to encrypt an Oblivious HTTP request to
    this TEE (the same configuration the chat-app browser client reads).

    Attributes:
        key_id: OHTTP key configuration id.
        kem_id: HPKE KEM id (0x0020 = DHKEM(X25519, HKDF-SHA256)).
        kdf_id: HPKE KDF id (0x0001 = HKDF-SHA256).
        aead_id: HPKE AEAD id (0x0003 = ChaCha20-Poly1305).
        public_key: The TEE's HPKE (X25519) recipient public key.
        key_config: The serialized OHTTP key config blob.
        registered_at: Block timestamp the OHTTP config was registered.
    """

    key_id: int
    kem_id: int
    kdf_id: int
    aead_id: int
    public_key: bytes
    key_config: bytes
    registered_at: int


class TEEInfo(NamedTuple):
    """Mirrors the on-chain TEERegistry.TEEInfo struct (full record).

    Includes the ``ohttp_config`` sub-struct and the RSA ``public_key`` signing
    key, so callers can both encrypt OHTTP requests to the TEE and verify the
    signatures it returns — not just dial its endpoint.
    """

    owner: str
    payment_address: str
    endpoint: str
    public_key: bytes
    tls_certificate: bytes
    pcr_hash: bytes
    tee_type: int
    enabled: bool
    registered_at: int
    last_heartbeat_at: int
    ohttp_config: OhttpConfig


@dataclass(frozen=True)
class TEEEndpoint:
    """A verified TEE resolved from the registry.

    Carries everything needed for both trust paths: the endpoint + pinned TLS
    cert for a direct x402 connection, and the OHTTP/HPKE key material +
    signing key for the oblivious-HTTP relay path.

    Attributes:
        tee_id: keccak256 of the TEE's signing public key (0x-prefixed hex).
        endpoint: The TEE gateway endpoint URL.
        tls_cert_der: DER-encoded TLS certificate pinned at registration.
        payment_address: x402 settlement address for this TEE.
        signing_public_key_der: DER (SPKI) RSA public key the TEE signs with.
        ohttp_config: The TEE's OHTTP/HPKE key configuration, if present.
    """

    tee_id: str
    endpoint: str
    tls_cert_der: bytes
    payment_address: str
    signing_public_key_der: bytes = b""
    ohttp_config: Optional[OhttpConfig] = None


class TEERegistry:
    """
    Queries the on-chain TEE Registry contract to retrieve verified TEE endpoints
    and their TLS certificates.

    Instead of blindly trusting the TLS certificate presented by a TEE server
    (TOFU), this class fetches the certificate that was submitted and verified
    during TEE registration.  Any certificate that does not match the one stored
    in the registry should be rejected.

    Args:
        rpc_url: RPC endpoint for the chain where the registry is deployed.
        registry_address: Address of the deployed TEERegistry contract.
    """

    def __init__(self, rpc_url: str, registry_address: str):
        self._web3 = Web3(Web3.HTTPProvider(rpc_url))
        abi = get_abi("TEERegistry.abi")
        self._contract = self._web3.eth.contract(
            address=Web3.to_checksum_address(registry_address),
            abi=abi,
        )

    def get_active_tees_by_type(self, tee_type: int) -> List[TEEEndpoint]:
        """
        Return all active TEEs of the given type with their endpoints and TLS certs.

        Uses the contract's ``getActiveTEEs(teeType)`` which returns only TEEs that
        are enabled, have a valid (non-revoked) PCR, and a fresh heartbeat — all in
        a single on-chain call.

        Args:
            tee_type: Integer TEE type (0=LLMProxy, 1=Validator).

        Returns:
            List of TEEEndpoint objects for active TEEs of that type.
        """
        type_label = {TEE_TYPE_LLM_PROXY: "LLMProxy", TEE_TYPE_VALIDATOR: "Validator"}.get(tee_type, str(tee_type))

        try:
            tee_infos = self._contract.functions.getActiveTEEs(tee_type).call()
        except Exception as e:
            logger.warning("Failed to fetch active TEEs from registry (type=%s): %s", type_label, e)
            return []

        logger.debug("Registry returned %d active TEE(s) for type=%s", len(tee_infos), type_label)

        endpoints: List[TEEEndpoint] = []
        for raw in tee_infos:
            tee = TEEInfo(*raw)
            tee_id_hex = Web3.keccak(tee.public_key).hex()
            if not tee.endpoint or not tee.tls_certificate:
                logger.warning("  teeId=%s  missing endpoint or TLS cert  (skipped)", tee_id_hex)
                continue

            endpoints.append(
                TEEEndpoint(
                    tee_id=tee_id_hex,
                    endpoint=tee.endpoint,
                    tls_cert_der=bytes(tee.tls_certificate),
                    payment_address=tee.payment_address,
                    signing_public_key_der=bytes(tee.public_key),
                    ohttp_config=_parse_ohttp_config(tee.ohttp_config),
                )
            )

        return endpoints

    def get_llm_tee(self) -> Optional[TEEEndpoint]:
        """
        Return a random active LLM proxy TEE from the registry.

        The returned ``TEEEndpoint`` is the full record: endpoint + pinned TLS
        cert for direct x402 connections, plus the OHTTP/HPKE ``ohttp_config``
        and ``signing_public_key_der`` for the oblivious-HTTP relay path.

        Returns:
            TEEEndpoint for an active LLM proxy TEE, or None if none are available.
        """
        tees = self.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        if not tees:
            logger.warning("No active LLM proxy TEEs found in registry")
            return None

        return random.choice(tees)

    def get_llm_tee_ohttp_config(self) -> Optional[TEEEndpoint]:
        """
        Return a random active LLM proxy TEE that advertises an OHTTP config.

        Like ``get_llm_tee`` but skips TEEs missing HPKE key material, so the
        result is guaranteed usable for the Oblivious HTTP path.

        Returns:
            A TEEEndpoint with a non-empty ``ohttp_config``, or None.
        """
        candidates = [
            tee
            for tee in self.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
            if tee.ohttp_config is not None and len(tee.ohttp_config.public_key) == 32
        ]
        if not candidates:
            logger.warning("No active LLM proxy TEEs with an OHTTP config found in registry")
            return None

        return random.choice(candidates)


def _parse_ohttp_config(raw: Sequence[Any]) -> Optional[OhttpConfig]:
    """Coerce the decoded on-chain ohttpConfig tuple into an OhttpConfig.

    Returns None when the TEE has no OHTTP config registered (empty public key).
    """
    try:
        cfg = OhttpConfig(
            key_id=int(raw[0]),
            kem_id=int(raw[1]),
            kdf_id=int(raw[2]),
            aead_id=int(raw[3]),
            public_key=bytes(raw[4]),
            key_config=bytes(raw[5]),
            registered_at=int(raw[6]),
        )
    except (TypeError, IndexError, ValueError):
        return None
    if not cfg.public_key:
        return None
    return cfg


def build_ssl_context_from_der(der_cert: bytes) -> ssl.SSLContext:
    """
    Build an ssl.SSLContext that trusts *only* the given DER-encoded certificate.

    Hostname verification is disabled because TEE servers are typically addressed
    by IP while the cert may be issued for a different hostname.  The pinned
    certificate itself is the trust anchor — only that cert is accepted.

    Args:
        der_cert: DER-encoded X.509 certificate bytes as stored in the registry.

    Returns:
        ssl.SSLContext configured to accept only the pinned certificate.
    """
    pem = ssl.DER_cert_to_PEM_cert(der_cert)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cadata=pem)
    ctx.check_hostname = False  # TEE cert may be issued for a hostname, we connect via IP
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx
