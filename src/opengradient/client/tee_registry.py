"""TEE Registry client for fetching verified TEE endpoints and TLS certificates."""

import logging
import ssl
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

from web3 import Web3

from ._utils import get_abi

logger = logging.getLogger(__name__)

# TEE types as defined in the registry contract
TEE_TYPE_LLM_PROXY = 0
TEE_TYPE_VALIDATOR = 1


class TEEInfo(NamedTuple):
    """Mirrors the on-chain TEERegistry.TEEInfo struct."""

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


@dataclass(frozen=True)
class TEEEndpoint:
    """A verified TEE with its endpoint URL and TLS certificate from the registry."""

    tee_id: str
    endpoint: str
    tls_cert_der: bytes
    payment_address: str


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
                )
            )

        return endpoints

    def get_llm_tee(self) -> Optional[TEEEndpoint]:
        """
        Return the first active LLM proxy TEE from the registry.

        Returns:
            TEEEndpoint for an active LLM proxy TEE, or None if none are available.
        """
        tees = self.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        if not tees:
            logger.warning("No active LLM proxy TEEs found in registry")
            return None

        return tees[0]


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
