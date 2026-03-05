"""TEE Registry client for fetching verified TEE endpoints and TLS certificates."""

import logging
import ssl
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from web3 import Web3

from ._utils import get_abi

logger = logging.getLogger(__name__)

# TEE types as defined in the registry contract
TEE_TYPE_LLM_PROXY = 0
TEE_TYPE_VALIDATOR = 1


@dataclass
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

        Args:
            tee_type: Integer TEE type (0=LLMProxy, 1=Validator).

        Returns:
            List of TEEEndpoint objects for active TEEs of that type.
        """
        type_label = {TEE_TYPE_LLM_PROXY: "LLMProxy", TEE_TYPE_VALIDATOR: "Validator"}.get(tee_type, str(tee_type))

        try:
            tee_ids: List[bytes] = self._contract.functions.getTEEsByType(tee_type).call()
        except Exception as e:
            logger.warning("Failed to fetch TEE IDs from registry (type=%s): %s", type_label, e)
            return []

        logger.debug("Registry returned %d TEE ID(s) for type=%s", len(tee_ids), type_label)

        endpoints: List[TEEEndpoint] = []
        for tee_id in tee_ids:
            tee_id_hex = tee_id.hex()
            try:
                info = self._contract.functions.getTEE(tee_id).call()
                # TEEInfo tuple order: owner, paymentAddress, endpoint, publicKey,
                #                     tlsCertificate, pcrHash, teeType, active,
                #                     registeredAt, lastUpdatedAt
                owner, payment_address, endpoint, _pub_key, tls_cert_der, _pcr_hash, _tee_type, active, _reg_at, _upd_at = info
                if not active:
                    logger.debug("  teeId=%s  status=inactive  endpoint=%s  (skipped)", tee_id_hex, endpoint)
                    continue
                if not endpoint or not tls_cert_der:
                    logger.warning("  teeId=%s  missing endpoint or TLS cert  (skipped)", tee_id_hex)
                    continue
                logger.info(
                    "  teeId=%s  endpoint=%s  paymentAddress=%s  certBytes=%d",
                    tee_id_hex,
                    endpoint,
                    payment_address,
                    len(tls_cert_der),
                )
                endpoints.append(
                    TEEEndpoint(
                        tee_id=tee_id_hex,
                        endpoint=endpoint,
                        tls_cert_der=bytes(tls_cert_der),
                        payment_address=payment_address,
                    )
                )
            except Exception as e:
                logger.warning("Failed to fetch TEE info for teeId=%s: %s", tee_id_hex, e)

        logger.info("Discovered %d active %s TEE(s) from registry", len(endpoints), type_label)
        return endpoints

    def get_llm_tee(self) -> Optional[TEEEndpoint]:
        """
        Return the first active LLM proxy TEE from the registry.

        Returns:
            TEEEndpoint for an active LLM proxy TEE, or None if none are available.
        """
        logger.debug("Querying TEE registry for active LLM proxy TEEs...")
        tees = self.get_active_tees_by_type(TEE_TYPE_LLM_PROXY)
        if tees:
            logger.info("Selected LLM TEE: endpoint=%s  teeId=%s", tees[0].endpoint, tees[0].tee_id)
        else:
            logger.warning("No active LLM proxy TEEs found in registry")
        return tees[0] if tees else None


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

    cert_file = tempfile.NamedTemporaryFile(
        prefix="og_tee_tls_", suffix=".pem", delete=False, mode="w"
    )
    cert_file.write(pem)
    cert_file.flush()
    cert_file.close()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cert_file.name)
    ctx.check_hostname = False  # TEE cert may be issued for a hostname; we connect via IP
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx
