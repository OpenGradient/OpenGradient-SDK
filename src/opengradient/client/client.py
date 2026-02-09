"""Main Client class that unifies all OpenGradient service namespaces."""

from typing import Optional

from web3 import Web3

from ..defaults import (
    DEFAULT_API_URL,
    DEFAULT_INFERENCE_CONTRACT_ADDRESS,
    DEFAULT_OPENGRADIENT_LLM_SERVER_URL,
    DEFAULT_OPENGRADIENT_LLM_STREAMING_SERVER_URL,
    DEFAULT_RPC_URL,
)
from ..types import PaymentNetwork
from .alpha import Alpha
from .llm import LLM
from .model_hub import ModelHub


class Client:
    """
    Main OpenGradient SDK client.

    Provides unified access to all OpenGradient services including LLM inference,
    on-chain model inference, and the Model Hub. Handles authentication via
    blockchain private key and optional Model Hub credentials.

    Usage:
        client = og.Client(private_key="0x...")
        result = client.llm.chat(model=TEE_LLM.CLAUDE_3_5_HAIKU, messages=[...])
        result = client.alpha.infer(model_cid, InferenceMode.VANILLA, input_data)
        client.model_hub.upload(model_path, model_name, version)
    """

    model_hub: ModelHub
    """Model Hub for creating, versioning, and uploading ML models."""

    llm: LLM
    """LLM chat and completion via TEE-verified execution."""

    alpha: Alpha
    """Alpha Testnet features including on-chain inference, workflow management, and ML model execution."""

    def __init__(
        self,
        private_key: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        rpc_url: str = DEFAULT_RPC_URL,
        api_url: str = DEFAULT_API_URL,
        contract_address: str = DEFAULT_INFERENCE_CONTRACT_ADDRESS,
        og_llm_server_url: Optional[str] = DEFAULT_OPENGRADIENT_LLM_SERVER_URL,
        og_llm_streaming_server_url: Optional[str] = DEFAULT_OPENGRADIENT_LLM_STREAMING_SERVER_URL,
        payment_network: PaymentNetwork = PaymentNetwork.OG_EVM,
    ):
        """
        Initialize the OpenGradient client.

        Args:
            private_key: Private key for OpenGradient transactions.
            email: Email for Model Hub authentication. Optional.
            password: Password for Model Hub authentication. Optional.
            rpc_url: RPC URL for the blockchain network.
            api_url: API URL for the OpenGradient API.
            contract_address: Inference contract address.
            og_llm_server_url: OpenGradient LLM server URL.
            og_llm_streaming_server_url: OpenGradient LLM streaming server URL.
            payment_network: Payment network for x402 transactions. Defaults to OG_EVM.
        """
        blockchain = Web3(Web3.HTTPProvider(rpc_url))
        wallet_account = blockchain.eth.account.from_key(private_key)

        hub_user = None
        if email is not None:
            hub_user = ModelHub._login_to_hub(email, password)

        # Create namespaces
        self.model_hub = ModelHub(hub_user=hub_user)

        self.llm = LLM(
            wallet_account=wallet_account,
            og_llm_server_url=og_llm_server_url,
            og_llm_streaming_server_url=og_llm_streaming_server_url,
            network_filter=payment_network.value,
        )

        self.alpha = Alpha(
            blockchain=blockchain,
            wallet_account=wallet_account,
            inference_hub_contract_address=contract_address,
            api_url=api_url,
        )
