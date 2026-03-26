"""
OpenGradient Agent Framework Adapters

This module provides adapter interfaces to use OpenGradient LLMs with popular AI frameworks
like LangChain. These adapters allow seamless integration of OpenGradient models
into existing applications and agent frameworks.
"""

from typing import Optional, Union

from ..client.llm import LLM
from ..types import TEE_LLM, x402SettlementMode
from .og_langchain import *


def langchain_adapter(
    private_key: Optional[str] = None,
    model_cid: Optional[Union[TEE_LLM, str]] = None,
    model: Optional[Union[TEE_LLM, str]] = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
    client: Optional[LLM] = None,
    rpc_url: Optional[str] = None,
    tee_registry_address: Optional[str] = None,
    llm_server_url: Optional[str] = None,
) -> OpenGradientChatModel:
    """
    Returns an OpenGradient LLM that implements LangChain's LLM interface
    and can be plugged into LangChain agents.
    """
    return OpenGradientChatModel(
        private_key=private_key,
        model_cid=model_cid,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        x402_settlement_mode=x402_settlement_mode,
        client=client,
        rpc_url=rpc_url,
        tee_registry_address=tee_registry_address,
        llm_server_url=llm_server_url,
    )


__all__ = [
    "langchain_adapter",
]

__pdoc__ = {"og_langchain": False}
