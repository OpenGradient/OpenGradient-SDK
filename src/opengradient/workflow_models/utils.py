"""Utility functions for the models module."""

import logging
from typing import Callable

from opengradient.client.alpha import Alpha

from .constants import BLOCK_EXPLORER_URL
from .types import WorkflowModelOutput

logger = logging.getLogger(__name__)


def create_block_explorer_link_smart_contract(transaction_hash: str) -> str:
    """Create block explorer link for smart contract."""
    block_explorer_url = BLOCK_EXPLORER_URL + "address/" + transaction_hash
    return block_explorer_url


def create_block_explorer_link_transaction(transaction_hash: str) -> str:
    """Create block explorer link for transaction."""
    block_explorer_url = BLOCK_EXPLORER_URL + "tx/" + transaction_hash
    return block_explorer_url


def read_workflow_wrapper(alpha: Alpha, contract_address: str, format_function: Callable[..., str]) -> WorkflowModelOutput:
    """
    Wrapper function for reading from models through workflows.

    Args:
        alpha (Alpha): The alpha namespace from an initialized OpenGradient client (client.alpha).
        contract_address (str): Smart contract address of the workflow
        format_function (Callable): Function for formatting the result returned by read_workflow

    Raises:
        KeyError: If the workflow result is missing an expected output key.
        RuntimeError: If reading or formatting the workflow result fails.
    """
    try:
        result = alpha.read_workflow_result(contract_address)

        formatted_result = format_function(result)
        block_explorer_link = create_block_explorer_link_smart_contract(contract_address)

        return WorkflowModelOutput(
            result=formatted_result,
            block_explorer_link=block_explorer_link,
        )
    except KeyError as e:
        raise KeyError(
            f"Workflow at {contract_address} is missing expected output key {e}. "
            f"Available keys: {list(result.numbers.keys()) if hasattr(result, 'numbers') else 'unknown'}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Error reading from workflow with address {contract_address}: {e!s}") from e
