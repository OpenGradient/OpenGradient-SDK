import json
import time
from pathlib import Path
from typing import Callable

_ABI_DIR = Path(__file__).parent.parent / "abi"
_BIN_DIR = Path(__file__).parent.parent / "bin"

# How many times we retry a transaction because of nonce conflict
DEFAULT_MAX_RETRY = 5
DEFAULT_RETRY_DELAY_SEC = 1

_NONCE_TOO_LOW = "nonce too low"
_NONCE_TOO_HIGH = "nonce too high"
_INVALID_NONCE = "invalid nonce"
_NONCE_ERRORS = [_INVALID_NONCE, _NONCE_TOO_LOW, _NONCE_TOO_HIGH]


def get_abi(abi_name: str) -> dict:
    """Returns the ABI for the requested contract."""
    abi_path = _ABI_DIR / abi_name
    with open(abi_path, "r") as f:
        return json.load(f)


def get_bin(bin_name: str) -> str:
    """Returns the bytecode for the requested contract."""
    bin_path = _BIN_DIR / bin_name
    with open(bin_path, "r", encoding="utf-8") as f:
        bytecode = f.read().strip()
        if not bytecode.startswith("0x"):
            bytecode = "0x" + bytecode
        return bytecode


def run_with_retry(
    txn_function: Callable,
    max_retries=DEFAULT_MAX_RETRY,
    retry_delay=DEFAULT_RETRY_DELAY_SEC,
):
    """
    Execute a blockchain transaction with retry logic.

    Args:
        txn_function: Function that executes the transaction
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay in seconds between retries for nonce issues
    """
    effective_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRY

    for attempt in range(effective_retries):
        try:
            return txn_function()
        except Exception as e:
            error_msg = str(e).lower()

            if any(error in error_msg for error in _NONCE_ERRORS):
                if attempt == effective_retries - 1:
                    raise RuntimeError(f"Transaction failed after {effective_retries} attempts: {e}")
                time.sleep(retry_delay)
                continue

            raise
