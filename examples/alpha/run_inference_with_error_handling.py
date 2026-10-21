import os
import time

import opengradient as og
from web3.exceptions import ContractLogicError

MODEL_CID = "hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M"


def _classify_runtime_error(exc: RuntimeError) -> tuple[str, bool]:
    """Map RuntimeError messages to actionable labels and retryability."""
    message = str(exc).lower()
    if "timeout" in message:
        return "Request timed out while waiting for inference result.", True
    if "network" in message or "connection" in message:
        return "Network/API connectivity issue while fetching inference result.", True
    if "insufficient" in message or "balance" in message:
        return "Insufficient wallet funds for gas or execution.", False
    if "cid" in message or "model" in message:
        return "Invalid model CID or model output mismatch.", False
    return f"Inference failed: {exc}", False


def run_inference_with_retry(max_attempts: int = 3) -> None:
    """Run alpha inference with structured error handling and retry logic."""
    alpha = og.Alpha(private_key=os.environ["OG_PRIVATE_KEY"])

    for attempt in range(1, max_attempts + 1):
        try:
            result = alpha.infer(
                model_cid=MODEL_CID,
                model_input={
                    "open_high_low_close": [
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                    ]
                },
                inference_mode=og.InferenceMode.VANILLA,
            )
            print(f"Output: {result.model_output}")
            print(f"Tx hash: {result.transaction_hash}")
            return
        except ValueError as exc:
            print(f"Invalid input or retry configuration: {exc}")
            return
        except ContractLogicError as exc:
            print(f"Contract reverted during simulation/execution: {exc}")
            return
        except RuntimeError as exc:
            reason, retryable = _classify_runtime_error(exc)
            print(reason)
            if not retryable or attempt == max_attempts:
                return

            backoff_seconds = 2 ** (attempt - 1)
            print(f"Retrying in {backoff_seconds}s (attempt {attempt}/{max_attempts})...")
            time.sleep(backoff_seconds)


if __name__ == "__main__":
    run_inference_with_retry()
