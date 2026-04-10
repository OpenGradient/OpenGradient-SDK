import pytest
import opengradient as og
from opengradient.types import InferenceMode
from pydantic import ValidationError

def test_alpha_infer_validation():
    # Dummy key used for testing validation logic
    alpha = og.Alpha(private_key="0x" + "1" * 64) 
    
    # 1. Test that invalid CID and empty input RAISE an error
    with pytest.raises(ValidationError):
        alpha.infer(
            model_cid="invalid_id", 
            inference_mode=InferenceMode.VANILLA, 
            model_input={}
        )

    # 2. Test that valid data DOES NOT raise a ValidationError
    # Note: It might fail later due to the dummy key, but Pydantic should pass it
    try:
        alpha.infer(
            model_cid="QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco", 
            inference_mode=InferenceMode.VANILLA, 
            model_input={"data": [1, 2, 3]}
        )
    except ValidationError:
        pytest.fail("Pydantic rejected a valid CID!")
    except Exception as e:
        # We ignore other errors (like RPC/Key errors) because 
        # we are only testing the VALIDATION layer here.
        print(f"Skipping non-validation error: {type(e).__name__}")