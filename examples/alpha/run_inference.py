import os

import opengradient as og

alpha = og.Alpha(private_key=os.environ["OG_PRIVATE_KEY"])

inference_result = alpha.infer(
    model_cid="hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M",
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

print(f"Output: {inference_result.model_output}")
print(f"Tx hash: {inference_result.transaction_hash}")
