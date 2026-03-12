import os

import opengradient as og

alpha = og.Alpha(private_key=os.environ.get("OG_PRIVATE_KEY"))

# Define model input
input_query = og.HistoricalInputQuery(
    base="ETH",
    quote="USD",
    total_candles=10,
    candle_duration_in_mins=30,
    order=og.CandleOrder.ASCENDING,
    candle_types=[og.CandleType.OPEN, og.CandleType.HIGH, og.CandleType.LOW, og.CandleType.CLOSE],
)

# Define schedule
scheduler_params = og.SchedulerParams(frequency=60, duration_hours=2)

# Base model CID (ETH volatility forecast) - from https://hub.opengradient.ai
model_cid = "hJD2Ja3akZFt1A2LT-D_1oxOCz_OtuGYw4V9eE1m39M"

# Deploy schedule
contract_address = alpha.new_workflow(
    model_cid=model_cid,
    input_query=input_query,
    # Input name in ONNX model
    input_tensor_name="open_high_low_close",
    scheduler_params=scheduler_params,
)

print(f"Deployed workflow at address: {contract_address}")
