import os

import opengradient as og

hub = og.ModelHub(
    email=os.environ.get("OG_MODEL_HUB_EMAIL"),
    password=os.environ.get("OG_MODEL_HUB_PASSWORD"),
)

hub.create_model(model_name="example-model", model_desc="An example machine learning model for demonstration purposes", version="1.0.0")
