import os
import unittest
from dataclasses import dataclass

import opengradient as og
from opengradient.workflow_models import (
    read_btc_1_hour_price_forecast,
    read_eth_1_hour_price_forecast,
    read_eth_usdt_one_hour_volatility_forecast,
    read_sol_1_hour_price_forecast,
    read_sui_1_hour_price_forecast,
    read_sui_usdt_6_hour_price_forecast,
    read_sui_usdt_30_min_price_forecast,
)
from opengradient.workflow_models.constants import (
    BTC_1_HOUR_PRICE_FORECAST_ADDRESS,
    ETH_1_HOUR_PRICE_FORECAST_ADDRESS,
    ETH_USDT_1_HOUR_VOLATILITY_ADDRESS,
    SOL_1_HOUR_PRICE_FORECAST_ADDRESS,
    SUI_1_HOUR_PRICE_FORECAST_ADDRESS,
    SUI_6_HOUR_PRICE_FORECAST_ADDRESS,
    SUI_30_MINUTE_PRICE_FORECAST_ADDRESS,
)
from opengradient.workflow_models.utils import create_block_explorer_link_smart_contract


@dataclass
class ModelInfo:
    output_name: str
    address: str


class TestWorkflowModels(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        private_key = os.environ.get("PRIVATE_KEY")
        if not private_key:
            raise ValueError("PRIVATE_KEY environment variable is not set")

        self.alpha = og.Alpha(private_key=private_key)

    def test_models(self):
        model_functions = {
            read_eth_usdt_one_hour_volatility_forecast: ModelInfo(output_name="Y", address=ETH_USDT_1_HOUR_VOLATILITY_ADDRESS),
            read_btc_1_hour_price_forecast: ModelInfo(output_name="regression_output", address=BTC_1_HOUR_PRICE_FORECAST_ADDRESS),
            read_eth_1_hour_price_forecast: ModelInfo(output_name="regression_output", address=ETH_1_HOUR_PRICE_FORECAST_ADDRESS),
            read_sol_1_hour_price_forecast: ModelInfo(output_name="regression_output", address=SOL_1_HOUR_PRICE_FORECAST_ADDRESS),
            read_sui_1_hour_price_forecast: ModelInfo(output_name="regression_output", address=SUI_1_HOUR_PRICE_FORECAST_ADDRESS),
            read_sui_usdt_30_min_price_forecast: ModelInfo(
                output_name="destandardized_prediction", address=SUI_30_MINUTE_PRICE_FORECAST_ADDRESS
            ),
            read_sui_usdt_6_hour_price_forecast: ModelInfo(
                output_name="destandardized_prediction", address=SUI_6_HOUR_PRICE_FORECAST_ADDRESS
            ),
        }

        for function, model_info in model_functions.items():
            workflow_result = function(self.alpha)
            expected_result = format(
                float(self.alpha.read_workflow_result(model_info.address).numbers[model_info.output_name].item()), ".10%"
            )
            print(function)
            print("Workflow result: ", workflow_result)
            assert workflow_result.result == expected_result
            assert workflow_result.block_explorer_link == create_block_explorer_link_smart_contract(model_info.address)


if __name__ == "__main__":
    unittest.main()
