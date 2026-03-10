import os
import unittest
from enum import Enum

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

import opengradient as og
from opengradient import LLM, InferenceResult
from opengradient.alphasense import ToolType, create_read_workflow_tool, create_run_model_tool
from opengradient.llm import OpenGradientChatModel


class TestLLM(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        private_key = os.environ.get("PRIVATE_KEY")
        if not private_key:
            raise ValueError("PRIVATE_KEY environment variable is not set")

        self.client = og.Client(private_key=private_key)
        self.llm = OpenGradientChatModel(private_key=private_key, model_cid=LLM.CLAUDE_SONNET_4_6)

    def test_simple_completion(self):
        message = self.llm.invoke("say 'hello'. literally")
        self.assertIn("hello", message.content)

    def test_tool_call(self):
        @tool
        def get_balance():
            """Returns the user's balance"""
            return "Your balance is 0.145"

        agent_executor = create_react_agent(self.llm, [get_balance])
        events = agent_executor.stream({"messages": [("user", "What is my balance?")]}, stream_mode="values", debug=False)

        self.assertIn("0.145", list(events)[-1]["messages"][-1].content)

    def test_read_workflow(self):
        # Read current workflow result
        workflow_result = self.client.alpha.read_workflow_result(contract_address="0x6e0641925b845A1ca8aA9a890C4DEF388E9197e0")
        expected_result = str(workflow_result.numbers["Y"][0])

        btc_workflow_tool = create_read_workflow_tool(
            tool_type=ToolType.LANGCHAIN,
            workflow_contract_address="0x6e0641925b845A1ca8aA9a890C4DEF388E9197e0",
            tool_name="ETH_Price_Forecast",
            tool_description="Reads latest forecast for ETH price",
            alpha=self.client.alpha,
            output_formatter=lambda x: x,
        )

        agent_executor = create_react_agent(self.llm, [btc_workflow_tool])
        events = agent_executor.stream(
            {"messages": [("user", "Please print the raw value of the latest ETH forecast?")]}, stream_mode="values", debug=False
        )

        # Just checks that the first 5 values are in the result
        self.assertIn(expected_result[:5], list(events)[-1]["messages"][-1].content)

    def test_run_model_no_schema(self):
        model_input = {
            "open_high_low_close": [
                [2535.79, 2535.79, 2505.37, 2515.36],
                [2515.37, 2516.37, 2497.27, 2506.94],
                [2506.94, 2515, 2506.35, 2508.77],
                [2508.77, 2519, 2507.55, 2518.79],
                [2518.79, 2522.1, 2513.79, 2517.92],
                [2517.92, 2521.4, 2514.65, 2518.13],
                [2518.13, 2525.4, 2517.2, 2522.6],
                [2522.59, 2528.81, 2519.49, 2526.12],
                [2526.12, 2530, 2524.11, 2529.99],
                [2529.99, 2530.66, 2525.29, 2526],
            ]
        }

        def model_input_provider():
            return model_input

        def output_formatter(inference_result: InferenceResult):
            return format(float(inference_result.model_output["Y"].item()), ".3%")

        run_model_tool = create_run_model_tool(
            tool_type=ToolType.LANGCHAIN,
            model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
            tool_name="One_hour_volatility_ETH_USDT",
            model_input_provider=model_input_provider,
            model_output_formatter=output_formatter,
            inference=self.client.alpha,
            tool_description="This tool measures the live 1 hour volatility for the trading pair ETH/USDT.",
            inference_mode=og.InferenceMode.VANILLA,
        )

        expected_result = self.client.alpha.infer(
            inference_mode=og.InferenceMode.VANILLA, model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N", model_input=model_input
        )
        formatted_expected_result = format(float(expected_result.model_output["Y"].item()), ".3%")

        agent_executor = create_react_agent(self.llm, [run_model_tool])
        events = agent_executor.stream(
            {"messages": [("user", "Please calculate the most recent 1 hour volatility measure for ETH/USDT")]},
            stream_mode="values",
            debug=False,
        )

        self.assertIn(formatted_expected_result, list(events)[-1]["messages"][-1].content)

    def test_run_model(
        self,
    ):
        class Token(str, Enum):
            ETH = "ethereum"
            BTC = "bitcoin"

        # If model_input_provider doesn't require agent input then schema is not necessary.
        class InputSchema(BaseModel):
            token: Token = Field(default=Token.ETH, description="Token name specified by user.")

        eth_model_input = {
            "price_series": [
                98.76,
                99.45,
                100.89,
                101.34,
                101.23,
                102.89,
                103.78,
                104.56,
                105.89,
                106.78,
                107.45,
                108.34,
                109.23,
                110.12,
                111.23,
                112.34,
                113.45,
                114.56,
                115.67,
                116.78,
                117.89,
                118.9,
                119.91,
                120.92,
            ]
        }

        btc_model_input = {
            "price_series": [
                120.92,
                119.91,
                118.9,
                117.89,
                116.78,
                115.67,
                114.56,
                113.45,
                112.34,
                111.23,
                110.12,
                109.23,
                108.34,
                107.45,
                106.78,
                105.89,
                104.56,
                103.78,
                102.89,
                101.23,
                101.34,
                100.89,
                99.45,
                98.76,
            ]
        }

        def model_input_provider(**llm_input):
            token = llm_input.get("token")
            if token == Token.BTC:
                return btc_model_input
            elif token == Token.ETH:
                return eth_model_input
            else:
                raise ValueError("Unexpected option found")

        def output_formatter(inference_result: InferenceResult):
            return format(float(inference_result.model_output["std"].item()), ".3%")

        run_model_tool = create_run_model_tool(
            tool_type=ToolType.LANGCHAIN,
            model_cid="QmZdSfHWGJyzBiB2K98egzu3MypPcv4R1ASypUxwZ1MFUG",
            tool_name="Return_volatility_tool",
            model_input_provider=model_input_provider,
            model_output_formatter=output_formatter,
            inference=self.client.alpha,
            tool_input_schema=InputSchema,
            tool_description="This tool takes a token and measures the return volatility (standard deviation of returns).",
            inference_mode=og.InferenceMode.VANILLA,
        )

        # Test option ETH
        expected_result_eth = self.client.alpha.infer(
            inference_mode=og.InferenceMode.VANILLA, model_cid="QmZdSfHWGJyzBiB2K98egzu3MypPcv4R1ASypUxwZ1MFUG", model_input=eth_model_input
        )
        formatted_expected_result_eth = format(float(expected_result_eth.model_output["std"].item()), ".3%")

        agent_executor = create_react_agent(self.llm, [run_model_tool])
        events = agent_executor.stream(
            {"messages": [("user", "Please calculate the volatility measurement for ETH")]},
            stream_mode="values",
            debug=False,
        )

        self.assertIn(formatted_expected_result_eth, list(events)[-1]["messages"][-1].content)

        # Test option BTC
        expected_result_btc = self.client.alpha.infer(
            inference_mode=og.InferenceMode.VANILLA, model_cid="QmZdSfHWGJyzBiB2K98egzu3MypPcv4R1ASypUxwZ1MFUG", model_input=btc_model_input
        )
        formatted_expected_result_btc = format(float(expected_result_btc.model_output["std"].item()), ".3%")

        agent_executor = create_react_agent(self.llm, [run_model_tool])
        events = agent_executor.stream(
            {"messages": [("user", "Please calculate the volatility measurement for BTC")]},
            stream_mode="values",
            debug=False,
        )

        self.assertIn(formatted_expected_result_btc, list(events)[-1]["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
