from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from ..client.alpha import Alpha
from ..types import InferenceMode, InferenceResult
from .types import ToolType


def create_run_model_tool(
    tool_type: ToolType,
    model_cid: str,
    tool_name: str,
    model_input_provider: Callable[..., Dict[str, Union[str, int, float, List, np.ndarray]]],
    model_output_formatter: Callable[[InferenceResult], str],
    inference: Optional[Alpha] = None,
    tool_input_schema: Optional[Type[BaseModel]] = None,
    tool_description: str = "Executes the given ML model",
    inference_mode: InferenceMode = InferenceMode.VANILLA,
) -> BaseTool | Callable:
    """
    Creates a tool that wraps an OpenGradient model for inference.

    This function generates a tool that can be integrated into either a LangChain pipeline
    or a Swarm system, allowing the model to be executed as part of a chain of operations.
    The tool uses the provided input_getter function to obtain the necessary input data and
    runs inference using the specified OpenGradient model.

    Args:
        tool_type (ToolType): Specifies the framework to create the tool for. Use
            ToolType.LANGCHAIN for LangChain integration or ToolType.SWARM for Swarm
            integration.
        model_cid (str): The CID of the OpenGradient model to be executed.
        tool_name (str): The name to assign to the created tool. This will be used to identify
            and invoke the tool within the agent.
        model_input_provider (Callable): A function that takes in the tool_input_schema with arguments
            filled by the agent and returns input data required by the model.

            The function should return data in a format compatible with the model's expectations.
        model_output_formatter (Callable[..., str]): A function that takes the output of
            the OpenGradient infer method (with type InferenceResult) and formats it into a string.

            This is required to ensure the output is compatible with the tool framework.

            This argument is required and has no default formatter.

            InferenceResult has attributes:
                * transaction_hash (str): Blockchain hash for the transaction
                * model_output (Dict[str, np.ndarray]): Output of the ONNX model
        inference (Alpha, optional): The alpha namespace from an initialized OpenGradient client
            (client.alpha). If not provided, falls back to the global client set via ``opengradient.init()``.
        tool_input_schema (Type[BaseModel], optional): A Pydantic BaseModel class defining the
            input schema.

            For LangChain tools the schema will be used directly. The defined schema will be used as
            input keyword arguments for the `model_input_provider` function. If no arguments are required
            for the `model_input_provider` function then this schema can be unspecified.

            For Swarm tools the schema will be converted to appropriate annotations.

            Default is None -- an empty schema will be provided for LangChain.
        tool_description (str, optional): A description of what the tool does. Defaults to
            "Executes the given ML model".
        inference_mode (InferenceMode, optional): The inference mode to use when running
            the model. Defaults to VANILLA.

    Returns:
        BaseTool: For ToolType.LANGCHAIN, returns a LangChain StructuredTool.
        Callable: For ToolType.SWARM, returns a decorated function with appropriate metadata.

    Raises:
        ValueError: If an invalid tool_type is provided.

    Examples:
        >>> from pydantic import BaseModel, Field
        >>> from enum import Enum
        >>> from opengradient.alphasense import create_run_model_tool
        >>> class Token(str, Enum):
        ...     ETH = "ethereum"
        ...     BTC = "bitcoin"
        ...
        >>> class InputSchema(BaseModel):
        ...     token: Token = Field(default=Token.ETH, description="Token name specified by user.")
        ...
        >>> eth_model_input = {"price_series": [2010.1, 2012.3, 2020.1, 2019.2]}        # Example data
        >>> btc_model_input = {"price_series": [100001.1, 100013.2, 100149.2, 99998.1]} # Example data
        >>> def model_input_provider(**llm_input):
        ...     token = llm_input.get("token")
        ...     if token == Token.BTC:
        ...             return btc_model_input
        ...     elif token == Token.ETH:
        ...             return eth_model_input
        ...     else:
        ...             raise ValueError("Unexpected token found")
        ...
        >>> def output_formatter(inference_result):
        ...     return format(float(inference_result.model_output["std"].item()), ".3%")
        ...
        >>> run_model_tool = create_run_model_tool(
        ...     tool_type=ToolType.LANGCHAIN,
        ...     model_cid="QmZdSfHWGJyzBiB2K98egzu3MypPcv4R1ASypUxwZ1MFUG",
        ...     tool_name="Return_volatility_tool",
        ...     model_input_provider=model_input_provider,
        ...     model_output_formatter=output_formatter,
        ...     inference=client.alpha,
        ...     tool_input_schema=InputSchema,
        ...     tool_description="This tool takes a token and measures the return volatility (standard deviation of returns).",
        ...     inference_mode=og.InferenceMode.VANILLA,
        ... )
    """

    if inference is None:
        import opengradient as og

        if og.global_client is None:
            raise ValueError(
                "No inference instance provided and no global client initialized. "
                "Either pass inference=client.alpha or call opengradient.init() first."
            )
        inference = og.global_client.alpha

    def model_executor(**llm_input):
        # Pass LLM input arguments (formatted based on tool_input_schema) as parameters into model_input_provider
        model_input = model_input_provider(**llm_input)

        inference_result = inference.infer(model_cid=model_cid, inference_mode=inference_mode, model_input=model_input)

        return model_output_formatter(inference_result)

    if tool_type == ToolType.LANGCHAIN:
        if not tool_input_schema:
            tool_input_schema = type("EmptyInputSchema", (BaseModel,), {})

        return StructuredTool.from_function(
            func=model_executor, name=tool_name, description=tool_description, args_schema=tool_input_schema
        )
    elif tool_type == ToolType.SWARM:
        model_executor.__name__ = tool_name
        model_executor.__doc__ = tool_description
        # Convert Pydantic model to Swarm annotations if provided
        if tool_input_schema:
            model_executor.__annotations__ = _convert_pydantic_to_annotations(tool_input_schema)
        return model_executor
    else:
        raise ValueError(f"Invalid tooltype: {tool_type}")


def _convert_pydantic_to_annotations(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model to function annotations format used by Swarm.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        Dict mapping field names to (type, description) tuples
    """
    annotations = {}
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        description = field.description or ""
        annotations[field_name] = (field_type, description)
    return annotations
