from typing import Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool

from ..client.alpha import Alpha
from .types import ToolType


def create_read_workflow_tool(
    tool_type: ToolType,
    workflow_contract_address: str,
    tool_name: str,
    tool_description: str,
    alpha: Optional[Alpha] = None,
    output_formatter: Callable[..., str] = lambda x: x,
) -> BaseTool | Callable:
    """
    Creates a tool that reads results from a workflow contract on OpenGradient.

    This function generates a tool that can be integrated into either a LangChain pipeline
    or a Swarm system, allowing the workflow results to be retrieved and formatted as part
    of a chain of operations.

    Args:
        tool_type (ToolType): Specifies the framework to create the tool for. Use
            ToolType.LANGCHAIN for LangChain integration or ToolType.SWARM for Swarm
            integration.
        workflow_contract_address (str): The address of the workflow contract from which
            to read results.
        tool_name (str): The name to assign to the created tool. This will be used to
            identify and invoke the tool within the agent.
        tool_description (str): A description of what the tool does and how it processes
            the workflow results.
        alpha (Alpha, optional): The alpha namespace from an initialized OpenGradient client
            (client.alpha). If not provided, falls back to the global client set via ``opengradient.init()``.
        output_formatter (Callable[..., str], optional): A function that takes the workflow output
            and formats it into a string. This ensures the output is compatible with
            the tool framework. Default returns string as is.

    Returns:
        BaseTool: For ToolType.LANGCHAIN, returns a LangChain StructuredTool.
        Callable: For ToolType.SWARM, returns a decorated function with appropriate metadata.

    Raises:
        ValueError: If an invalid tool_type is provided.

    Examples:
        >>> def format_output(output):
        ...     return f"Workflow status: {output.get('status', 'Unknown')}"
        >>> # Create a LangChain tool
        >>> langchain_tool = create_read_workflow_tool(
        ...     tool_type=ToolType.LANGCHAIN,
        ...     workflow_contract_address="0x123...",
        ...     tool_name="workflow_reader",
        ...     alpha=client.alpha,
        ...     output_formatter=format_output,
        ...     tool_description="Reads and formats workflow execution results"
        ... )
    """

    if alpha is None:
        raise ValueError("No alpha instance provided. Pass alpha=og.Alpha(private_key=...).")

    # define runnable
    def read_workflow():
        output = alpha.read_workflow_result(contract_address=workflow_contract_address)
        return output_formatter(output)

    if tool_type == ToolType.LANGCHAIN:
        return StructuredTool.from_function(func=read_workflow, name=tool_name, description=tool_description, args_schema=None)
    elif tool_type == ToolType.SWARM:
        read_workflow.__name__ = tool_name
        read_workflow.__doc__ = tool_description
        return read_workflow
    else:
        raise ValueError(f"Invalid tooltype: {tool_type}")
