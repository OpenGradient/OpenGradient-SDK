---
outline: [2,3]
---

[opengradient](../index) / alphasense

# Package opengradient.alphasense

OpenGradient AlphaSense Tools

## Functions

---

### `create_read_workflow_tool()`

```python
def create_read_workflow_tool(
    tool_type: [ToolType](./types),
    workflow_contract_address: str,
    tool_name: str,
    tool_description: str,
    alpha: Optional[`Alpha`] = None,
    output_formatter: Callable[..., str] = &lt;function &lt;lambda&gt;&gt;
) ‑> Union[`BaseTool`, Callable]
```
Creates a tool that reads results from a workflow contract on OpenGradient.

This function generates a tool that can be integrated into either a LangChain pipeline
or a Swarm system, allowing the workflow results to be retrieved and formatted as part
of a chain of operations.

**Arguments**

* **`tool_type (ToolType)`**: Specifies the framework to create the tool for. Use
        ToolType.LANGCHAIN for LangChain integration or ToolType.SWARM for Swarm
        integration.
* **`workflow_contract_address (str)`**: The address of the workflow contract from which
        to read results.
* **`tool_name (str)`**: The name to assign to the created tool. This will be used to
        identify and invoke the tool within the agent.
* **`tool_description (str)`**: A description of what the tool does and how it processes
        the workflow results.
* **`alpha (Alpha, optional)`**: The alpha namespace from an initialized OpenGradient client
        (client.alpha). If not provided, falls back to the global client set via ``opengradient.init()``.
* **`output_formatter (Callable[..., str], optional)`**: A function that takes the workflow output
        and formats it into a string. This ensures the output is compatible with
        the tool framework. Default returns string as is.

**Returns**

BaseTool: For ToolType.LANGCHAIN, returns a LangChain StructuredTool.
Callable: For ToolType.SWARM, returns a decorated function with appropriate metadata.

**Raises**

* **`ValueError`**: If an invalid tool_type is provided.

---

### `create_run_model_tool()`

```python
def create_run_model_tool(
    tool_type: [ToolType](./types),
    model_cid: str,
    tool_name: str,
    model_input_provider: Callable[..., Dict[str, Union[str, int, float, List, `ndarray`]]],
    model_output_formatter: Callable[[`InferenceResult`], str],
    inference: Optional[`Alpha`] = None,
    tool_input_schema: Optional[Type[`BaseModel`]] = None,
    tool_description: str = 'Executes the given ML model',
    inference_mode: `InferenceMode` = InferenceMode.VANILLA
) ‑> Union[`BaseTool`, Callable]
```
Creates a tool that wraps an OpenGradient model for inference.

This function generates a tool that can be integrated into either a LangChain pipeline
or a Swarm system, allowing the model to be executed as part of a chain of operations.
The tool uses the provided input_getter function to obtain the necessary input data and
runs inference using the specified OpenGradient model.

**Arguments**

* **`tool_type (ToolType)`**: Specifies the framework to create the tool for. Use
        ToolType.LANGCHAIN for LangChain integration or ToolType.SWARM for Swarm
        integration.
* **`model_cid (str)`**: The CID of the OpenGradient model to be executed.
* **`tool_name (str)`**: The name to assign to the created tool. This will be used to identify
        and invoke the tool within the agent.
* **`model_input_provider (Callable)`**: A function that takes in the tool_input_schema with arguments
        filled by the agent and returns input data required by the model.

        The function should return data in a format compatible with the model's expectations.
* **`model_output_formatter (Callable[..., str])`**: A function that takes the output of
        the OpenGradient infer method (with type InferenceResult) and formats it into a string.

        This is required to ensure the output is compatible with the tool framework.

        This argument is required and has no default formatter.

        InferenceResult has attributes:
            * transaction_hash (str): Blockchain hash for the transaction
            * model_output (Dict[str, np.ndarray]): Output of the ONNX model
* **`inference (Alpha, optional)`**: The alpha namespace from an initialized OpenGradient client
        (client.alpha). If not provided, falls back to the global client set via ``opengradient.init()``.
* **`tool_input_schema (Type[BaseModel], optional)`**: A Pydantic BaseModel class defining the
        input schema.

        For LangChain tools the schema will be used directly. The defined schema will be used as
        input keyword arguments for the `model_input_provider` function. If no arguments are required
        for the `model_input_provider` function then this schema can be unspecified.

        For Swarm tools the schema will be converted to appropriate annotations.

        Default is None -- an empty schema will be provided for LangChain.
* **`tool_description (str, optional)`**: A description of what the tool does. Defaults to
        "Executes the given ML model".
* **`inference_mode (InferenceMode, optional)`**: The inference mode to use when running
        the model. Defaults to VANILLA.

**Returns**

BaseTool: For ToolType.LANGCHAIN, returns a LangChain StructuredTool.
Callable: For ToolType.SWARM, returns a decorated function with appropriate metadata.

**Raises**

* **`ValueError`**: If an invalid tool_type is provided.

## Classes

### `ToolType`

Indicates the framework the tool is compatible with.

#### Variables

* static `LANGCHAIN`
* static `SWARM`