import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.opengradient.agents.og_langchain import OpenGradientChatModel, _extract_content, _parse_tool_call
from src.opengradient.types import TEE_LLM, TextGenerationOutput, x402SettlementMode


@pytest.fixture
def mock_client():
    """Create a mock Client instance."""
    with patch("src.opengradient.agents.og_langchain.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def model(mock_client):
    """Create an OpenGradientChatModel with a mocked client."""
    return OpenGradientChatModel(private_key="0x" + "a" * 64, model_cid=TEE_LLM.GPT_5)


class TestOpenGradientChatModel:
    def test_initialization(self, model):
        """Test model initializes with correct fields."""
        assert model.model_cid == TEE_LLM.GPT_5
        assert model.max_tokens == 300
        assert model.x402_settlement_mode == x402SettlementMode.SETTLE_BATCH
        assert model._llm_type == "opengradient"

    def test_initialization_custom_max_tokens(self, mock_client):
        """Test model initializes with custom max_tokens."""
        model = OpenGradientChatModel(private_key="0x" + "a" * 64, model_cid=TEE_LLM.CLAUDE_HAIKU_4_5, max_tokens=1000)
        assert model.max_tokens == 1000

    def test_initialization_custom_settlement_mode(self, mock_client):
        """Test model initializes with custom settlement mode."""
        model = OpenGradientChatModel(
            private_key="0x" + "a" * 64,
            model_cid=TEE_LLM.GPT_5,
            x402_settlement_mode=x402SettlementMode.SETTLE,
        )
        assert model.x402_settlement_mode == x402SettlementMode.SETTLE

    def test_identifying_params(self, model):
        """Test _identifying_params returns model name."""
        assert model._identifying_params == {"model_name": TEE_LLM.GPT_5}


class TestGenerate:
    def test_text_response(self, model, mock_client):
        """Test _generate with a simple text response."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "Hello there!"},
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello there!"
        assert result.generations[0].generation_info == {"finish_reason": "stop"}

    def test_tool_call_response_flat_format(self, model, mock_client):
        """Test _generate with tool calls in flat format {name, arguments}."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="tool_call",
            chat_output={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "get_balance",
                        "arguments": json.dumps({"account": "main"}),
                    }
                ],
            },
        )

        result = model._generate([HumanMessage(content="What is my balance?")])

        ai_msg = result.generations[0].message
        assert ai_msg.content == ""
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["id"] == "call_123"
        assert ai_msg.tool_calls[0]["name"] == "get_balance"
        assert ai_msg.tool_calls[0]["args"] == {"account": "main"}

    def test_tool_call_response_nested_format(self, model, mock_client):
        """Test _generate with tool calls in OpenAI nested format {function: {name, arguments}}."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="tool_call",
            chat_output={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "get_balance",
                            "arguments": json.dumps({"account": "savings"}),
                        },
                    }
                ],
            },
        )

        result = model._generate([HumanMessage(content="What is my balance?")])

        ai_msg = result.generations[0].message
        assert ai_msg.content == ""
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["id"] == "call_456"
        assert ai_msg.tool_calls[0]["name"] == "get_balance"
        assert ai_msg.tool_calls[0]["args"] == {"account": "savings"}

    def test_content_as_list_of_blocks(self, model, mock_client):
        """Test _generate when API returns content as a list of content blocks."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={
                "role": "assistant",
                "content": [{"index": 0, "text": "Hello there!", "type": "text"}],
            },
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "Hello there!"

    def test_empty_chat_output(self, model, mock_client):
        """Test _generate handles None chat_output gracefully."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output=None,
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == ""


class TestMessageConversion:
    def test_converts_all_message_types(self, model, mock_client):
        """Test that all LangChain message types are correctly converted to SDK format."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "ok"},
        )

        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!", tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]

        model._generate(messages)

        sdk_messages = mock_client.llm.chat.call_args.kwargs["messages"]

        assert sdk_messages[0] == {"role": "system", "content": "You are helpful."}
        assert sdk_messages[1] == {"role": "user", "content": "Hi"}
        # AIMessage with no tool_calls should not include tool_calls key
        assert sdk_messages[2] == {"role": "assistant", "content": "Hello!"}
        assert "tool_calls" not in sdk_messages[2]
        # AIMessage with tool_calls should include them in OpenAI nested format
        assert sdk_messages[3]["role"] == "assistant"
        assert len(sdk_messages[3]["tool_calls"]) == 1
        assert sdk_messages[3]["tool_calls"][0]["type"] == "function"
        assert sdk_messages[3]["tool_calls"][0]["function"]["name"] == "search"
        assert sdk_messages[3]["tool_calls"][0]["function"]["arguments"] == json.dumps({"q": "test"})
        # ToolMessage
        assert sdk_messages[4] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}

    def test_unsupported_message_type_raises(self, model, mock_client):
        """Test that unsupported message types raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected message type"):
            model._generate([MagicMock(spec=[])])

    def test_passes_correct_params_to_client(self, model, mock_client):
        """Test that _generate passes model params correctly to the SDK client."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "ok"},
        )

        model._generate([HumanMessage(content="Hi")], stop=["END"])

        mock_client.llm.chat.assert_called_once_with(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stop_sequence=["END"],
            max_tokens=300,
            tools=[],
            x402_settlement_mode=x402SettlementMode.SETTLE_BATCH,
        )


class TestBindTools:
    def test_bind_base_tool(self, model):
        """Test binding a LangChain BaseTool."""

        @tool
        def get_weather(city: str) -> str:
            """Gets the weather for a city."""
            return f"Sunny in {city}"

        result = model.bind_tools([get_weather])

        assert result is model
        assert len(model._tools) == 1
        assert model._tools[0]["type"] == "function"
        assert model._tools[0]["function"]["name"] == "get_weather"
        assert model._tools[0]["function"]["description"] == "Gets the weather for a city."
        assert "properties" in model._tools[0]["function"]["parameters"]

    def test_bind_dict_tool(self, model):
        """Test binding a raw dict tool definition."""
        tool_dict = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A custom tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        model.bind_tools([tool_dict])

        assert model._tools == [tool_dict]

    def test_tools_used_in_generate(self, model, mock_client):
        """Test that bound tools are passed to the client chat call."""
        mock_client.llm.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "ok"},
        )

        tool_dict = {"type": "function", "function": {"name": "my_tool"}}
        model.bind_tools([tool_dict])
        model._generate([HumanMessage(content="Hi")])

        assert mock_client.llm.chat.call_args.kwargs["tools"] == [tool_dict]


class TestExtractContent:
    def test_string_passthrough(self):
        assert _extract_content("hello") == "hello"

    def test_empty_string(self):
        assert _extract_content("") == ""

    def test_none(self):
        assert _extract_content(None) == ""

    def test_list_of_text_blocks(self):
        content = [
            {"index": 0, "text": "Hello ", "type": "text"},
            {"index": 1, "text": "world!", "type": "text"},
        ]
        assert _extract_content(content) == "Hello world!"

    def test_list_of_strings(self):
        assert _extract_content(["hello ", "world"]) == "hello world"


class TestParseToolCall:
    def test_flat_format(self):
        tc = _parse_tool_call({"id": "1", "name": "foo", "arguments": '{"x": 1}'})
        assert tc["name"] == "foo"
        assert tc["args"] == {"x": 1}

    def test_nested_function_format(self):
        tc = _parse_tool_call(
            {
                "id": "2",
                "type": "function",
                "function": {"name": "bar", "arguments": '{"y": 2}'},
            }
        )
        assert tc["name"] == "bar"
        assert tc["args"] == {"y": 2}
        assert tc["id"] == "2"
