import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool

from opengradient.agents.og_langchain import (
    OpenGradientChatModel,
    _extract_content,
    _parse_tool_args,
    _parse_tool_call,
)
from opengradient.types import StreamChoice, StreamChunk, StreamDelta, StreamUsage, TEE_LLM, TextGenerationOutput, x402SettlementMode
from opengradient.types import TEE_LLM, TextGenerationOutput, x402SettlementMode


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM instance."""
    with patch("opengradient.agents.og_langchain.LLM") as MockLLM:
        mock_instance = MagicMock()
        mock_instance.chat = AsyncMock()
        mock_instance.close = AsyncMock()
        MockLLM.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def model(mock_llm_client):
    """Create an OpenGradientChatModel with a mocked LLM client."""
    return OpenGradientChatModel(private_key="0x" + "a" * 64, model_cid=TEE_LLM.GPT_5)


class TestOpenGradientChatModel:
    def test_initialization(self, model):
        assert model.model_cid == TEE_LLM.GPT_5
        assert model.max_tokens == 300
        assert model.temperature == 0.0
        assert model.x402_settlement_mode == x402SettlementMode.BATCH_HASHED
        assert model._llm_type == "opengradient"

    def test_initialization_custom_fields(self, mock_llm_client):
        model = OpenGradientChatModel(
            private_key="0x" + "a" * 64,
            model_cid=TEE_LLM.CLAUDE_HAIKU_4_5,
            max_tokens=1000,
            temperature=0.2,
            x402_settlement_mode=x402SettlementMode.PRIVATE,
        )
        assert model.max_tokens == 1000
        assert model.temperature == 0.2
        assert model.x402_settlement_mode == x402SettlementMode.PRIVATE

    def test_initialization_with_client(self):
        client = MagicMock()
        model = OpenGradientChatModel(client=client, model=TEE_LLM.GPT_5)
        assert model._llm is client
        assert model._owns_client is False

    def test_requires_model(self):
        with pytest.raises(ValueError, match="model_cid \\(or model\\) is required"):
            OpenGradientChatModel(private_key="0x" + "a" * 64)

    def test_validates_model_format(self):
        with pytest.raises(ValueError, match="Expected provider/model format"):
            OpenGradientChatModel(private_key="0x" + "a" * 64, model="gpt-5")

    def test_identifying_params(self, model):
        assert model._identifying_params == {
            "model_name": TEE_LLM.GPT_5,
            "temperature": 0.0,
            "max_tokens": 300,
        }


class TestGenerate:
    def test_text_response(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "Hello there!"},
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello there!"
        assert result.generations[0].generation_info == {"finish_reason": "stop"}

    async def test_async_text_response(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "Hello async!"},
        )

        result = await model._agenerate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "Hello async!"
        assert result.generations[0].generation_info == {"finish_reason": "stop"}

    def test_tool_call_response_flat_format(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
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

    def test_tool_call_response_nested_format(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
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

    def test_content_as_list_of_blocks(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={
                "role": "assistant",
                "content": [{"index": 0, "text": "Hello there!", "type": "text"}],
            },
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "Hello there!"

    def test_empty_chat_output(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output=None,
        )

        result = model._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == ""


class TestMessageConversion:
    def test_converts_all_message_types(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
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
            ChatMessage(role="developer", content="Prefer concise answers."),
        ]

        model._generate(messages)

        sdk_messages = mock_llm_client.chat.call_args.kwargs["messages"]

        assert sdk_messages[0] == {"role": "system", "content": "You are helpful."}
        assert sdk_messages[1] == {"role": "user", "content": "Hi"}
        assert sdk_messages[2] == {"role": "assistant", "content": "Hello!"}
        assert "tool_calls" not in sdk_messages[2]
        assert sdk_messages[3]["role"] == "assistant"
        assert len(sdk_messages[3]["tool_calls"]) == 1
        assert sdk_messages[3]["tool_calls"][0]["type"] == "function"
        assert sdk_messages[3]["tool_calls"][0]["function"]["name"] == "search"
        assert sdk_messages[3]["tool_calls"][0]["function"]["arguments"] == json.dumps({"q": "test"})
        assert sdk_messages[4] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}
        assert sdk_messages[5] == {"role": "developer", "content": "Prefer concise answers."}

    def test_unsupported_message_type_raises(self, model):
        with pytest.raises(ValueError, match="Unexpected message type"):
            model._convert_messages_to_sdk([MagicMock(spec=[])])

    def test_passes_correct_params_to_client(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "ok"},
        )

        model._generate([HumanMessage(content="Hi")], stop=["END"])

        mock_llm_client.chat.assert_called_once_with(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stop_sequence=["END"],
            max_tokens=300,
            temperature=0.0,
            tools=[],
            tool_choice=None,
            x402_settlement_mode=x402SettlementMode.BATCH_HASHED,
            stream=False,
        )

    def test_build_chat_kwargs_accepts_string_settlement_mode(self, model):
        chat_kwargs = model._build_chat_kwargs(
            sdk_messages=[{"role": "user", "content": "Hi"}],
            stop=None,
            stream=False,
            x402_settlement_mode="private",
        )
        assert chat_kwargs["x402_settlement_mode"] == x402SettlementMode.PRIVATE


class TestBindTools:
    def test_bind_base_tool(self, model):
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

    def test_bind_tool_choice(self, model):
        tool_dict = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A custom tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        model.bind_tools([tool_dict], tool_choice="required")

        assert model._tool_choice == "required"

    def test_tools_used_in_generate(self, model, mock_llm_client):
        mock_llm_client.chat.return_value = TextGenerationOutput(
            transaction_hash="external",
            finish_reason="stop",
            chat_output={"role": "assistant", "content": "ok"},
        )

        tool_dict = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A custom tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        model.bind_tools([tool_dict])
        model._generate([HumanMessage(content="Hi")])

        assert mock_llm_client.chat.call_args.kwargs["tools"] == [tool_dict]


class TestStreaming:
    def test_stream_chunk_to_generation(self):
        chunk = StreamChunk(
            choices=[
                StreamChoice(
                    delta=StreamDelta(
                        content="partial",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "index": 0,
                                "function": {"name": "search", "arguments": '{"q":"weather"}'},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            model="gpt-5",
            usage=StreamUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            tee_signature="sig",
            tee_timestamp="ts",
            tee_id="tee_1",
            tee_endpoint="https://tee.example",
            tee_payment_address="0xabc",
        )

        generation = OpenGradientChatModel._stream_chunk_to_generation(chunk)

        assert generation.message.content == "partial"
        assert generation.message.tool_call_chunks[0]["name"] == "search"
        assert generation.message.tool_call_chunks[0]["args"] == '{"q":"weather"}'
        assert generation.message.usage_metadata == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        assert generation.generation_info["finish_reason"] == "tool_calls"
        assert generation.generation_info["tee_signature"] == "sig"


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
    def test_parse_tool_args_invalid_json(self):
        assert _parse_tool_args("not json") == {}

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
