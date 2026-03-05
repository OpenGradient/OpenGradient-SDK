# mypy: ignore-errors
import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
)
from langchain_core.messages.tool import ToolMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from ..client import Client
from ..types import TEE_LLM, x402SettlementMode

__all__ = ["OpenGradientChatModel"]
_STREAM_END = object()


def _extract_content(content: Any) -> str:
    """Normalize content to a plain string.

    The API may return content as a string or as a list of content blocks
    like [{"type": "text", "text": "..."}]. This extracts the text in either case.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def _parse_tool_call(tool_call: Dict) -> ToolCall:
    """Parse a tool call from the API response.

    Handles both flat format {"id", "name", "arguments"} and
    OpenAI nested format {"id", "function": {"name", "arguments"}}.
    """
    if "function" in tool_call:
        func = tool_call["function"]
        return ToolCall(
            id=tool_call.get("id", ""),
            name=func["name"],
            args=json.loads(func.get("arguments", "{}")),
        )
    return ToolCall(
        id=tool_call.get("id", ""),
        name=tool_call["name"],
        args=json.loads(tool_call.get("arguments", "{}")),
    )


class OpenGradientChatModel(BaseChatModel):
    """OpenGradient adapter class for LangChain chat model"""

    model_cid: Union[TEE_LLM, str]
    temperature: float = 0.0
    max_tokens: int = 300
    x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.SETTLE_BATCH

    _client: Client = PrivateAttr()
    _tools: List[Dict] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        model_cid: Union[TEE_LLM, str],
        private_key: Optional[str] = None,
        client: Optional[Client] = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
        x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.SETTLE_BATCH,
        **kwargs,
    ):
        super().__init__(
            model_cid=model_cid,
            temperature=temperature,
            max_tokens=max_tokens,
            x402_settlement_mode=x402_settlement_mode,
            **kwargs,
        )
        if client is not None and private_key is not None:
            raise ValueError("Pass either client or private_key, not both.")
        if client is None:
            if private_key is None:
                raise ValueError("Either client or private_key must be provided.")
            client = Client(private_key=private_key)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "opengradient"

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
        ],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model."""
        tool_dicts: List[Dict] = []

        for tool in tools:
            if isinstance(tool, BaseTool):
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": (
                                tool.args_schema.model_json_schema()
                                if hasattr(tool, "args_schema") and tool.args_schema is not None
                                else {}
                            ),
                        },
                    }
                )
            else:
                tool_dicts.append(tool)

        self._tools = tool_dicts

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sdk_messages = self._to_sdk_messages(messages)

        chat_output = self._client.llm.chat(
            model=self.model_cid,
            messages=sdk_messages,
            stop_sequence=stop,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self._tools,
            x402_settlement_mode=self.x402_settlement_mode,
        )

        finish_reason = chat_output.finish_reason or ""
        chat_response = chat_output.chat_output or {}

        if chat_response.get("tool_calls"):
            tool_calls = [_parse_tool_call(tc) for tc in chat_response["tool_calls"]]
            ai_message = AIMessage(content="", tool_calls=tool_calls)
        else:
            ai_message = AIMessage(content=_extract_content(chat_response.get("content", "")))

        return ChatResult(generations=[ChatGeneration(message=ai_message, generation_info={"finish_reason": finish_reason})])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        sdk_messages = self._to_sdk_messages(messages)
        stream = self._client.llm.chat(
            model=self.model_cid,
            messages=sdk_messages,
            stop_sequence=stop,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self._tools,
            x402_settlement_mode=self.x402_settlement_mode,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            content = _extract_content(delta.content)

            additional_kwargs: Dict[str, Any] = {}
            if delta.tool_calls:
                additional_kwargs["tool_calls"] = delta.tool_calls

            chunk_kwargs: Dict[str, Any] = {
                "content": content,
                "additional_kwargs": additional_kwargs,
            }
            if chunk.usage:
                chunk_kwargs["usage_metadata"] = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            generation_info = {"finish_reason": choice.finish_reason} if choice.finish_reason else None
            yield ChatGenerationChunk(
                message=AIMessageChunk(**chunk_kwargs),
                generation_info=generation_info,
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Bridge the sync iterator from the SDK to LangChain's async streaming API.
        iterator = self._stream(messages=messages, stop=stop, **kwargs)
        while True:
            # Use next(..., default) so StopIteration does not cross Future boundaries.
            chunk = await asyncio.to_thread(next, iterator, _STREAM_END)
            if chunk is _STREAM_END:
                break
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_cid,
            "temperature": self.temperature,
        }

    @staticmethod
    def _to_sdk_messages(messages: List[Any]) -> List[Dict[str, Any]]:
        sdk_messages: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                sdk_messages.append({"role": "system", "content": _extract_content(message.content)})
            elif isinstance(message, HumanMessage):
                sdk_messages.append({"role": "user", "content": _extract_content(message.content)})
            elif isinstance(message, AIMessage):
                msg: Dict[str, Any] = {"role": "assistant", "content": _extract_content(message.content)}
                if message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                        }
                        for call in message.tool_calls
                    ]
                sdk_messages.append(msg)
            elif isinstance(message, ToolMessage):
                sdk_messages.append(
                    {
                        "role": "tool",
                        "content": _extract_content(message.content),
                        "tool_call_id": message.tool_call_id,
                    }
                )
            elif isinstance(message, dict):
                role = message.get("role")
                if role not in {"system", "user", "assistant", "tool"}:
                    raise ValueError(f"Unexpected message role in dict message: {role}")

                sdk_message: Dict[str, Any] = {
                    "role": role,
                    "content": _extract_content(message.get("content", "")),
                }
                if role == "assistant" and message.get("tool_calls"):
                    sdk_message["tool_calls"] = message["tool_calls"]
                if role == "tool" and message.get("tool_call_id"):
                    sdk_message["tool_call_id"] = message["tool_call_id"]

                sdk_messages.append(sdk_message)
            else:
                raise ValueError(f"Unexpected message type: {message}")
        return sdk_messages
