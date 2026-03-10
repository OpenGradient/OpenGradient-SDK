# mypy: ignore-errors
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
)
from langchain_core.messages.tool import ToolMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from ..client import Client
from ..types import TEE_LLM, x402SettlementMode

__all__ = ["OpenGradientChatModel"]


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

    model_cid: str
    max_tokens: int = 300
    x402_settlement_mode: Optional[str] = x402SettlementMode.BATCH_HASHED

    _client: Client = PrivateAttr()
    _tools: List[Dict] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        private_key: str,
        model_cid: TEE_LLM,
        max_tokens: int = 300,
        x402_settlement_mode: Optional[x402SettlementMode] = x402SettlementMode.BATCH_HASHED,
        **kwargs,
    ):
        super().__init__(
            model_cid=model_cid,
            max_tokens=max_tokens,
            x402_settlement_mode=x402_settlement_mode,
            **kwargs,
        )
        self._client = Client(private_key=private_key)

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
        sdk_messages = []
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
            else:
                raise ValueError(f"Unexpected message type: {message}")

        chat_output = self._client.llm.chat(
            model=self.model_cid,
            messages=sdk_messages,
            stop_sequence=stop,
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

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_cid,
        }
