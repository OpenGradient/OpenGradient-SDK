# mypy: ignore-errors
import asyncio
import json
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, Sequence, Union, cast

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
)
from langchain_core.messages.tool import ToolCallChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from ..client.llm import LLM
from ..types import StreamChunk, TEE_LLM, TextGenerationOutput, x402SettlementMode

__all__ = ["OpenGradientChatModel"]


def _extract_content(content: Any) -> str:
    """Normalize content to a plain string."""
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


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _serialize_tool_args(raw_args: Any) -> str:
    if raw_args is None:
        return "{}"
    if isinstance(raw_args, str):
        return raw_args
    return json.dumps(raw_args)


def _parse_tool_call(tool_call: Dict[str, Any]) -> ToolCall:
    """Parse a tool call from flat or OpenAI nested response formats."""
    if "function" in tool_call:
        func = tool_call["function"]
        return ToolCall(
            id=tool_call.get("id", ""),
            name=func["name"],
            args=_parse_tool_args(func.get("arguments")),
        )
    return ToolCall(
        id=tool_call.get("id", ""),
        name=tool_call["name"],
        args=_parse_tool_args(tool_call.get("arguments")),
    )


def _parse_tool_call_chunk(tool_call: Dict[str, Any], default_index: int) -> ToolCallChunk:
    if "function" in tool_call:
        func = tool_call.get("function", {})
        name = func.get("name")
        raw_args = func.get("arguments")
    else:
        name = tool_call.get("name")
        raw_args = tool_call.get("arguments")

    args: Optional[str]
    if raw_args is None:
        args = None
    elif isinstance(raw_args, str):
        args = raw_args
    else:
        args = json.dumps(raw_args)

    return ToolCallChunk(
        id=tool_call.get("id"),
        index=tool_call.get("index", default_index),
        name=name,
        args=args,
    )


def _run_coro_sync(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    raise RuntimeError(
        "Synchronous LangChain calls cannot run inside an active event loop for this adapter. "
        "Use `ainvoke`/`astream` instead of `invoke`/`stream`."
    )


def _validate_model_string(model: Union[TEE_LLM, str]) -> Union[TEE_LLM, str]:
    if isinstance(model, Enum):
        model_str = str(model.value)
    else:
        model_str = str(model)
    if "/" not in model_str:
        raise ValueError(
            f"Unsupported model value '{model_str}'. "
            "Expected provider/model format (for example: 'openai/gpt-5')."
        )
    return model


class OpenGradientChatModel(BaseChatModel):
    """OpenGradient adapter class for LangChain chat models."""

    model_cid: Union[TEE_LLM, str]
    max_tokens: int = 300
    temperature: float = 0.0
    x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED

    _llm: LLM = PrivateAttr()
    _owns_client: bool = PrivateAttr(default=False)
    _tools: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tool_choice: Optional[Any] = PrivateAttr(default=None)

    def __init__(
        self,
        private_key: Optional[str] = None,
        model_cid: Optional[Union[TEE_LLM, str]] = None,
        model: Optional[Union[TEE_LLM, str]] = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        x402_settlement_mode: x402SettlementMode = x402SettlementMode.BATCH_HASHED,
        client: Optional[LLM] = None,
        rpc_url: Optional[str] = None,
        tee_registry_address: Optional[str] = None,
        llm_server_url: Optional[str] = None,
        **kwargs: Any,
    ):
        resolved_model_cid = model_cid or model
        if resolved_model_cid is None:
            raise ValueError("model_cid (or model) is required.")
        resolved_model_cid = _validate_model_string(resolved_model_cid)

        super().__init__(
            model_cid=resolved_model_cid,
            max_tokens=max_tokens,
            temperature=temperature,
            x402_settlement_mode=x402_settlement_mode,
            **kwargs,
        )

        if client is not None:
            self._llm = client
            self._owns_client = False
            return

        if not private_key:
            raise ValueError("private_key is required when client is not provided.")

        llm_kwargs: Dict[str, Any] = {}
        if rpc_url is not None:
            llm_kwargs["rpc_url"] = rpc_url
        if tee_registry_address is not None:
            llm_kwargs["tee_registry_address"] = tee_registry_address
        if llm_server_url is not None:
            llm_kwargs["llm_server_url"] = llm_server_url

        self._llm = LLM(private_key=private_key, **llm_kwargs)
        self._owns_client = True

    @property
    def _llm_type(self) -> str:
        return "opengradient"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._llm.close()

    def close(self) -> None:
        if self._owns_client:
            _run_coro_sync(self._llm.close)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model."""
        strict = kwargs.get("strict")
        self._tools = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        self._tool_choice = tool_choice or kwargs.get("tool_choice")
        return self

    @staticmethod
    def _stream_chunk_to_generation(chunk: StreamChunk) -> ChatGenerationChunk:
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None

        usage = None
        if chunk.usage is not None:
            usage = {
                "input_tokens": chunk.usage.prompt_tokens,
                "output_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }

        tool_call_chunks: List[ToolCallChunk] = []
        if delta and delta.tool_calls:
            for index, tool_call in enumerate(delta.tool_calls):
                tool_call_chunks.append(_parse_tool_call_chunk(tool_call, index))

        message_chunk = AIMessageChunk(
            content=_extract_content(delta.content if delta else ""),
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage,
        )

        generation_info: Dict[str, Any] = {}
        if choice and choice.finish_reason is not None:
            generation_info["finish_reason"] = choice.finish_reason

        for key in ["tee_signature", "tee_timestamp", "tee_id", "tee_endpoint", "tee_payment_address"]:
            value = getattr(chunk, key, None)
            if value is not None:
                generation_info[key] = value

        return ChatGenerationChunk(
            message=message_chunk,
            generation_info=generation_info or None,
        )

    def _convert_messages_to_sdk(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
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
                            "id": call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": _serialize_tool_args(call.get("args")),
                            },
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
            elif isinstance(message, ChatMessage):
                sdk_messages.append({"role": message.role, "content": _extract_content(message.content)})
            else:
                raise ValueError(f"Unexpected message type: {message}")
        return sdk_messages

    def _build_chat_kwargs(
        self,
        sdk_messages: List[Dict[str, Any]],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        x402_settlement_mode = kwargs.get("x402_settlement_mode", self.x402_settlement_mode)
        if isinstance(x402_settlement_mode, str):
            x402_settlement_mode = x402SettlementMode(x402_settlement_mode)
        model = _validate_model_string(kwargs.get("model", self.model_cid))

        return {
            "model": model,
            "messages": sdk_messages,
            "stop_sequence": stop,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "tools": kwargs.get("tools", self._tools),
            "tool_choice": kwargs.get("tool_choice", self._tool_choice),
            "x402_settlement_mode": x402_settlement_mode,
            "stream": stream,
        }

    @staticmethod
    def _build_chat_result(chat_output: TextGenerationOutput) -> ChatResult:
        finish_reason = chat_output.finish_reason or ""
        chat_response = chat_output.chat_output or {}
        response_content = _extract_content(chat_response.get("content", ""))

        if chat_response.get("tool_calls"):
            tool_calls = [_parse_tool_call(tc) for tc in chat_response["tool_calls"]]
            ai_message = AIMessage(content=response_content, tool_calls=tool_calls)
        else:
            ai_message = AIMessage(content=response_content)

        generation_info = {"finish_reason": finish_reason} if finish_reason else {}
        return ChatResult(generations=[ChatGeneration(message=ai_message, generation_info=generation_info)])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sdk_messages = self._convert_messages_to_sdk(messages)
        chat_kwargs = self._build_chat_kwargs(sdk_messages, stop, stream=False, **kwargs)
        chat_output = _run_coro_sync(lambda: self._llm.chat(**chat_kwargs))
        if not isinstance(chat_output, TextGenerationOutput):
            raise RuntimeError("Expected non-streaming chat output but received streaming generator.")
        return self._build_chat_result(chat_output)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sdk_messages = self._convert_messages_to_sdk(messages)
        chat_kwargs = self._build_chat_kwargs(sdk_messages, stop, stream=False, **kwargs)
        chat_output = await self._llm.chat(**chat_kwargs)
        if not isinstance(chat_output, TextGenerationOutput):
            raise RuntimeError("Expected non-streaming chat output but received streaming generator.")
        return self._build_chat_result(chat_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        sdk_messages = self._convert_messages_to_sdk(messages)
        chat_kwargs = self._build_chat_kwargs(sdk_messages, stop, stream=True, **kwargs)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Synchronous stream cannot run inside an active event loop for this adapter. "
                "Use `astream` instead."
            )

        loop = asyncio.new_event_loop()
        try:
            stream = loop.run_until_complete(self._llm.chat(**chat_kwargs))
            stream_iter = cast(AsyncIterator[StreamChunk], stream)

            while True:
                try:
                    chunk = loop.run_until_complete(stream_iter.__anext__())
                except StopAsyncIteration:
                    break
                yield self._stream_chunk_to_generation(chunk)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        sdk_messages = self._convert_messages_to_sdk(messages)
        chat_kwargs = self._build_chat_kwargs(sdk_messages, stop, stream=True, **kwargs)
        stream = await self._llm.chat(**chat_kwargs)
        async for chunk in cast(AsyncIterator[StreamChunk], stream):
            yield self._stream_chunk_to_generation(chunk)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_cid,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
