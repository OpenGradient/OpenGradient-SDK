"""
OpenGradient Specific Types
"""

import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class x402SettlementMode(str, Enum):
    """
    Settlement modes for x402 payment protocol transactions.

    These modes control how inference data is recorded on-chain for payment settlement
    and auditability. Each mode offers different trade-offs between data completeness,
    privacy, and transaction costs.

    Attributes:
        PRIVATE: Payment-only settlement.
            Only the payment is settled on-chain — no input or output hashes are posted.
            Your inference data remains completely off-chain, ensuring maximum privacy.
            Suitable when payment settlement is required without any on-chain record of execution.
            CLI usage: --settlement-mode private

        BATCH_HASHED: Batch settlement with hashes (default).
            Aggregates multiple inferences into a single settlement transaction
            using a Merkle tree containing input hashes, output hashes, and signatures.
            Most cost-efficient for high-volume applications.
            CLI usage: --settlement-mode batch-hashed

        INDIVIDUAL_FULL: Individual settlement with full metadata.
            Records input data, output data, timestamp, and verification on-chain.
            Provides maximum transparency and auditability.
            Higher gas costs due to larger data storage.
            CLI usage: --settlement-mode individual-full

    Examples:
        >>> from opengradient import x402SettlementMode
        >>> mode = x402SettlementMode.PRIVATE
        >>> print(mode.value)
        'private'
    """

    PRIVATE = "private"
    BATCH_HASHED = "batch"
    INDIVIDUAL_FULL = "individual"


class CandleOrder(IntEnum):
    ASCENDING = 0
    DESCENDING = 1


class CandleType(IntEnum):
    HIGH = 0
    LOW = 1
    OPEN = 2
    CLOSE = 3
    VOLUME = 4


@dataclass
class HistoricalInputQuery:
    base: str
    quote: str
    total_candles: int
    candle_duration_in_mins: int
    order: CandleOrder
    candle_types: List[CandleType]

    def to_abi_format(self) -> tuple:
        """Convert to format expected by contract ABI"""
        return (
            self.base,
            self.quote,
            self.total_candles,
            self.candle_duration_in_mins,
            int(self.order),
            [int(ct) for ct in self.candle_types],
        )


@dataclass
class Number:
    value: int
    decimals: int


@dataclass
class NumberTensor:
    """
    A container for numeric tensor data used as input for ONNX models.

    Attributes:

        name: Identifier for this tensor in the model.

        values: List of integer tuples representing the tensor data.
    """

    name: str
    values: List[Tuple[int, int]]


@dataclass
class StringTensor:
    """
    A container for string tensor data used as input for ONNX models.

    Attributes:

        name: Identifier for this tensor in the model.

        values: List of strings representing the tensor data.
    """

    name: str
    values: List[str]


@dataclass
class ModelInput:
    """
    A collection of tensor inputs required for ONNX model inference.

    Attributes:

        numbers: Collection of numeric tensors for the model.

        strings: Collection of string tensors for the model.
    """

    numbers: List[NumberTensor]
    strings: List[StringTensor]


class InferenceMode(Enum):
    """Enum for the different inference modes available for inference (VANILLA, ZKML, TEE)"""

    VANILLA = 0
    ZKML = 1
    TEE = 2


@dataclass
class ModelOutput:
    """
    Model output struct based on translations from smart contract.
    """

    numbers: Dict[str, np.ndarray]
    strings: Dict[str, np.ndarray]
    jsons: Dict[str, np.ndarray]  # Converts to JSON dictionary
    is_simulation_result: bool


@dataclass
class InferenceResult:
    """
    Output for ML inference requests.
    This class has two fields
        transaction_hash (str): Blockchain hash for the transaction
        model_output (Dict[str, np.ndarray]): Output of the ONNX model
    """

    transaction_hash: str
    model_output: Dict[str, np.ndarray]


@dataclass
class StreamDelta:
    """
    Represents a delta (incremental change) in a streaming response.

    Attributes:
        content: Incremental text content (if any)
        role: Message role (appears in first chunk)
        tool_calls: Tool call information (if function calling is used)
    """

    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class StreamChoice:
    """
    Represents a choice in a streaming response.

    Attributes:
        delta: The incremental changes in this chunk
        index: Choice index (usually 0)
        finish_reason: Reason for completion (appears in final chunk)
    """

    delta: StreamDelta
    index: int = 0
    finish_reason: Optional[str] = None


@dataclass
class StreamUsage:
    """
    Token usage information for a streaming response.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class StreamChunk:
    """
    Represents a single chunk in a streaming LLM response.

    This follows the OpenAI streaming format but is provider-agnostic.
    Each chunk contains incremental data, with the final chunk including
    usage information.

    Attributes:
        choices: List of streaming choices (usually contains one choice)
        model: Model identifier
        usage: Token usage information (only in final chunk)
        is_final: Whether this is the final chunk (before [DONE])
        tee_signature: RSA-PSS signature over the response, present on the final chunk
        tee_timestamp: ISO timestamp from the TEE at signing time, present on the final chunk
        tee_id: On-chain TEE registry ID of the enclave that served this request (final chunk only)
        tee_endpoint: Endpoint URL of the TEE that served this request (final chunk only)
        tee_payment_address: Payment address registered for the TEE (final chunk only)
    """

    choices: List[StreamChoice]
    model: str
    usage: Optional[StreamUsage] = None
    is_final: bool = False
    tee_signature: Optional[str] = None
    tee_timestamp: Optional[str] = None
    tee_id: Optional[str] = None
    tee_endpoint: Optional[str] = None
    tee_payment_address: Optional[str] = None

    @classmethod
    def from_sse_data(cls, data: Dict) -> "StreamChunk":
        """
        Parse a StreamChunk from SSE data dictionary.

        Args:
            data: Dictionary parsed from SSE data line

        Returns:
            StreamChunk instance
        """
        choices = []
        for choice_data in data.get("choices", []):
            # The TEE proxy sometimes sends SSE events using the non-streaming "message"
            # key instead of the standard streaming "delta" key.  Fall back gracefully.
            delta_data = choice_data.get("delta") or choice_data.get("message") or {}
            delta = StreamDelta(content=delta_data.get("content"), role=delta_data.get("role"), tool_calls=delta_data.get("tool_calls"))
            choice = StreamChoice(delta=delta, index=choice_data.get("index", 0), finish_reason=choice_data.get("finish_reason"))
            choices.append(choice)

        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = StreamUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        is_final = any(c.finish_reason is not None for c in choices) or usage is not None

        return cls(
            choices=choices,
            model=data.get("model", "unknown"),
            usage=usage,
            is_final=is_final,
            tee_signature=data.get("tee_signature"),
            tee_timestamp=data.get("tee_timestamp"),
        )


@dataclass
class TextGenerationStream:
    """
    Iterator over ``StreamChunk`` objects from a streaming chat response.

    Returned by ``**`opengradient.client.llm`**.LLM.chat`` when
    ``stream=True``.  Iterate over the stream to receive incremental
    chunks as they arrive from the server.

    Each ``StreamChunk`` contains a list of ``StreamChoice`` objects.
    Access the incremental text via ``chunk.choices[0].delta.content``.
    The final chunk will have ``is_final=True`` and may include
    ``usage`` and ``tee_signature`` / ``tee_timestamp`` fields.

    Usage:
        stream = client.llm.chat(model=og.TEE_LLM.CLAUDE_HAIKU_4_5, messages=[...], stream=True)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
    """

    _iterator: Union[Iterator[str], AsyncIterator[str]]
    _is_async: bool = False

    def __iter__(self):
        """Iterate over stream chunks."""
        return self

    def __aiter__(self):
        """Return async iterator (required for async for loops)."""
        if not self._is_async:
            raise TypeError("Use __iter__ for sync iterators")
        return self

    def __next__(self) -> StreamChunk:
        """Get next stream chunk."""
        import json

        while True:
            try:
                line = next(self._iterator)  # type: ignore[arg-type]
            except StopIteration:
                raise

            if not line or not line.strip():
                continue

            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix

            if data_str.strip() == "[DONE]":
                raise StopIteration

            try:
                data = json.loads(data_str)
                return StreamChunk.from_sse_data(data)
            except json.JSONDecodeError:
                # Skip malformed chunks
                continue

    async def __anext__(self) -> StreamChunk:
        """Get next stream chunk (async version)."""
        import json

        if not self._is_async:
            raise TypeError("Use __next__ for sync iterators")

        while True:
            try:
                line = await self._iterator.__anext__()  # type: ignore[union-attr]
            except StopAsyncIteration:
                raise

            if not line or not line.strip():
                continue

            if not line.startswith("data: "):
                continue

            data_str = line[6:]

            if data_str.strip() == "[DONE]":
                raise StopAsyncIteration

            try:
                data = json.loads(data_str)
                return StreamChunk.from_sse_data(data)
            except json.JSONDecodeError:
                continue


@dataclass
class TextGenerationOutput:
    """
    Output from a non-streaming ``chat()`` or ``completion()`` call.

    Returned by ``**`opengradient.client.llm`**.LLM.chat`` (when ``stream=False``)
    and ``**`opengradient.client.llm`**.LLM.completion``.

    For **chat** requests the response is in ``chat_output``; for
    **completion** requests it is in ``completion_output``.  Only the
    field that matches the request type will be populated.

    Every response includes a ``tee_signature`` and ``tee_timestamp``
    that can be used to cryptographically verify the inference was
    performed inside a TEE enclave.

    Attributes:
        transaction_hash: Blockchain transaction hash.  Set to
            ``"external"`` for TEE-routed providers.
        finish_reason: Reason the model stopped generating
            (e.g. ``"stop"``, ``"tool_call"``, ``"error"``).
            Only populated for chat requests.
        chat_output: Dictionary with the assistant message returned by
            a chat request.  Contains ``role``, ``content``, and
            optionally ``tool_calls``.
        completion_output: Raw text returned by a completion request.
        payment_hash: Payment hash for the x402 transaction.
        tee_signature: RSA-PSS signature over the response produced
            by the TEE enclave.
        tee_timestamp: ISO-8601 timestamp from the TEE at signing
            time.
    """

    transaction_hash: str
    """Blockchain transaction hash. Set to ``"external"`` for TEE-routed providers."""

    finish_reason: Optional[str] = None
    """Reason the model stopped generating (e.g. ``"stop"``, ``"tool_call"``, ``"error"``). Only populated for chat requests."""

    chat_output: Optional[Dict] = None
    """Dictionary with the assistant message returned by a chat request. Contains ``role``, ``content``, and optionally ``tool_calls``."""

    completion_output: Optional[str] = None
    """Raw text returned by a completion request."""

    usage: Optional[Dict] = None
    """Token usage for the request. Contains ``prompt_tokens``, ``completion_tokens``, and ``total_tokens`` when reported by the server."""

    payment_hash: Optional[str] = None
    """Payment hash for the x402 transaction."""

    tee_signature: Optional[str] = None
    """RSA-PSS signature over the response produced by the TEE enclave."""

    tee_timestamp: Optional[str] = None
    """ISO-8601 timestamp from the TEE at signing time."""

    tee_id: Optional[str] = None
    """On-chain TEE registry ID (keccak256 of the enclave's public key) of the TEE that served this request."""

    tee_endpoint: Optional[str] = None
    """Endpoint URL of the TEE that served this request, as registered on-chain."""

    tee_payment_address: Optional[str] = None
    """Payment address registered for the TEE that served this request."""


@dataclass
class AbiFunction:
    name: str
    inputs: List[Union[str, "AbiFunction"]]
    outputs: List[Union[str, "AbiFunction"]]
    state_mutability: str


@dataclass
class Abi:
    functions: List[AbiFunction]

    @classmethod
    def from_json(cls, abi_json):
        functions = []
        for item in abi_json:
            if item["type"] == "function":
                inputs = cls._parse_inputs_outputs(item["inputs"])
                outputs = cls._parse_inputs_outputs(item["outputs"])
                functions.append(AbiFunction(name=item["name"], inputs=inputs, outputs=outputs, state_mutability=item["stateMutability"]))
        return cls(functions=functions)

    @staticmethod
    def _parse_inputs_outputs(items):
        result = []
        for item in items:
            if "components" in item:
                result.append(
                    AbiFunction(name=item["name"], inputs=Abi._parse_inputs_outputs(item["components"]), outputs=[], state_mutability="")
                )
            else:
                result.append(f"{item['name']}:{item['type']}")
        return result


class TEE_LLM(str, Enum):
    """
    Enum for LLM models available for TEE (Trusted Execution Environment) execution.

    TEE mode provides cryptographic verification that inference was performed
    correctly in a secure enclave. Use this for applications requiring
    auditability and tamper-proof AI inference.

    Usage:
        # TEE-verified inference
        result = client.llm.chat(
            model=og.TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    # OpenAI models via TEE
    GPT_4_1_2025_04_14 = "openai/gpt-4.1-2025-04-14"
    O4_MINI = "openai/o4-mini"
    GPT_5 = "openai/gpt-5"
    GPT_5_MINI = "openai/gpt-5-mini"
    GPT_5_2 = "openai/gpt-5.2"

    # Anthropic models via TEE
    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4-5"
    CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4-6"
    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4-5"
    CLAUDE_OPUS_4_5 = "anthropic/claude-opus-4-5"
    CLAUDE_OPUS_4_6 = "anthropic/claude-opus-4-6"

    # Google models via TEE
    # Note: gemini-2.5-flash, gemini-2.5-pro, and gemini-2.5-flash-lite are scheduled
    # for deprecation on June 17, 2026 (flash-lite: July 22, 2026). Replacements will be
    # gemini-3-flash-preview, gemini-3.1-pro-preview, and gemini-3.1-flash-lite-preview.
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
    GEMINI_3_FLASH = "google/gemini-3-flash-preview"

    # xAI Grok models via TEE
    GROK_4 = "x-ai/grok-4"
    GROK_4_FAST = "x-ai/grok-4-fast"
    GROK_4_1_FAST = "x-ai/grok-4-1-fast"
    GROK_4_1_FAST_NON_REASONING = "x-ai/grok-4-1-fast-non-reasoning"


@dataclass
class ResponseFormat:
    """Controls the output format enforced by the TEE gateway.

    Use ``type="json_object"`` to receive any valid JSON object (supported by
    OpenAI, Gemini, and Grok). Use ``type="json_schema"`` with a ``json_schema``
    definition to enforce a specific schema (supported by all providers,
    including Anthropic).

    Attributes:
        type: One of ``"text"``, ``"json_object"``, or ``"json_schema"``.
        json_schema: Schema definition (required when ``type="json_schema"``).
            Must contain ``name`` (str) and ``schema`` (dict).
            ``strict`` (bool) is optional.

    Raises:
        ValueError: If ``type`` is not a recognised value, or if
            ``type="json_schema"`` is used without providing ``json_schema``.

    Examples::

        # Any valid JSON object — OpenAI, Gemini, Grok only
        ResponseFormat(type="json_object")

        # Strict schema — all providers including Anthropic
        ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        )
    """

    type: str
    json_schema: Optional[Dict] = None

    def __post_init__(self) -> None:
        valid_types = ("text", "json_object", "json_schema")
        if self.type not in valid_types:
            raise ValueError(
                f"ResponseFormat.type must be one of {valid_types}, got '{self.type}'"
            )
        if self.type == "json_schema" and not self.json_schema:
            raise ValueError(
                "ResponseFormat.json_schema is required when type='json_schema'"
            )

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict for the TEE gateway request payload."""
        d: Dict = {"type": self.type}
        if self.json_schema is not None:
            d["json_schema"] = self.json_schema
        return d


@dataclass
class SchedulerParams:
    frequency: int
    duration_hours: int

    @property
    def end_time(self) -> int:
        return int(time.time()) + (self.duration_hours * 60 * 60)

    @staticmethod
    def from_dict(data: Optional[Dict[str, int]]) -> Optional["SchedulerParams"]:
        if data is None:
            return None
        return SchedulerParams(frequency=data.get("frequency", 600), duration_hours=data.get("duration_hours", 2))


@dataclass
class ModelRepository:
    name: str
    initialVersion: str


@dataclass
class FileUploadResult:
    modelCid: str
    size: int
