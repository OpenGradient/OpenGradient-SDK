import json
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.opengradient.client import Client
from src.opengradient.types import (
    TEE_LLM,
    ResponseFormat,
    StreamChunk,
    TextGenerationOutput,
    x402SettlementMode,
)

# --- Fixtures ---


@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance."""
    with patch("src.opengradient.client.client.Web3") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock.HTTPProvider.return_value = MagicMock()

        mock_instance.eth.account.from_key.return_value = MagicMock(address="0x1234567890abcdef1234567890abcdef12345678")
        mock_instance.eth.get_transaction_count.return_value = 0
        mock_instance.eth.gas_price = 1000000000
        mock_instance.eth.contract.return_value = MagicMock()

        yield mock_instance


@pytest.fixture
def mock_abi_files():
    """Mock ABI file reads."""
    inference_abi = [{"type": "function", "name": "run", "inputs": [], "outputs": []}]
    precompile_abi = [{"type": "function", "name": "infer", "inputs": [], "outputs": []}]

    def mock_file_open(path, *args, **kwargs):
        if "inference.abi" in str(path):
            return mock_open(read_data=json.dumps(inference_abi))()
        elif "InferencePrecompile.abi" in str(path):
            return mock_open(read_data=json.dumps(precompile_abi))()
        return mock_open(read_data="{}")()

    with patch("builtins.open", side_effect=mock_file_open):
        yield


@pytest.fixture
def client(mock_web3, mock_abi_files):
    """Create a Client instance with mocked dependencies."""
    return Client(
        private_key="0x" + "a" * 64,
        rpc_url="https://test.rpc.url",
        api_url="https://test.api.url",
        contract_address="0x" + "b" * 40,
    )


# --- Client Initialization Tests ---


class TestClientInitialization:
    def test_client_initialization_without_auth(self, mock_web3, mock_abi_files):
        """Test basic client initialization without authentication."""
        client = Client(
            private_key="0x" + "a" * 64,
            rpc_url="https://test.rpc.url",
            api_url="https://test.api.url",
            contract_address="0x" + "b" * 40,
        )

        assert client.model_hub._hub_user is None

    def test_client_initialization_with_auth(self, mock_web3, mock_abi_files):
        """Test client initialization with email/password authentication."""
        with (
            patch("src.opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("src.opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.return_value = {
                "idToken": "test_token",
                "email": "test@test.com",
            }
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth

            client = Client(
                private_key="0x" + "a" * 64,
                rpc_url="https://test.rpc.url",
                api_url="https://test.api.url",
                contract_address="0x" + "b" * 40,
                email="test@test.com",
                password="test_password",
            )

            assert client.model_hub._hub_user is not None
            assert client.model_hub._hub_user["idToken"] == "test_token"

    def test_client_initialization_custom_llm_urls(self, mock_web3, mock_abi_files):
        """Test client initialization with custom LLM server URLs."""
        custom_llm_url = "https://custom.llm.server"
        custom_streaming_url = "https://custom.streaming.server"

        client = Client(
            private_key="0x" + "a" * 64,
            rpc_url="https://test.rpc.url",
            api_url="https://test.api.url",
            contract_address="0x" + "b" * 40,
            og_llm_server_url=custom_llm_url,
            og_llm_streaming_server_url=custom_streaming_url,
        )

        assert client.llm._og_llm_server_url == custom_llm_url
        assert client.llm._og_llm_streaming_server_url == custom_streaming_url


class TestAlphaProperty:
    def test_alpha_initialized_on_client_creation(self, client):
        """Test that alpha is initialized during client creation."""
        assert client.alpha is not None

    def test_alpha_has_infer_method(self, client):
        """Test that alpha namespace has the infer method."""
        assert hasattr(client.alpha, "infer")


# --- Authentication Tests ---


class TestAuthentication:
    def test_login_to_hub_success(self, mock_web3, mock_abi_files):
        """Test successful login to hub."""
        with (
            patch("src.opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("src.opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.return_value = {
                "idToken": "success_token",
                "email": "user@test.com",
            }
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth

            client = Client(
                private_key="0x" + "a" * 64,
                rpc_url="https://test.rpc.url",
                api_url="https://test.api.url",
                contract_address="0x" + "b" * 40,
                email="user@test.com",
                password="password123",
            )

            mock_auth.sign_in_with_email_and_password.assert_called_once_with("user@test.com", "password123")
            assert client.model_hub._hub_user["idToken"] == "success_token"

    def test_login_to_hub_failure(self, mock_web3, mock_abi_files):
        """Test login failure raises exception."""
        with (
            patch("src.opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("src.opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.side_effect = Exception("Invalid credentials")
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth

            with pytest.raises(Exception, match="Invalid credentials"):
                Client(
                    private_key="0x" + "a" * 64,
                    rpc_url="https://test.rpc.url",
                    api_url="https://test.api.url",
                    contract_address="0x" + "b" * 40,
                    email="user@test.com",
                    password="wrong_password",
                )


# --- LLM Tests ---


class TestLLMCompletion:
    def test_llm_completion_success(self, client):
        """Test successful LLM completion."""
        with patch.object(client.llm, "_tee_llm_completion") as mock_tee:
            mock_tee.return_value = TextGenerationOutput(
                transaction_hash="external",
                completion_output="Hello! How can I help?",
                payment_hash="0xpayment123",
            )

            result = client.llm.completion(
                model=TEE_LLM.GPT_4O,
                prompt="Hello",
                max_tokens=100,
            )

            assert result.completion_output == "Hello! How can I help?"
            mock_tee.assert_called_once()


class TestLLMChat:
    def test_llm_chat_success_non_streaming(self, client):
        """Test successful non-streaming LLM chat."""
        with patch.object(client.llm, "_tee_llm_chat") as mock_tee:
            mock_tee.return_value = TextGenerationOutput(
                transaction_hash="external",
                chat_output={"role": "assistant", "content": "Hi there!"},
                finish_reason="stop",
                payment_hash="0xpayment",
            )

            result = client.llm.chat(
                model=TEE_LLM.GPT_4O,
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            assert result.chat_output["content"] == "Hi there!"
            mock_tee.assert_called_once()

    def test_llm_chat_streaming(self, client):
        """Test streaming LLM chat."""
        with patch.object(client.llm, "_tee_llm_chat_stream_sync") as mock_stream:
            mock_chunks = [
                StreamChunk(choices=[], model="gpt-4o"),
                StreamChunk(choices=[], model="gpt-4o", is_final=True),
            ]
            mock_stream.return_value = iter(mock_chunks)

            result = client.llm.chat(
                model=TEE_LLM.GPT_4O,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            chunks = list(result)
            assert len(chunks) == 2
            mock_stream.assert_called_once()


# --- StreamChunk Tests ---


class TestStreamChunk:
    def test_from_sse_data_basic(self):
        """Test parsing basic SSE data."""
        data = {
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.model == "gpt-4o"
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta.content == "Hello"
        assert not chunk.is_final

    def test_from_sse_data_with_finish_reason(self):
        """Test parsing SSE data with finish reason."""
        data = {
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.is_final
        assert chunk.choices[0].finish_reason == "stop"

    def test_from_sse_data_with_usage(self):
        """Test parsing SSE data with usage info."""
        data = {
            "model": "gpt-4o",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        chunk = StreamChunk.from_sse_data(data)

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 10
        assert chunk.usage.total_tokens == 30
        assert chunk.is_final


# --- x402 Settlement Mode Tests ---


class TestX402SettlementMode:
    def test_settlement_modes_values(self):
        """Test settlement mode enum values."""
        assert x402SettlementMode.SETTLE == "private"
        assert x402SettlementMode.SETTLE_BATCH == "batch"
        assert x402SettlementMode.SETTLE_METADATA == "individual"

    def test_settlement_mode_aliases(self):
        """Test settlement mode aliases."""
        assert x402SettlementMode.SETTLE_INDIVIDUAL == x402SettlementMode.SETTLE
        assert x402SettlementMode.SETTLE_INDIVIDUAL_WITH_METADATA == x402SettlementMode.SETTLE_METADATA


# --- ResponseFormat Tests ---


class TestResponseFormat:
    def test_json_object_format(self):
        """Test basic json_object response format serialization."""
        fmt = ResponseFormat(type="json_object")
        assert fmt.to_dict() == {"type": "json_object"}

    def test_json_schema_format(self):
        """Test json_schema response format with full schema."""
        schema = {
            "name": "sentiment",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["positive", "negative"]},
                    "score": {"type": "number"},
                },
                "required": ["label", "score"],
            },
        }
        fmt = ResponseFormat(type="json_schema", json_schema=schema)
        result = fmt.to_dict()

        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "sentiment"
        assert "properties" in result["json_schema"]["schema"]

    def test_json_schema_none_omitted(self):
        """Test that json_schema key is omitted when not provided."""
        fmt = ResponseFormat(type="json_object")
        result = fmt.to_dict()
        assert "json_schema" not in result

    def test_completion_passes_response_format(self, client):
        """Test that completion forwards response_format to the internal method."""
        fmt = ResponseFormat(type="json_object")
        with patch.object(client.llm, "_tee_llm_completion") as mock_tee:
            mock_tee.return_value = TextGenerationOutput(
                transaction_hash="external",
                completion_output='{"result": "ok"}',
            )

            client.llm.completion(
                model=TEE_LLM.GPT_4O,
                prompt="Return JSON",
                response_format=fmt,
            )

            call_kwargs = mock_tee.call_args[1]
            assert call_kwargs["response_format"] is fmt

    def test_chat_passes_response_format(self, client):
        """Test that chat forwards response_format to the internal method."""
        fmt = {"type": "json_object"}
        with patch.object(client.llm, "_tee_llm_chat") as mock_tee:
            mock_tee.return_value = TextGenerationOutput(
                transaction_hash="external",
                chat_output={"role": "assistant", "content": '{"ok": true}'},
                finish_reason="stop",
            )

            client.llm.chat(
                model=TEE_LLM.GPT_4O,
                messages=[{"role": "user", "content": "Return JSON"}],
                response_format=fmt,
            )

            call_kwargs = mock_tee.call_args[1]
            assert call_kwargs["response_format"] is fmt

    def test_chat_stream_passes_response_format(self, client):
        """Test that streaming chat forwards response_format."""
        fmt = ResponseFormat(type="json_object")
        with patch.object(client.llm, "_tee_llm_chat_stream_sync") as mock_stream:
            mock_stream.return_value = iter([
                StreamChunk(choices=[], model="gpt-4o", is_final=True),
            ])

            result = client.llm.chat(
                model=TEE_LLM.GPT_4O,
                messages=[{"role": "user", "content": "Return JSON"}],
                stream=True,
                response_format=fmt,
            )
            list(result)

            call_kwargs = mock_stream.call_args[1]
            assert call_kwargs["response_format"] is fmt
