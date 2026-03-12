import json
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.opengradient.client import Client
from src.opengradient.types import (
    StreamChunk,
    x402SettlementMode,
)


# --- Fixtures ---


@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance."""
    with (
        patch("src.opengradient.client.client.Web3") as mock,
        patch("src.opengradient.client.llm.TEERegistry") as mock_tee_registry,
    ):
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock.HTTPProvider.return_value = MagicMock()

        mock_instance.eth.account.from_key.return_value = MagicMock(address="0x1234567890abcdef1234567890abcdef12345678")
        mock_instance.eth.get_transaction_count.return_value = 0
        mock_instance.eth.gas_price = 1000000000
        mock_instance.eth.contract.return_value = MagicMock()

        # Return a fake active TEE endpoint so Client.__init__ doesn't need a live registry
        mock_tee = MagicMock()
        mock_tee.endpoint = "https://test.tee.server"
        mock_tee.tls_cert_der = None
        mock_tee.tee_id = "test-tee-id"
        mock_tee.payment_address = "0xTestPaymentAddress"
        mock_tee_registry.return_value.get_llm_tee.return_value = mock_tee

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
        """Test client initialization with custom LLM server URL."""
        custom_llm_url = "https://custom.llm.server"

        client = Client(
            private_key="0x" + "a" * 64,
            rpc_url="https://test.rpc.url",
            api_url="https://test.api.url",
            contract_address="0x" + "b" * 40,
            og_llm_server_url=custom_llm_url,
        )

        assert client.llm._tee_endpoint == custom_llm_url


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
        assert x402SettlementMode.PRIVATE == "private"
        assert x402SettlementMode.BATCH_HASHED == "batch"
        assert x402SettlementMode.INDIVIDUAL_FULL == "individual"
