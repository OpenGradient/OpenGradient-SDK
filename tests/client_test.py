import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from opengradient.client.alpha import Alpha
from opengradient.client.llm import LLM
from opengradient.client.model_hub import ModelHub
from opengradient.client._conversions import convert_to_model_output
from opengradient.types import (
    StreamChunk,
    x402SettlementMode,
)

FAKE_PRIVATE_KEY = "0x" + "a" * 64

# --- Fixtures ---


@pytest.fixture
def mock_tee_registry():
    """Mock the TEE registry so LLM.__init__ doesn't need a live registry."""
    with (
        patch("opengradient.client.llm.TEERegistry") as mock_tee_registry,
        patch(
            "opengradient.client.tee_connection.build_ssl_context_from_der",
            return_value=MagicMock(),
        ),
    ):
        mock_tee = MagicMock()
        mock_tee.endpoint = "https://test.tee.server"
        mock_tee.tls_cert_der = b"fake-der"
        mock_tee.tee_id = "test-tee-id"
        mock_tee.payment_address = "0xTestPaymentAddress"
        mock_tee_registry.return_value.get_llm_tee.return_value = mock_tee
        yield mock_tee_registry


@pytest.fixture
def mock_web3():
    """Create a mock Web3 instance for Alpha."""
    with patch("opengradient.client.alpha.Web3") as mock:
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


# --- LLM Initialization Tests ---


class TestLLMInitialization:
    def test_llm_initialization(self, mock_tee_registry):
        """Test basic LLM initialization."""
        llm = LLM(private_key=FAKE_PRIVATE_KEY)
        assert llm._tee.get().endpoint == "https://test.tee.server"

    def test_llm_initialization_custom_url(self, mock_tee_registry):
        """Test LLM initialization with custom server URL."""
        custom_llm_url = "https://custom.llm.server"
        llm = LLM.from_url(private_key=FAKE_PRIVATE_KEY, llm_server_url=custom_llm_url)
        assert llm._tee.get().endpoint == custom_llm_url


# --- ModelHub Authentication Tests ---


class TestAuthentication:
    def test_login_to_hub_success(self):
        """Test successful login to hub."""
        with (
            patch("opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.return_value = {
                "idToken": "success_token",
                "email": "user@test.com",
            }
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth

            hub = ModelHub(email="user@test.com", password="password123")

            mock_auth.sign_in_with_email_and_password.assert_called_once_with("user@test.com", "password123")
            assert hub._hub_user["idToken"] == "success_token"

    def test_login_to_hub_failure(self):
        """Test login failure raises exception."""
        with (
            patch("opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.side_effect = Exception("Invalid credentials")
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth

            with pytest.raises(Exception, match="Invalid credentials"):
                ModelHub(email="user@test.com", password="wrong_password")


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


# --- Fix #3: response.json() guard in ModelHub.upload ---


class TestModelHubUploadErrorHandling:
    """upload() must not crash with JSONDecodeError when the server returns HTML."""

    def _make_hub(self):
        with (
            patch("opengradient.client.model_hub._FIREBASE_CONFIG", {"apiKey": "fake"}),
            patch("opengradient.client.model_hub.firebase") as mock_firebase,
        ):
            mock_auth = MagicMock()
            mock_auth.sign_in_with_email_and_password.return_value = {
                "idToken": "tok",
                "email": "u@t.com",
                "expiresIn": "3600",
                "refreshToken": "rt",
            }
            mock_firebase.initialize_app.return_value.auth.return_value = mock_auth
            return ModelHub(email="u@t.com", password="pw")

    def test_html_error_response_raises_runtime_error_not_json_error(self, tmp_path):
        """When the server returns HTML (e.g. a WAF page), upload raises RuntimeError not JSONDecodeError."""
        dummy_file = tmp_path / "model.onnx"
        dummy_file.write_bytes(b"dummy")

        hub = self._make_hub()

        html_response = MagicMock()
        html_response.status_code = 403
        html_response.json.side_effect = ValueError("No JSON")
        html_response.text = "<html>Forbidden</html>"

        with patch("opengradient.client.model_hub.requests.post", return_value=html_response):
            with patch("opengradient.client.model_hub.MultipartEncoder"):
                with pytest.raises(RuntimeError, match="Upload failed"):
                    hub.upload(str(dummy_file), "my-model", "1.0")

    def test_json_error_response_raises_runtime_error(self, tmp_path):
        """When the server returns JSON error, upload raises RuntimeError with detail message."""
        dummy_file = tmp_path / "model.onnx"
        dummy_file.write_bytes(b"dummy")

        hub = self._make_hub()

        json_response = MagicMock()
        json_response.status_code = 422
        json_response.json.return_value = {"detail": "Unprocessable entity"}
        json_response.text = '{"detail": "Unprocessable entity"}'

        with patch("opengradient.client.model_hub.requests.post", return_value=json_response):
            with patch("opengradient.client.model_hub.MultipartEncoder"):
                with pytest.raises(RuntimeError, match="Unprocessable entity"):
                    hub.upload(str(dummy_file), "my-model", "1.0")


# --- Fix #5: event_data type guard in convert_to_model_output ---


class TestConvertToModelOutputGuard:
    """convert_to_model_output must raise TypeError for non-dict input."""

    def test_none_input_raises_type_error(self):
        """Passing None raises TypeError with a clear message."""
        with pytest.raises(TypeError, match="event_data must be a dict-like object"):
            convert_to_model_output(None)

    def test_string_input_raises_type_error(self):
        """Passing a string raises TypeError."""
        with pytest.raises(TypeError, match="event_data must be a dict-like object"):
            convert_to_model_output("not a dict")

    def test_valid_empty_dict_returns_empty(self):
        """An empty dict returns an empty output dict without crashing."""
        result = convert_to_model_output({})
        assert result == {}


# --- Fix #16: contract address validation in Alpha constructor ---


class TestAlphaAddressValidation:
    """Alpha constructor must reject invalid Ethereum addresses immediately."""

    def _make_alpha(self, address):
        with patch("opengradient.client.alpha.Web3") as mock_web3_cls:
            mock_w3 = MagicMock()
            mock_web3_cls.return_value = mock_w3
            mock_web3_cls.HTTPProvider.return_value = MagicMock()
            mock_web3_cls.is_address.side_effect = lambda a: a.startswith("0x") and len(a) == 42
            mock_w3.eth.account.from_key.return_value = MagicMock(address="0xDEAD")
            return Alpha(private_key="0x" + "a" * 64, inference_contract_address=address)

    def test_invalid_address_raises_value_error(self):
        """A clearly wrong address raises ValueError at construction time."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            self._make_alpha("not-an-address")

    def test_valid_address_does_not_raise(self):
        """A valid checksummed address does not raise."""
        self._make_alpha("0x" + "b" * 40)

    def test_empty_address_raises_value_error(self):
        """An empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            self._make_alpha("")
