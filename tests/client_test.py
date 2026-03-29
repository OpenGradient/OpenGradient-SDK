import json
import logging
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from opengradient.client.alpha import Alpha
from opengradient.client.llm import LLM
from opengradient.client.model_hub import ModelHub
from opengradient.types import (
    StreamChunk,
    TextGenerationStream,
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


# --- Fix #4: choices[] guard ---


class TestChoicesGuard:
    """_chat_request must raise RuntimeError for any malformed choices value."""

    @pytest.mark.asyncio
    async def test_choices_empty_list_raises(self, mock_tee_registry):
        """choices = [] → RuntimeError."""
        llm = LLM(private_key="0x" + "a" * 64)
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aread = AsyncMock(return_value=json.dumps({"choices": []}).encode())

        mock_tee = MagicMock()
        mock_tee.http_client.post = AsyncMock(return_value=mock_response)
        mock_tee.metadata.return_value = {"tee_id": None, "tee_endpoint": None, "tee_payment_address": None}
        llm._tee.get = MagicMock(return_value=mock_tee)
        llm._tee.ensure_refresh_loop = MagicMock()

        from opengradient.types import TEE_LLM
        with pytest.raises(RuntimeError, match="choices"):
            await llm.chat(model=TEE_LLM.CLAUDE_HAIKU_4_5, messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_choices_contains_none_raises(self, mock_tee_registry):
        """choices = [None] passes the old 'not choices' guard but must now raise RuntimeError."""
        llm = LLM(private_key="0x" + "a" * 64)
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aread = AsyncMock(return_value=json.dumps({"choices": [None]}).encode())

        mock_tee = MagicMock()
        mock_tee.http_client.post = AsyncMock(return_value=mock_response)
        mock_tee.metadata.return_value = {"tee_id": None, "tee_endpoint": None, "tee_payment_address": None}
        llm._tee.get = MagicMock(return_value=mock_tee)
        llm._tee.ensure_refresh_loop = MagicMock()

        from opengradient.types import TEE_LLM
        with pytest.raises(RuntimeError, match="choices"):
            await llm.chat(model=TEE_LLM.CLAUDE_HAIKU_4_5, messages=[{"role": "user", "content": "hi"}])


# --- Fix #6: SSE JSON logging ---


class TestSSEJsonLogging:
    """Malformed SSE JSON must emit a warning log, not silently disappear."""

    def test_malformed_sse_logs_warning_in_sync_stream(self):
        """TextGenerationStream.__next__ logs a warning for broken JSON."""
        lines = iter(["data: { broken json \n", "data: [DONE]\n"])
        stream = TextGenerationStream(_iterator=lines, _is_async=False)

        with patch("opengradient.types.logger") as mock_logger:
            try:
                next(stream)
            except StopIteration:
                pass
            mock_logger.warning.assert_called_once()
            assert "Skipping malformed SSE JSON" in mock_logger.warning.call_args[0][0]

    def test_valid_sse_does_not_log_warning(self):
        """Well-formed SSE chunks must not trigger any warning."""
        valid_data = json.dumps({
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        })
        lines = iter([f"data: {valid_data}\n", "data: [DONE]\n"])
        stream = TextGenerationStream(_iterator=lines, _is_async=False)

        with patch("opengradient.types.logger") as mock_logger:
            chunk = next(stream)
            mock_logger.warning.assert_not_called()
            assert chunk.choices[0].delta.content == "hi"


# --- Fix #13 & #14: Alpha exception type and logging ---


class TestAlphaErrorHandling:
    """Alpha.new_workflow raises RuntimeError (not bare Exception) on deployment failure,
    and uses logger instead of print() for non-fatal warnings."""

    @pytest.fixture
    def alpha(self):
        with patch("opengradient.client.alpha.Web3") as mock_web3_cls:
            mock_w3 = MagicMock()
            mock_web3_cls.return_value = mock_w3
            mock_web3_cls.HTTPProvider.return_value = MagicMock()
            mock_web3_cls.to_checksum_address.side_effect = lambda x: x
            mock_w3.eth.account.from_key.return_value = MagicMock(address="0xDEAD")
            mock_w3.eth.get_transaction_count.return_value = 1
            mock_w3.eth.gas_price = 1000000000
            mock_w3.eth.chain_id = 1
            yield Alpha(private_key="0x" + "a" * 64)

    def test_deployment_failure_raises_runtime_error(self, alpha):
        """Status=0 receipt must raise RuntimeError, not bare Exception."""
        mock_contract = MagicMock()
        mock_contract.constructor.return_value.estimate_gas.return_value = 100000
        mock_contract.constructor.return_value.build_transaction.return_value = {}

        fake_receipt = {"status": 0, "transactionHash": b"\xde\xad"}
        alpha._blockchain.eth.contract.return_value = mock_contract
        alpha._blockchain.eth.send_raw_transaction.return_value = b"\xde\xad"
        alpha._blockchain.eth.wait_for_transaction_receipt.return_value = fake_receipt

        from opengradient.types import HistoricalInputQuery, CandleOrder, CandleType
        query = HistoricalInputQuery("BTC", "USDT", 10, 60, CandleOrder.DESCENDING, [CandleType.CLOSE])

        with patch("opengradient.client.alpha.get_abi", return_value=[]):
            with patch("opengradient.client.alpha.get_bin", return_value="0x"):
                with patch("opengradient.client.alpha.run_with_retry", side_effect=lambda fn, *a, **kw: fn()):
                    with pytest.raises(RuntimeError, match="Contract deployment failed"):
                        alpha.new_workflow("Qm123", query, "input")

    def test_gas_estimation_failure_logs_warning(self, alpha):
        """Gas estimation failure must call logger.warning, not print()."""
        mock_contract = MagicMock()
        mock_contract.constructor.return_value.estimate_gas.side_effect = Exception("gas error")
        mock_contract.constructor.return_value.build_transaction.return_value = {}

        fake_receipt = MagicMock()
        fake_receipt.__getitem__ = lambda self, key: 1 if key == "status" else None
        fake_receipt.contractAddress = "0xNEW"
        alpha._blockchain.eth.contract.return_value = mock_contract
        alpha._blockchain.eth.send_raw_transaction.return_value = b"\xca\xfe"
        alpha._blockchain.eth.wait_for_transaction_receipt.return_value = fake_receipt

        from opengradient.types import HistoricalInputQuery, CandleOrder, CandleType
        query = HistoricalInputQuery("BTC", "USDT", 10, 60, CandleOrder.DESCENDING, [CandleType.CLOSE])

        with patch("opengradient.client.alpha.get_abi", return_value=[]):
            with patch("opengradient.client.alpha.get_bin", return_value="0x"):
                with patch("opengradient.client.alpha.run_with_retry", side_effect=lambda fn, *a, **kw: fn()):
                    with patch("opengradient.client.alpha.logger") as mock_logger:
                        alpha.new_workflow("Qm123", query, "input")
                        mock_logger.warning.assert_called_once()
                        assert "Gas estimation failed" in mock_logger.warning.call_args[0][0]

    def test_scheduler_failure_logs_warning(self, alpha):
        """Scheduler registration failure must call logger.warning, not print()."""
        alpha._blockchain.eth.contract.return_value.functions.registerTask.return_value.build_transaction.side_effect = Exception("scheduler error")
        alpha._blockchain.eth.get_transaction_count.return_value = 1

        from opengradient.types import SchedulerParams
        with patch("opengradient.client.alpha.get_abi", return_value=[]):
            with patch("opengradient.client.alpha.logger") as mock_logger:
                alpha._register_with_scheduler("0xCONTRACT", SchedulerParams(frequency=60, duration_hours=1))
                mock_logger.warning.assert_called_once()
                assert "scheduler" in mock_logger.warning.call_args[0][0].lower()
