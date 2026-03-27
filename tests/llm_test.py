"""Tests for LLM class.

Construction patches the x402 boundary (x402HttpxClient, EthAccountSigner, etc.)
so LLM builds normally — no test-only constructor params, no mocking of private methods.
"""

import json
import ssl
from contextlib import asynccontextmanager
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.opengradient.client.llm import LLM
from src.opengradient.client.tee_connection import TEEConnection
from src.opengradient.types import TEE_LLM, x402SettlementMode

# ── Fake HTTP transport ──────────────────────────────────────────────


class FakeHTTPClient:
    """Stands in for x402HttpxClient.

    Configured per-test with set_response / set_stream_response, then
    injected via the x402HttpxClient patch so LLM's normal __init__
    assigns it to self._http_client.
    """

    def __init__(self, *_args, **_kwargs):
        self._response_status: int = 200
        self._response_body: bytes = b"{}"
        self._post_calls: List[dict] = []
        self._stream_response = None
        self._error_on_next: BaseException | None = None
        self._stream_error_on_next: BaseException | None = None

    def set_response(self, status_code: int, body: dict) -> None:
        self._response_status = status_code
        self._response_body = json.dumps(body).encode()

    def set_stream_response(self, status_code: int, chunks: List[bytes]) -> None:
        self._stream_response = _FakeStreamResponse(status_code, chunks)

    @property
    def post_calls(self) -> List[dict]:
        return self._post_calls

    def fail_next_post(self, exc: BaseException) -> None:
        """Make the next post() call raise *exc*, then revert to normal."""
        self._error_on_next = exc

    def fail_next_stream(self, exc: BaseException) -> None:
        """Make the next stream() call raise *exc*, then revert to normal."""
        self._stream_error_on_next = exc

    async def post(self, url: str, *, json=None, headers=None, timeout=None) -> "_FakeResponse":
        self._post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._error_on_next is not None:
            exc, self._error_on_next = self._error_on_next, None
            raise exc
        resp = _FakeResponse(self._response_status, self._response_body)
        if self._response_status >= 400:
            resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock()))
        return resp

    @asynccontextmanager
    async def stream(self, method: str, url: str, *, json=None, headers=None, timeout=None):
        self._post_calls.append({"method": method, "url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._stream_error_on_next is not None:
            exc, self._stream_error_on_next = self._stream_error_on_next, None
            raise exc
        yield self._stream_response

    async def aclose(self):
        pass


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        pass

    async def aread(self) -> bytes:
        return self._body


class _FakeStreamResponse:
    def __init__(self, status_code: int, chunks: List[bytes]):
        self.status_code = status_code
        self._chunks = chunks

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return b"".join(self._chunks)


# ── Fixture: construct LLM through its normal path ───────────────────

# Patch the external x402/signer libs at the module where they're imported,
# so LLM.__init__ runs its real code but gets our FakeHTTPClient.

_PATCHES = {
    "x402_httpx": "src.opengradient.client.tee_connection.x402HttpxClient",
    "x402_client": "src.opengradient.client.llm.x402Client",
    "signer": "src.opengradient.client.llm.EthAccountSigner",
    "register_exact": "src.opengradient.client.llm.register_exact_evm_client",
    "register_upto": "src.opengradient.client.llm.register_upto_evm_client",
}


@pytest.fixture
def fake_http():
    """Patch x402 externals and return the FakeHTTPClient that LLM will use."""
    http = FakeHTTPClient()

    with (
        patch(_PATCHES["x402_httpx"], return_value=http),
        patch(_PATCHES["x402_client"]),
        patch(_PATCHES["signer"]),
        patch(_PATCHES["register_exact"]),
        patch(_PATCHES["register_upto"]),
    ):
        yield http


FAKE_PRIVATE_KEY = "0x" + "a" * 64


def _make_llm(
    endpoint: str = "https://test.tee.server",
) -> LLM:
    """Build an LLM with an explicit server URL (skips registry lookup)."""
    llm = LLM(private_key=FAKE_PRIVATE_KEY, llm_server_url=endpoint)
    # llm_server_url path sets tee_id/payment_address to None; replace with test values.
    from dataclasses import replace
    llm._tee._active = replace(llm._tee.get(), tee_id="test-tee-id", payment_address="0xTestPayment")
    return llm


# ── Completion tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCompletion:
    async def test_returns_completion_output(self, fake_http):
        fake_http.set_response(
            200,
            {
                "completion": "Hello world",
                "tee_signature": "sig-abc",
                "tee_timestamp": "2025-01-01T00:00:00Z",
            },
        )
        llm = _make_llm()

        result = await llm.completion(model=TEE_LLM.GPT_5, prompt="Say hello")

        assert result.completion_output == "Hello world"
        assert result.tee_signature == "sig-abc"
        assert result.tee_timestamp == "2025-01-01T00:00:00Z"
        assert result.transaction_hash == "external"
        assert result.tee_id == "test-tee-id"
        assert result.tee_payment_address == "0xTestPayment"

    async def test_sends_correct_payload(self, fake_http):
        fake_http.set_response(200, {"completion": "ok"})
        llm = _make_llm()

        await llm.completion(
            model=TEE_LLM.GPT_5,
            prompt="Hello",
            max_tokens=50,
            temperature=0.5,
            stop_sequence=["END"],
        )

        assert len(fake_http.post_calls) == 1
        payload = fake_http.post_calls[0]["json"]
        assert payload["model"] == "gpt-5"
        assert payload["prompt"] == "Hello"
        assert payload["max_tokens"] == 50
        assert payload["temperature"] == 0.5
        assert payload["stop"] == ["END"]

    async def test_sends_to_completion_endpoint(self, fake_http):
        fake_http.set_response(200, {"completion": "ok"})
        llm = _make_llm(endpoint="https://my.server")

        await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")

        assert fake_http.post_calls[0]["url"] == "https://my.server/v1/completions"

    async def test_stop_sequence_omitted_when_none(self, fake_http):
        fake_http.set_response(200, {"completion": "ok"})
        llm = _make_llm()

        await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")

        payload = fake_http.post_calls[0]["json"]
        assert "stop" not in payload

    async def test_settlement_mode_header(self, fake_http):
        fake_http.set_response(200, {"completion": "ok"})
        llm = _make_llm()

        await llm.completion(
            model=TEE_LLM.GPT_5,
            prompt="Hi",
            x402_settlement_mode=x402SettlementMode.PRIVATE,
        )

        headers = fake_http.post_calls[0]["headers"]
        assert headers["X-SETTLEMENT-TYPE"] == "private"

    async def test_http_error_raises_opengradient_error(self, fake_http):
        fake_http.set_response(500, {"error": "boom"})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="TEE LLM completion failed"):
            await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")


# ── Chat (non-streaming) tests ───────────────────────────────────────


@pytest.mark.asyncio
class TestChat:
    async def test_returns_chat_output(self, fake_http):
        fake_http.set_response(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "Hi there!"}, "finish_reason": "stop"}],
                "tee_signature": "sig-xyz",
                "tee_timestamp": "2025-06-01T00:00:00Z",
            },
        )
        llm = _make_llm()

        result = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.chat_output["content"] == "Hi there!"
        assert result.chat_output["role"] == "assistant"
        assert result.finish_reason == "stop"
        assert result.tee_signature == "sig-xyz"

    async def test_flattens_content_blocks(self, fake_http):
        fake_http.set_response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": "world"},
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
        llm = _make_llm()

        result = await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

        assert result.chat_output["content"] == "Hello world"

    async def test_sends_correct_payload(self, fake_http):
        fake_http.set_response(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )
        llm = _make_llm()

        await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=200,
            temperature=0.7,
            stop_sequence=["STOP"],
        )

        payload = fake_http.post_calls[0]["json"]
        assert payload["model"] == "gpt-5"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["max_tokens"] == 200
        assert payload["temperature"] == 0.7
        assert payload["stop"] == ["STOP"]
        assert "stream" not in payload

    async def test_sends_to_chat_endpoint(self, fake_http):
        fake_http.set_response(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )
        llm = _make_llm(endpoint="https://my.server")

        await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

        assert fake_http.post_calls[0]["url"] == "https://my.server/v1/chat/completions"

    async def test_tools_included_in_payload(self, fake_http):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        fake_http.set_response(
            200,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        llm = _make_llm()

        result = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Weather?"}],
            tools=tools,
            tool_choice="required",
        )

        payload = fake_http.post_calls[0]["json"]
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "required"
        assert result.chat_output["tool_calls"] == [{"id": "1"}]

    async def test_tool_choice_defaults_to_auto(self, fake_http):
        tools = [{"type": "function", "function": {"name": "f"}}]
        fake_http.set_response(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )
        llm = _make_llm()

        await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}], tools=tools)

        payload = fake_http.post_calls[0]["json"]
        assert payload["tool_choice"] == "auto"

    async def test_empty_choices_raises(self, fake_http):
        fake_http.set_response(200, {"choices": []})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="'choices' missing or empty"):
            await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

    async def test_missing_choices_raises(self, fake_http):
        fake_http.set_response(200, {"result": "no choices key"})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="'choices' missing or empty"):
            await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

    async def test_http_error_raises_opengradient_error(self, fake_http):
        fake_http.set_response(500, {"error": "internal"})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="TEE LLM chat failed"):
            await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])


# ── Streaming tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestChatStreaming:
    async def test_streams_chunks(self, fake_http):
        fake_http.set_stream_response(
            200,
            [
                b'data: {"model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
                b'data: {"model":"gpt-5","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}],"tee_signature":"sig"}\n\n',
                b"data: [DONE]\n\n",
            ],
        )
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        chunks = [chunk async for chunk in gen]
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hi"
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[1].choices[0].delta.content == " there"
        assert chunks[1].choices[0].finish_reason == "stop"

    async def test_stream_payload_includes_stream_flag(self, fake_http):
        fake_http.set_stream_response(200, [b"data: [DONE]\n\n"])
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        _ = [chunk async for chunk in gen]

        payload = fake_http.post_calls[0]["json"]
        assert payload["stream"] is True

    async def test_stream_sets_tee_metadata_on_final_chunk(self, fake_http):
        fake_http.set_stream_response(
            200,
            [
                b'data: {"model":"gpt-5","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
        )
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in gen]

        final = chunks[-1]
        assert final.is_final
        assert final.tee_id == "test-tee-id"
        assert final.tee_payment_address == "0xTestPayment"

    async def test_stream_error_raises(self, fake_http):
        fake_http.set_stream_response(500, [b"Internal Server Error"])
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        with pytest.raises(RuntimeError, match="streaming request failed"):
            _ = [chunk async for chunk in gen]

    async def test_tools_with_stream_falls_back_to_single_chunk(self, fake_http):
        """When tools + stream=True, LLM falls back to non-streaming and yields one chunk."""
        tools = [{"type": "function", "function": {"name": "f"}}]
        fake_http.set_response(
            200,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Weather?"}],
            tools=tools,
            stream=True,
        )
        chunks = [chunk async for chunk in gen]

        assert len(chunks) == 1
        assert chunks[0].is_final
        assert chunks[0].choices[0].delta.tool_calls == [{"id": "tc1"}]
        assert chunks[0].choices[0].finish_reason == "tool_calls"


# ── ensure_opg_approval tests ────────────────────────────────────────


class TestEnsureOpgApproval:
    def test_rejects_amount_below_minimum(self, fake_http):
        llm = _make_llm()

        with pytest.raises(ValueError, match="at least"):
            llm.ensure_opg_approval(opg_amount=0.01)


# ── Lifecycle tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestLifecycle:
    async def test_close_delegates_to_http_client(self, fake_http):
        llm = _make_llm()

        await llm.close()
        # FakeHTTPClient.aclose is a no-op; just verify it doesn't blow up.


# ── TEE resolution tests ─────────────────────────────────────────────


class TestResolveTeE:
    def test_explicit_url_skips_registry(self):
        endpoint, cert, tee_id, pay_addr = TEEConnection._resolve_tee("https://explicit.url", None)

        assert endpoint == "https://explicit.url"
        assert cert is None
        assert tee_id is None
        assert pay_addr is None

    def test_missing_rpc_and_registry_raises(self):
        with pytest.raises(ValueError):
            TEEConnection._resolve_tee(None, None)

    def test_missing_registry_address_raises(self):
        with pytest.raises(ValueError):
            TEEConnection._resolve_tee(None, None)

    def test_registry_returns_none_raises(self):
        mock_reg = MagicMock()
        mock_reg.get_llm_tee.return_value = None

        with pytest.raises(ValueError, match="No active LLM proxy TEE"):
            TEEConnection._resolve_tee(None, mock_reg)

    def test_registry_success(self):
        mock_reg = MagicMock()
        mock_tee = MagicMock()
        mock_tee.endpoint = "https://registry.tee"
        mock_tee.tls_cert_der = b"cert-bytes"
        mock_tee.tee_id = "tee-42"
        mock_tee.payment_address = "0xPay"
        mock_reg.get_llm_tee.return_value = mock_tee

        endpoint, cert, tee_id, pay_addr = TEEConnection._resolve_tee(None, mock_reg)

        assert endpoint == "https://registry.tee"
        assert cert == b"cert-bytes"
        assert tee_id == "tee-42"
        assert pay_addr == "0xPay"


# ── TEE retry tests (non-streaming) ──────────────────────────────────


@pytest.mark.asyncio
class TestTeeRetryCompletion:
    async def test_retries_on_connection_error_and_succeeds(self, fake_http):
        """First call hits connection error → refresh TEE → second call succeeds."""
        fake_http.set_response(200, {"completion": "retried ok", "tee_signature": "s", "tee_timestamp": "t"})
        fake_http.fail_next_post(ConnectionError("connection refused"))
        llm = _make_llm()

        result = await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")

        assert result.completion_output == "retried ok"
        assert len(fake_http.post_calls) == 2

    async def test_http_status_error_not_retried(self, fake_http):
        """A server-side error (HTTP 500) should not trigger a TEE retry."""
        fake_http.set_response(500, {"error": "boom"})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="TEE LLM completion failed"):
            await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")
        assert len(fake_http.post_calls) == 1

    async def test_second_failure_propagates(self, fake_http):
        """If the retry also fails, the error should propagate."""
        call_count = 0

        async def always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("still broken")

        fake_http.post = always_fail
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="TEE LLM completion failed"):
            await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")
        assert call_count == 2


@pytest.mark.asyncio
class TestTeeRetryChat:
    async def test_retries_on_connection_error_and_succeeds(self, fake_http):
        fake_http.set_response(
            200,
            {"choices": [{"message": {"role": "assistant", "content": "retry ok"}, "finish_reason": "stop"}]},
        )
        fake_http.fail_next_post(OSError("network unreachable"))
        llm = _make_llm()

        result = await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

        assert result.chat_output["content"] == "retry ok"
        assert len(fake_http.post_calls) == 2

    async def test_http_status_error_not_retried(self, fake_http):
        fake_http.set_response(500, {"error": "internal"})
        llm = _make_llm()

        with pytest.raises(RuntimeError, match="TEE LLM chat failed"):
            await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])
        assert len(fake_http.post_calls) == 1


# ── TEE retry tests (streaming) ──────────────────────────────────────


@pytest.mark.asyncio
class TestTeeRetryStreaming:
    async def test_retries_stream_on_connection_error_before_chunks(self, fake_http):
        """Connection failure during stream setup (no chunks yielded) → retry succeeds."""
        fake_http.set_stream_response(
            200,
            [
                b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
        )
        fake_http.fail_next_stream(ConnectionError("reset by peer"))
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [c async for c in gen]

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "ok"
        assert len(fake_http.post_calls) == 2

    async def test_no_retry_after_chunks_yielded(self, fake_http):
        """Failure AFTER chunks were yielded must raise, not retry."""

        class _FailMidStream:
            def __init__(self):
                self.status_code = 200

            async def aiter_raw(self):
                yield b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
                raise ConnectionError("mid-stream disconnect")

            async def aread(self) -> bytes:
                return b""

        fake_http._stream_response = _FailMidStream()
        llm = _make_llm()

        gen = await llm.chat(
            model=TEE_LLM.GPT_5,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        with pytest.raises(ConnectionError):
            _ = [c async for c in gen]

        assert len(fake_http.post_calls) == 1


# ── TEE reconnect tests ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestReconnect:
    async def test_replaces_http_client(self):
        """After reconnect, the http client should be a new instance."""
        clients_created = []

        def make_client(*args, **kwargs):
            c = FakeHTTPClient()
            clients_created.append(c)
            return c

        with (
            patch(_PATCHES["x402_httpx"], side_effect=make_client),
            patch(_PATCHES["x402_client"]),
            patch(_PATCHES["signer"]),
            patch(_PATCHES["register_exact"]),
            patch(_PATCHES["register_upto"]),
        ):
            llm = _make_llm()
            old_client = llm._tee.get().http_client

            await llm._tee.reconnect()

            assert llm._tee.get().http_client is not old_client
            assert len(clients_created) == 2  # init + refresh

    async def test_closes_old_client(self, fake_http):
        llm = _make_llm()
        old_client = llm._tee.get().http_client
        old_client.aclose = AsyncMock()

        await llm._tee.reconnect()

        old_client.aclose.assert_awaited_once()

    async def test_close_failure_is_swallowed(self, fake_http):
        llm = _make_llm()
        old_client = llm._tee.get().http_client
        old_client.aclose = AsyncMock(side_effect=OSError("already closed"))

        # Should not raise
        await llm._tee.reconnect()


# ── TEE cert rotation (crash + re-register) tests ────────────────────


@pytest.mark.asyncio
class TestTeeCertRotation:
    """Simulate a TEE crashing and a new one registering at the same IP
    with a different ephemeral TLS certificate.  The old cert is now
    invalid, so the first request fails with SSLCertVerificationError.
    The retry should re-resolve from the registry (getting the new cert)
    and succeed."""

    async def test_ssl_verification_failure_triggers_tee_refresh_completion(self, fake_http):
        fake_http.set_response(200, {"completion": "ok after refresh", "tee_signature": "s", "tee_timestamp": "t"})
        fake_http.fail_next_post(ssl.SSLCertVerificationError("certificate verify failed"))
        llm = _make_llm()

        with patch.object(llm._tee, "_connect", wraps=llm._tee._connect) as spy:
            result = await llm.completion(model=TEE_LLM.GPT_5, prompt="Hi")

        # _connect was called once during the retry (reconnect)
        spy.assert_called_once()
        assert result.completion_output == "ok after refresh"
        assert len(fake_http.post_calls) == 2

    async def test_ssl_verification_failure_triggers_tee_refresh_chat(self, fake_http):
        fake_http.set_response(
            200,
            {"choices": [{"message": {"role": "assistant", "content": "ok after refresh"}, "finish_reason": "stop"}]},
        )
        fake_http.fail_next_post(ssl.SSLCertVerificationError("certificate verify failed"))
        llm = _make_llm()

        with patch.object(llm._tee, "_connect", wraps=llm._tee._connect) as spy:
            result = await llm.chat(model=TEE_LLM.GPT_5, messages=[{"role": "user", "content": "Hi"}])

        spy.assert_called_once()
        assert result.chat_output["content"] == "ok after refresh"
        assert len(fake_http.post_calls) == 2

    async def test_ssl_verification_failure_triggers_tee_refresh_streaming(self, fake_http):
        fake_http.set_stream_response(
            200,
            [
                b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
        )
        fake_http.fail_next_stream(ssl.SSLCertVerificationError("certificate verify failed"))
        llm = _make_llm()

        with patch.object(llm._tee, "_connect", wraps=llm._tee._connect) as spy:
            gen = await llm.chat(
                model=TEE_LLM.GPT_5,
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
            chunks = [c async for c in gen]

        spy.assert_called_once()
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "ok"
        assert len(fake_http.post_calls) == 2
