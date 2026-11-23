"""httpx transports that send OpenAI chat completions through the DreamFort
developer OHTTP relay.

Plug one into the ``http_client`` the OpenAI SDK already accepts::

    client = OpenAI(
        base_url="https://ohttp.invalid/v1",
        api_key="df_sk_...",
        http_client=httpx.Client(transport=OHTTPTransport()),
    )

For each ``POST .../chat/completions`` the transport:

  1. picks a TEE from the on-chain registry (cached; refreshed when chat-api
     says the TEE is no longer active),
  2. HPKE-seals the request to that TEE and POSTs only the ciphertext to
     ``/api/v1/developer/ohttp`` with the caller's ``Authorization`` header,
  3. decrypts the reply and checks the TEE's signature before the OpenAI SDK
     sees any of it (streams: before the signed final frame and ``[DONE]``).

The request's host is ignored; ``base_url`` only has to end in a path the
OpenAI SDK appends ``/chat/completions`` to. Use a ``.invalid`` host so a
client built without this transport fails instead of sending plaintext.
"""

from __future__ import annotations

from typing import AsyncGenerator, AsyncIterator, Generator, Iterable, Iterator, Optional, Union

import httpx

from ._errors import RegistryError, TeeVerificationError, UnsupportedRequestError
from ._hpke import EncapsulatedRequest
from ._protocol import (
    DEFAULT_RELAY_URL,
    DEFAULT_TIMEOUT,
    RELAY_CODE_TEE_NOT_ACTIVE,
    InnerError,
    PreparedCall,
    StreamVerifier,
    error_response,
    is_chat_completions,
    open_single_response,
    outer_request,
    parse_relay_error,
    prepare_call,
    proof_headers,
    relay_error_response,
    relay_url_from,
    seal,
    should_refresh_tee,
)
from ._registry import (
    DEFAULT_EXCLUDED_TEE_IDS,
    DEFAULT_REGISTRY_ADDRESS,
    DEFAULT_RPC_URL,
    DEFAULT_TEE_CACHE_TTL,
    AsyncTeeResolver,
    TeeResolver,
    TeeTarget,
)
from ._version import __version__

_USER_AGENT = f"opengradient-ohttp/{__version__}"
_UNSUPPORTED_ENDPOINT = "Only POST /chat/completions is supported over OHTTP."

TimeoutTypes = Union[httpx.Timeout, float]


def _timeout_dict(timeout: TimeoutTypes) -> dict:
    return (timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)).as_dict()


def _single_response(
    request: httpx.Request,
    call: PreparedCall,
    tee: TeeTarget,
    sealed: EncapsulatedRequest,
    content: bytes,
    content_type: str,
) -> httpx.Response:
    try:
        inner, proof = open_single_response(call, tee, sealed, content, content_type)
    except TeeVerificationError as exc:
        return error_response(
            request,
            502,
            f"TEE response failed verification: {exc}",
            code="tee_verification_failed",
            error_type="tee_verification_error",
        )
    except InnerError as exc:
        return error_response(request, exc.status, exc.message, code="tee_inner_error", error_type="upstream_error")
    return httpx.Response(200, json=inner, headers=proof_headers(proof), request=request)


class _SyncVerifiedStream(httpx.SyncByteStream):
    """Single-pass: iterating again continues where the last iteration stopped.

    The upstream body and the verifier's state can only be consumed once, and
    callers do iterate twice (the OpenAI SDK drains ``response.stream`` after
    ``[DONE]``), so a second ``__iter__`` must not restart from the top.
    """

    def __init__(self, upstream: httpx.Response, verifier: StreamVerifier):
        self._upstream = upstream
        self._verifier = verifier
        self._chunks: Optional[Generator[bytes, None, None]] = None

    def __iter__(self) -> Iterator[bytes]:
        if self._chunks is None:
            self._chunks = self._generate()
        # Not `yield from`: that would close the shared generator when this iteration is abandoned.
        for chunk in self._chunks:
            yield chunk

    def _generate(self) -> Generator[bytes, None, None]:
        try:
            for raw in self._upstream.iter_bytes():
                out = self._verifier.feed(raw)
                if out:
                    yield out
            out = self._verifier.finish()
            if out:
                yield out
        finally:
            self._upstream.close()

    def close(self) -> None:
        if self._chunks is not None:
            self._chunks.close()
        self._upstream.close()


class _AsyncVerifiedStream(httpx.AsyncByteStream):
    """Single-pass, like :class:`_SyncVerifiedStream`."""

    def __init__(self, upstream: httpx.Response, verifier: StreamVerifier):
        self._upstream = upstream
        self._verifier = verifier
        self._chunks: Optional[AsyncGenerator[bytes, None]] = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._chunks is None:
            self._chunks = self._generate()
        async for chunk in self._chunks:
            yield chunk

    async def _generate(self) -> AsyncGenerator[bytes, None]:
        try:
            async for raw in self._upstream.aiter_bytes():
                out = self._verifier.feed(raw)
                if out:
                    yield out
            out = self._verifier.finish()
            if out:
                yield out
        finally:
            await self._upstream.aclose()

    async def aclose(self) -> None:
        if self._chunks is not None:
            await self._chunks.aclose()
        await self._upstream.aclose()


class _Options:
    """Settings shared by the sync and async transports."""

    def __init__(
        self,
        *,
        relay_url: str,
        api_key: Optional[str],
        verify_before_release: bool,
        send_model_hint: bool,
        timeout: TimeoutTypes,
    ):
        self.relay_url = relay_url_from(relay_url)
        self.authorization = f"Bearer {api_key}" if api_key else None
        self.verify_before_release = verify_before_release
        self.send_model_hint = send_model_hint
        self.timeout = _timeout_dict(timeout)

    def build_outer(self, request: httpx.Request, call: PreparedCall, tee: TeeTarget, sealed: EncapsulatedRequest) -> httpx.Request:
        return outer_request(
            relay_url=self.relay_url,
            call=call,
            tee=tee,
            sealed=sealed,
            # An explicit api_key wins; otherwise the key the OpenAI SDK put on the request.
            authorization=self.authorization or request.headers.get("authorization"),
            send_model_hint=self.send_model_hint,
            timeout=request.extensions.get("timeout") or self.timeout,
            user_agent=_USER_AGENT,
        )

    def stream_response(self, request: httpx.Request, tee: TeeTarget, stream: httpx.SyncByteStream | httpx.AsyncByteStream) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-tee-id": tee.tee_id},
            stream=stream,
            request=request,
        )


class OHTTPTransport(httpx.BaseTransport):
    """Sync transport: ``httpx.Client(transport=OHTTPTransport())``.

    Args:
        relay_url: chat-api's developer relay, or just its origin (the
            ``/api/v1/developer/ohttp`` path is added when the URL has none).
        api_key: A ``df_sk_...`` key. Optional: by default the
            ``Authorization`` header the OpenAI SDK sets from its ``api_key`` is used.
        registry_rpc_url: JSON-RPC endpoint of the chain the TEE registry lives on.
        registry_address: The ``TEERegistry`` contract address.
        tee: Pin one TEE instead of reading the registry (self-hosted gateways, tests).
        tee_cache_ttl: Seconds to keep a registry pick.
        exclude_tee_ids: TEE ids never to use (defaults to the apps' list).
        verify_before_release: Streams only: hold all output until the whole
            stream has been verified. Default False: deltas arrive as they're
            decrypted, and the signed final frame is verified before it and
            ``[DONE]`` are passed on.
        send_model_hint: Send ``X-OHTTP-Model`` / ``X-OHTTP-Provider`` so the
            relay's metrics can label the model. Off by default: without it
            the relay can't tell which model you use.
        timeout: Used when the request carries no timeout of its own (the
            OpenAI SDK always sets one).
        transport: The transport for the outer hop to chat-api. Default
            ``httpx.HTTPTransport()`` (TLS verified). Pass your own for
            proxies or tests.
        registry_transport: The transport for registry RPC calls (tests).
    """

    def __init__(
        self,
        *,
        relay_url: str = DEFAULT_RELAY_URL,
        api_key: Optional[str] = None,
        registry_rpc_url: str = DEFAULT_RPC_URL,
        registry_address: str = DEFAULT_REGISTRY_ADDRESS,
        tee: Optional[TeeTarget] = None,
        tee_cache_ttl: float = DEFAULT_TEE_CACHE_TTL,
        exclude_tee_ids: Iterable[str] = DEFAULT_EXCLUDED_TEE_IDS,
        verify_before_release: bool = False,
        send_model_hint: bool = False,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
        registry_transport: Optional[httpx.BaseTransport] = None,
    ):
        self._options = _Options(
            relay_url=relay_url,
            api_key=api_key,
            verify_before_release=verify_before_release,
            send_model_hint=send_model_hint,
            timeout=timeout,
        )
        self._resolver = TeeResolver(
            rpc_url=registry_rpc_url,
            registry_address=registry_address,
            ttl=tee_cache_ttl,
            excluded_tee_ids=exclude_tee_ids,
            pinned=tee,
            transport=registry_transport,
        )
        self._inner = transport if transport is not None else httpx.HTTPTransport()

    @property
    def relay_url(self) -> str:
        return self._options.relay_url

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not is_chat_completions(request):
            return error_response(request, 404, _UNSUPPORTED_ENDPOINT, code="unsupported_endpoint", error_type="invalid_request_error")
        try:
            call = prepare_call(request.read())
        except UnsupportedRequestError as exc:
            return error_response(request, 400, str(exc), code="unsupported_request", error_type="invalid_request_error")

        attempts = 1 if self._resolver.is_pinned else 2
        for attempt in range(attempts):
            try:
                tee = self._resolver.get()
            except RegistryError as exc:
                return error_response(request, 503, str(exc), code="tee_registry_unavailable", retry=None)

            sealed = seal(call, tee)
            outer = self._options.build_outer(request, call, tee, sealed)
            response = self._inner.handle_request(outer)
            response.request = outer

            if response.status_code >= 400:
                try:
                    content = response.read()
                finally:
                    response.close()
                error = parse_relay_error(response.status_code, content, response.headers)
                if should_refresh_tee(error):
                    self._resolver.invalidate(tee.tee_id)
                if error.code == RELAY_CODE_TEE_NOT_ACTIVE and attempt + 1 < attempts:
                    continue  # chat-api didn't forward or bill; reseal to a fresh registry pick
                return relay_error_response(request, error)

            if call.stream:
                verifier = StreamVerifier(call, tee, sealed, verify_before_release=self._options.verify_before_release)
                return self._options.stream_response(request, tee, _SyncVerifiedStream(response, verifier))

            try:
                content = response.read()
            finally:
                response.close()
            return _single_response(request, call, tee, sealed, content, response.headers.get("content-type", ""))

        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        self._inner.close()
        self._resolver.close()


class AsyncOHTTPTransport(httpx.AsyncBaseTransport):
    """Async transport: ``httpx.AsyncClient(transport=AsyncOHTTPTransport())``.

    Same arguments as :class:`OHTTPTransport`; ``transport`` and
    ``registry_transport`` take ``httpx.AsyncBaseTransport`` instances.
    """

    def __init__(
        self,
        *,
        relay_url: str = DEFAULT_RELAY_URL,
        api_key: Optional[str] = None,
        registry_rpc_url: str = DEFAULT_RPC_URL,
        registry_address: str = DEFAULT_REGISTRY_ADDRESS,
        tee: Optional[TeeTarget] = None,
        tee_cache_ttl: float = DEFAULT_TEE_CACHE_TTL,
        exclude_tee_ids: Iterable[str] = DEFAULT_EXCLUDED_TEE_IDS,
        verify_before_release: bool = False,
        send_model_hint: bool = False,
        timeout: TimeoutTypes = DEFAULT_TIMEOUT,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        registry_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._options = _Options(
            relay_url=relay_url,
            api_key=api_key,
            verify_before_release=verify_before_release,
            send_model_hint=send_model_hint,
            timeout=timeout,
        )
        self._resolver = AsyncTeeResolver(
            rpc_url=registry_rpc_url,
            registry_address=registry_address,
            ttl=tee_cache_ttl,
            excluded_tee_ids=exclude_tee_ids,
            pinned=tee,
            transport=registry_transport,
        )
        self._inner = transport if transport is not None else httpx.AsyncHTTPTransport()

    @property
    def relay_url(self) -> str:
        return self._options.relay_url

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not is_chat_completions(request):
            return error_response(request, 404, _UNSUPPORTED_ENDPOINT, code="unsupported_endpoint", error_type="invalid_request_error")
        try:
            call = prepare_call(await request.aread())
        except UnsupportedRequestError as exc:
            return error_response(request, 400, str(exc), code="unsupported_request", error_type="invalid_request_error")

        attempts = 1 if self._resolver.is_pinned else 2
        for attempt in range(attempts):
            try:
                tee = await self._resolver.get()
            except RegistryError as exc:
                return error_response(request, 503, str(exc), code="tee_registry_unavailable", retry=None)

            sealed = seal(call, tee)
            outer = self._options.build_outer(request, call, tee, sealed)
            response = await self._inner.handle_async_request(outer)
            response.request = outer

            if response.status_code >= 400:
                try:
                    content = await response.aread()
                finally:
                    await response.aclose()
                error = parse_relay_error(response.status_code, content, response.headers)
                if should_refresh_tee(error):
                    self._resolver.invalidate(tee.tee_id)
                if error.code == RELAY_CODE_TEE_NOT_ACTIVE and attempt + 1 < attempts:
                    continue
                return relay_error_response(request, error)

            if call.stream:
                verifier = StreamVerifier(call, tee, sealed, verify_before_release=self._options.verify_before_release)
                return self._options.stream_response(request, tee, _AsyncVerifiedStream(response, verifier))

            try:
                content = await response.aread()
            finally:
                await response.aclose()
            return _single_response(request, call, tee, sealed, content, response.headers.get("content-type", ""))

        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        await self._inner.aclose()
        await self._resolver.aclose()
