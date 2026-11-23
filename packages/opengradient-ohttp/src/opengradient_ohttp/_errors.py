"""Exceptions raised by opengradient-ohttp.

Most failures are turned into an HTTP response the OpenAI SDK understands (so
``openai.AuthenticationError``, ``openai.RateLimitError`` and friends work as
usual). These exceptions surface where that isn't possible: from inside a
streaming response, or when the transport is used without the OpenAI SDK.
"""

from __future__ import annotations


class OHTTPError(Exception):
    """Base class for every error this package raises."""


class UnsupportedRequestError(OHTTPError, ValueError):
    """The OpenAI request can't be expressed as a gateway request (bad body, ``n > 1``, ...)."""


class TeeVerificationError(OHTTPError):
    """A response failed TEE verification: bad signature, hash mismatch, tampered ciphertext.

    Treat it as fatal. Content that failed verification must not be shown to
    an end user.
    """


class StreamIncompleteError(OHTTPError):
    """A streamed response ended before the TEE's signed final frame arrived.

    Usually a transient upstream failure; the content received so far is
    unverified.
    """


class RegistryError(OHTTPError):
    """The on-chain TEE registry couldn't be read, or listed no usable TEE."""
