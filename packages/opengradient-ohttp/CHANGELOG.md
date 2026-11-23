# Changelog

## 0.1.0rc1

- First release candidate.
- `OHTTPTransport` and `AsyncOHTTPTransport`: httpx transports for the OpenAI SDK that seal chat completions with OHTTP and post them to the DreamFort developer relay (`/api/v1/developer/ohttp`).
- The `Authorization` header comes from the OpenAI client's `api_key`.
- TEE and key selection come from the on-chain registry, cached for 5 minutes. A TEE the relay reports as inactive (`tee_not_active`) is replaced once.
- Single-shot and streaming responses are signature-checked. In a stream, the signed frame is verified before it, and `[DONE]`, are delivered.
- Relay errors map to the OpenAI SDK's typed exceptions. Failures a retry can't fix carry `x-should-retry: false`.
