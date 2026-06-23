---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / tee_registry

# Package opengradient.client.tee_registry

TEE Registry client for fetching verified TEE endpoints and TLS certificates.

## Functions

---

### `build_ssl_context_from_der()`

```python
def build_ssl_context_from_der(der_cert: bytes) ‑> `SSLContext`
```
Build an ssl.SSLContext that trusts *only* the given DER-encoded certificate.

Hostname verification is disabled because TEE servers are typically addressed
by IP while the cert may be issued for a different hostname.  The pinned
certificate itself is the trust anchor — only that cert is accepted.

**Arguments**

* **`der_cert`**: DER-encoded X.509 certificate bytes as stored in the registry.

**Returns**

ssl.SSLContext configured to accept only the pinned certificate.

## Classes

### `OhttpConfig`

Mirrors the on-chain TEERegistry.OhttpConfig struct.

The HPKE key material a client needs to encrypt an Oblivious HTTP request to
this TEE (the same configuration the chat-app browser client reads).

**Attributes**

* **`key_id`**: OHTTP key configuration id.
* **`kem_id`**: HPKE KEM id (0x0020 = DHKEM(X25519, HKDF-SHA256)).
* **`kdf_id`**: HPKE KDF id (0x0001 = HKDF-SHA256).
* **`aead_id`**: HPKE AEAD id (0x0003 = ChaCha20-Poly1305).
* **`public_key`**: The TEE's HPKE (X25519) recipient public key.
* **`key_config`**: The serialized OHTTP key config blob.
* **`registered_at`**: Block timestamp the OHTTP config was registered.

#### Constructor

```python
def __init__(
    key_id: int,
    kem_id: int,
    kdf_id: int,
    aead_id: int,
    public_key: bytes,
    key_config: bytes,
    registered_at: int
)
```

### `TEEEndpoint`

A verified TEE resolved from the registry.

Carries everything needed for both trust paths: the endpoint + pinned TLS
cert for a direct x402 connection, and the OHTTP/HPKE key material +
signing key for the oblivious-HTTP relay path.

**Attributes**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.

#### Constructor

```python
def __init__(
    tee_id: str,
    endpoint: str,
    tls_cert_der: bytes,
    payment_address: str,
    signing_public_key_der: bytes = b'',
    ohttp_config: Optional[`OhttpConfig`] = None,
    pcr_hash: str = ''
)
```

### `TEEInfo`

Mirrors the on-chain TEERegistry.TEEInfo struct (full record).

This is a thin positional wrapper over the tuple web3 decodes from the
contract (built via ``TEEInfo(*raw)``), so every field holds the raw
decoded value. In particular ``ohttp_config`` is the *raw* decoded
sub-tuple, not a parsed `OhttpConfig` — use `_parse_ohttp_config` (as
`TEERegistry` does) to coerce it. The parsed, typed form is surfaced on
`TEEEndpoint.ohttp_config`.

#### Constructor

```python
def __init__(
    owner: str,
    payment_address: str,
    endpoint: str,
    public_key: bytes,
    tls_certificate: bytes,
    pcr_hash: bytes,
    tee_type: int,
    enabled: bool,
    registered_at: int,
    last_heartbeat_at: int,
    ohttp_config: Sequence[Any]
)
```

#### Variables

* `enabled` : bool - Alias for field number 7
* `endpoint` : str - Alias for field number 2
* `last_heartbeat_at` : int - Alias for field number 9
* `ohttp_config` : Sequence[Any] - Alias for field number 10
* `owner` : str - Alias for field number 0
* `payment_address` : str - Alias for field number 1
* `pcr_hash` : bytes - Alias for field number 5
* `public_key` : bytes - Alias for field number 3
* `registered_at` : int - Alias for field number 8
* `tee_type` : int - Alias for field number 6
* `tls_certificate` : bytes - Alias for field number 4

### `TEERegistry`

Queries the on-chain TEE Registry contract to retrieve verified TEE endpoints
and their TLS certificates.

Instead of blindly trusting the TLS certificate presented by a TEE server
(TOFU), this class fetches the certificate that was submitted and verified
during TEE registration.  Any certificate that does not match the one stored
in the registry should be rejected.

#### Constructor

```python
def __init__(rpc_url: str, registry_address: str)
```

**Arguments**

* **`rpc_url`**: RPC endpoint for the chain where the registry is deployed.
* **`registry_address`**: Address of the deployed TEERegistry contract.

#### Methods

---

#### `get_active_tees_by_type()`

```python
def get_active_tees_by_type(self, tee_type: int) ‑> List[`TEEEndpoint`]
```
Return all active TEEs of the given type with their endpoints and TLS certs.

Uses the contract's ``getActiveTEEs(teeType)`` which returns only TEEs that
are enabled, have a valid (non-revoked) PCR, and a fresh heartbeat — all in
a single on-chain call.

**Arguments**

* **`tee_type`**: Integer TEE type (0=LLMProxy, 1=Validator).

**Returns**

List of TEEEndpoint objects for active TEEs of that type.

---

#### `get_llm_tee()`

```python
def get_llm_tee(self) ‑> Optional[`TEEEndpoint`]
```
Return a random active LLM proxy TEE from the registry.

The returned ``TEEEndpoint`` is the full record: endpoint + pinned TLS
cert for direct x402 connections, plus the OHTTP/HPKE ``ohttp_config``
and ``signing_public_key_der`` for the oblivious-HTTP relay path.

**Returns**

TEEEndpoint for an active LLM proxy TEE, or None if none are available.
**`TEEEndpoint` fields:**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.

---

#### `get_llm_tee_ohttp_config()`

```python
def get_llm_tee_ohttp_config(self) ‑> Optional[`TEEEndpoint`]
```
Return a random active LLM proxy TEE that advertises an OHTTP config.

Like ``get_llm_tee`` but skips TEEs missing HPKE key material, so the
result is guaranteed usable for the Oblivious HTTP path.

**Returns**

A TEEEndpoint with a non-empty ``ohttp_config``, or None.
**`TEEEndpoint` fields:**

* **`tee_id`**: keccak256 of the TEE's signing public key (0x-prefixed hex).
* **`endpoint`**: The TEE gateway endpoint URL.
* **`tls_cert_der`**: DER-encoded TLS certificate pinned at registration.
* **`payment_address`**: x402 settlement address for this TEE.
* **`signing_public_key_der`**: DER (SPKI) RSA public key the TEE signs with.
* **`ohttp_config`**: The TEE's OHTTP/HPKE key configuration, if present.
* **`pcr_hash`**: The reproducible-build PCR measurement hash recorded on-chain
        (``0x``-prefixed hex). Lets a caller refuse any TEE whose code
        fingerprint differs from a known-good build — trusting math over the
        registry operator.