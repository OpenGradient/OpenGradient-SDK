---
outline: [2,3]
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

### `TEEEndpoint`

A verified TEE with its endpoint URL and TLS certificate from the registry.

#### Constructor

```python
def __init__(
    tee_id: str,
    endpoint: str,
    tls_cert_der: bytes,
    payment_address: str
)
```

#### Variables

* static `endpoint` : str
* static `payment_address` : str
* static `tee_id` : str
* static `tls_cert_der` : bytes

### `TEEInfo`

Mirrors the on-chain TEERegistry.TEEInfo struct.

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
    last_heartbeat_at: int
)
```

#### Variables

* `enabled` : bool - Alias for field number 7
* `endpoint` : str - Alias for field number 2
* `last_heartbeat_at` : int - Alias for field number 9
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
Return the first active LLM proxy TEE from the registry.

**Returns**

TEEEndpoint for an active LLM proxy TEE, or None if none are available.