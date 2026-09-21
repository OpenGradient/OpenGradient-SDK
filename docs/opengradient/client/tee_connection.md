---
outline: [2,4]
---

[opengradient](../index) / [client](./index) / tee_connection

# Package opengradient.client.tee_connection

Manages the lifecycle of a connection to a TEE endpoint.

## Classes

### `ActiveTEE`

Snapshot of the currently connected TEE.

#### Constructor

```python
def __init__(
    endpoint: str,
    http_client: `x402HttpxClient`,
    tee_id: Optional[str],
    payment_address: Optional[str]
)
```

#### Methods

---

#### `metadata()`

```python
def metadata(self) ‑> Dict
```
Return TEE metadata dict for decorating responses.

#### Variables

* static `endpoint` : str
* static `http_client` : `x402HttpxClient`
* static `payment_address` : Optional[str]
* static `tee_id` : Optional[str]

### `RegistryTEEConnection`

TEE connection resolved from the on-chain registry.

Handles TLS certificate pinning, background health checks, and automatic
failover when the current TEE becomes unavailable.

#### Constructor

```python
def __init__(x402_client: `x402Client`, registry: `TEERegistry`)
```

**Arguments**

* **`x402_client`**: Configured x402 payment client for creating HTTP clients.
* **`registry`**: TEERegistry for looking up active TEEs.

#### Methods

---

#### `aresolve()`

```python
async def aresolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```
Event-loop-safe ``resolve`` for per-request use in async backends.

``resolve`` scans the registry with a blocking web3 call whenever the
requested TEE is not the active one, which stalls the event loop when
called per request. This variant runs the scan in a worker thread and
caches each pinned id's outcome — found or not-active — for
``_TEE_RESOLVE_TTL`` seconds, so steady traffic costs at most one
chain RPC per TTL window per TEE id, and concurrent cold lookups
collapse into a single scan. It also starts the background refresh
loop, so long-running relays fail over when the active TEE is retired
from the registry.

**Raises**

* **`ValueError`**: If the requested TEE id is not active in the registry
        (the miss may be cached for up to ``_TEE_RESOLVE_TTL`` seconds).

---

#### `close()`

```python
async def close(self) ‑> None
```
Cancel the background refresh loop and close the HTTP client.

---

#### `ensure_refresh_loop()`

```python
def ensure_refresh_loop(self) ‑> None
```
Start the background TEE refresh loop if not already running.

Called lazily from async request methods since ``__init__`` is synchronous.

---

#### `get()`

```python
def get(self) ‑> `ActiveTEE`
```
Return a snapshot of the current TEE connection.

---

#### `reconnect()`

```python
async def reconnect(self) ‑> None
```
Connect to a new TEE from the registry and rebuild the HTTP client.

The registry lookup is a blocking web3 call, so it runs in a worker
thread rather than on the event loop. A failed reconnect keeps the
previous connection.

---

#### `resolve()`

```python
def resolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```
Resolve a TEE connection, optionally pinned to an active TEE id.

Backend OHTTP relays can use this when the browser encrypted to a
specific on-chain TEE config, while the backend still owns x402 payment.

### `StaticTEEConnection`

TEE connection with a hardcoded endpoint URL.

Intended for self-hosted development only. No registry lookup, no
background refresh, and TLS certificate verification is disabled
(``verify=False``) because self-hosted TEE servers typically use
self-signed certs.

#### Constructor

```python
def __init__(x402_client: `x402Client`, endpoint: str)
```

**Arguments**

* **`x402_client`**: Configured x402 payment client for creating HTTP clients.
* **`endpoint`**: The TEE endpoint URL to connect to.

#### Methods

---

#### `aresolve()`

```python
async def aresolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```
Async variant of ``resolve``; static connections never do I/O.

---

#### `close()`

```python
async def close(self) ‑> None
```
Close the HTTP client.

---

#### `ensure_refresh_loop()`

```python
def ensure_refresh_loop(self) ‑> None
```
No-op — static connections don't refresh.

---

#### `get()`

```python
def get(self) ‑> `ActiveTEE`
```
Return a snapshot of the current TEE connection.

---

#### `reconnect()`

```python
async def reconnect(self) ‑> None
```
Rebuild the HTTP client (same endpoint).

---

#### `resolve()`

```python
def resolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```
Return the static connection.

Static/dev connections do not have a registry to validate selected
TEE ids against, so they always resolve to the configured endpoint.

### `TEEConnectionInterface`

Interface for TEE connection implementations.

#### Constructor

```python
def __init__(*args, **kwargs)
```

#### Methods

---

#### `aresolve()`

```python
async def aresolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```

---

#### `close()`

```python
async def close(self) ‑> None
```

---

#### `ensure_refresh_loop()`

```python
def ensure_refresh_loop(self) ‑> None
```

---

#### `get()`

```python
def get(self) ‑> `ActiveTEE`
```

---

#### `reconnect()`

```python
async def reconnect(self) ‑> None
```

---

#### `resolve()`

```python
def resolve(self, tee_id: Optional[str] = None) ‑> `ActiveTEE`
```