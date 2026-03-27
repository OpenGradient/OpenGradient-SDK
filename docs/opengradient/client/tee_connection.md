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
def __init__(x402_client: `x402Client`, registry: [TEERegistry](./tee_registry))
```

**Arguments**

* **`x402_client`**: Configured x402 payment client for creating HTTP clients.
* **`registry`**: TEERegistry for looking up active TEEs.

#### Methods

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

### `StaticTEEConnection`

TEE connection with a hardcoded endpoint URL.

No registry lookup, no background refresh. TLS certificate verification
is disabled because self-hosted TEE servers typically use self-signed certs.

#### Constructor

```python
def __init__(x402_client: `x402Client`, endpoint: str)
```

**Arguments**

* **`x402_client`**: Configured x402 payment client for creating HTTP clients.
* **`endpoint`**: The TEE endpoint URL to connect to.

#### Methods

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

### `TEEConnectionInterface`

Interface for TEE connection implementations.

#### Constructor

```python
def __init__(*args, **kwargs)
```

#### Methods

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