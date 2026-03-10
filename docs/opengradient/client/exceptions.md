---
outline: [2,3]
---

[opengradient](../index) / [client](./index) / exceptions

# Package opengradient.client.exceptions

Exception types for OpenGradient SDK errors.

## Classes

### `AuthenticationError`

Raised when there's an authentication error

#### Constructor

```python
def __init__(message='Authentication failed', **kwargs)
```

### `FileNotFoundError`

Raised when a file is not found

#### Constructor

```python
def __init__(file_path)
```

### `InferenceError`

Raised when there's an error during inference

#### Constructor

```python
def __init__(message, model_cid=None, **kwargs)
```

### `InsufficientCreditsError`

Raised when the user has insufficient credits for the operation

#### Constructor

```python
def __init__(
    message='Insufficient credits',
    required_credits=None,
    available_credits=None,
    **kwargs
)
```

### `InvalidInputError`

Raised when invalid input is provided

#### Constructor

```python
def __init__(message, invalid_fields=None, **kwargs)
```

### `NetworkError`

Raised when a network error occurs

#### Constructor

```python
def __init__(message, status_code=None, response=None)
```

### `OpenGradientError`

Base exception for OpenGradient SDK

#### Constructor

```python
def __init__(message, status_code=None, response=None)
```

#### Subclasses

* `AuthenticationError`
* `FileNotFoundError`
* `InferenceError`
* `InsufficientCreditsError`
* `InvalidInputError`
* `NetworkError`
* `RateLimitError`
* `ResultRetrievalError`
* `ServerError`
* `TimeoutError`
* `UnsupportedModelError`
* `UploadError`

### `RateLimitError`

Raised when API rate limit is exceeded

#### Constructor

```python
def __init__(message='Rate limit exceeded', retry_after=None, **kwargs)
```

### `ResultRetrievalError`

Raised when there's an error retrieving results

#### Constructor

```python
def __init__(message, inference_cid=None, **kwargs)
```

### `ServerError`

Raised when a server error occurs

#### Constructor

```python
def __init__(message, status_code=None, response=None)
```

### `TimeoutError`

Raised when a request times out

#### Constructor

```python
def __init__(message='Request timed out', timeout=None, **kwargs)
```

### `UnsupportedModelError`

Raised when an unsupported model type is used

#### Constructor

```python
def __init__(model_type)
```

### `UploadError`

Raised when there's an error during file upload

#### Constructor

```python
def __init__(message, file_path=None, **kwargs)
```