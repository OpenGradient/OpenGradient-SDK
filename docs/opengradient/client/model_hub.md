---
outline: [2,3]
---

[opengradient](../index) / [client](./index) / model_hub

# Package opengradient.client.model_hub

Model Hub for creating, versioning, and uploading ML models.

## Classes

### `ModelHub`

Model Hub namespace.

Provides access to the OpenGradient Model Hub for creating, versioning,
and uploading ML models. Requires email/password authentication.

#### Constructor

```python
def __init__(hub_user: Optional[Dict] = None)
```

#### Methods

---

#### `create_model()`

```python
def create_model(
    self,
    model_name: str,
    model_desc: str,
    version: str = '1.00'
) ‑> `ModelRepository`
```
Create a new model with the given model_name and model_desc, and a specified version.

**Arguments**

* **`model_name (str)`**: The name of the model.
* **`model_desc (str)`**: The description of the model.
* **`version (str)`**: The version identifier (default is "1.00").

**Returns**

dict: The server response containing model details.

**Raises**

* **`CreateModelError`**: If the model creation fails.

---

#### `create_version()`

```python
def create_version(
    self,
    model_name: str,
    notes: str = '',
    is_major: bool = False
) ‑> dict
```
Create a new version for the specified model.

**Arguments**

* **`model_name (str)`**: The unique identifier for the model.
* **`notes (str, optional)`**: Notes for the new version.
* **`is_major (bool, optional)`**: Whether this is a major version update. Defaults to False.

**Returns**

dict: The server response containing version details.

**Raises**

* **`Exception`**: If the version creation fails.

---

#### `list_files()`

```python
def list_files(self, model_name: str, version: str) ‑> List[Dict]
```
List files for a specific version of a model.

**Arguments**

* **`model_name (str)`**: The unique identifier for the model.
* **`version (str)`**: The version identifier for the model.

**Returns**

List[Dict]: A list of dictionaries containing file information.

**Raises**

* **`OpenGradientError`**: If the file listing fails.

---

#### `upload()`

```python
def upload(
    self,
    model_path: str,
    model_name: str,
    version: str
) ‑> `FileUploadResult`
```
Upload a model file to the server.

**Arguments**

* **`model_path (str)`**: The path to the model file.
* **`model_name (str)`**: The unique identifier for the model.
* **`version (str)`**: The version identifier for the model.

**Returns**

dict: The processed result.

**Raises**

* **`OpenGradientError`**: If the upload fails.