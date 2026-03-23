"""Model Hub for creating, versioning, and uploading ML models."""

import os
from typing import Dict, List, Optional

import firebase  # type: ignore[import-untyped]
import requests
from requests_toolbelt import MultipartEncoder  # type: ignore[import-untyped]

from ..types import FileUploadResult, ModelRepository

# Security Update: Credentials moved to environment variables
_FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", ""),
}


class ModelHub:
    """
    Model Hub namespace.

    Provides access to the OpenGradient Model Hub for creating, versioning,
    and uploading ML models. Requires email/password authentication.

    Usage:
        hub = og.ModelHub(email="user@example.com", password="...")
        repo = hub.create_model("my-model", "A description")
        hub.upload("model.onnx", repo.name, repo.version)
    """

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self._hub_user = self._login(email, password) if email is not None else None

    @staticmethod
    def _login(email: str, password: Optional[str]):
        if not _FIREBASE_CONFIG.get("apiKey"):
            raise ValueError("Firebase API Key is missing in environment variables")

        firebase_app = firebase.initialize_app(_FIREBASE_CONFIG)
        return firebase_app.auth().sign_in_with_email_and_password(email, password)

    def create_model(self, model_name: str, model_desc: str, version: str = "1.00") -> ModelRepository:
        """
        Create a new model with the given model_name and model_desc, and a specified version.

        Args:
            model_name (str): The name of the model.
            model_desc (str): The description of the model.
            version (str): The version identifier (default is "1.00").

        Returns:
            dict: The server response containing model details.

        Raises:
            CreateModelError: If the model creation fails.
        """
        if not self._hub_user:
            raise ValueError("User not authenticated")

        url = "https://api.opengradient.ai/api/v0/models/"
        headers = {"Authorization": f"Bearer {self._hub_user['idToken']}", "Content-Type": "application/json"}
        payload = {"name": model_name, "description": model_desc}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.HTTPError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise RuntimeError(f"Model creation failed: {error_details}") from e

        json_response = response.json()
        model_name = json_response.get("name")
        if not model_name:
            raise Exception(f"Model creation response missing 'name'. Full response: {json_response}")

        # Create the specified version for the newly created model
        version_response = self.create_version(model_name, version)

        return ModelRepository(model_name, version_response["versionString"])

    def create_version(self, model_name: str, notes: str = "", is_major: bool = False) -> dict:
        """
        Create a new version for the specified model.

        Args:
            model_name (str): The unique identifier for the model.
            notes (str, optional): Notes for the new version.
            is_major (bool, optional): Whether this is a major version update. Defaults to False.

        Returns:
            dict: The server response containing version details.

        Raises:
            Exception: If the version creation fails.
        """
        if not self._hub_user:
            raise ValueError("User not authenticated")

        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions"
        headers = {"Authorization": f"Bearer {self._hub_user['idToken']}", "Content-Type": "application/json"}
        payload = {"notes": notes, "is_major": is_major}

        try:
            response = requests.post(url, json=payload, headers=headers, allow_redirects=False)
            response.raise_for_status()

            json_response = response.json()

            if isinstance(json_response, list) and not json_response:
                return {"versionString": "Unknown", "note": "Created based on empty response"}
            elif isinstance(json_response, dict):
                version_string = json_response.get("versionString")
                if not version_string:
                    return {"versionString": "Unknown", "note": "Version ID not provided in response"}
                return {"versionString": version_string}
            else:
                raise Exception(f"Unexpected response type: {type(json_response)}")

        except requests.RequestException as e:
            raise Exception(f"Version creation failed: {str(e)}")
        except Exception:
            raise

    def upload(self, model_path: str, model_name: str, version: str) -> FileUploadResult:
        """
        Upload a model file to the server.

        Args:
            model_path (str): The path to the model file.
            model_name (str): The unique identifier for the model.
            version (str): The version identifier for the model.

        Returns:
            dict: The processed result.

        Raises:
            RuntimeError: If the upload fails.
        """

        if not self._hub_user:
            raise ValueError("User not authenticated")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions/{version}/files"
        headers = {"Authorization": f"Bearer {self._hub_user['idToken']}"}

        try:
            with open(model_path, "rb") as file:
                encoder = MultipartEncoder(fields={"file": (os.path.basename(model_path), file, "application/octet-stream")})
                headers["Content-Type"] = encoder.content_type

                response = requests.post(url, data=encoder, headers=headers, timeout=3600)

                if response.status_code == 201:
                    if response.content and response.content != b"null":
                        json_response = response.json()
                        return FileUploadResult(json_response.get("ipfsCid"), json_response.get("size"))
                    else:
                        raise RuntimeError("Empty or null response content received")
                elif response.status_code == 500:
                    raise RuntimeError(f"Internal server error occurred (status_code=500)")
                else:
                    error_message = response.json().get("detail", "Unknown error occurred")
                    raise RuntimeError(f"Upload failed: {error_message} (status_code={response.status_code})")

        except requests.RequestException as e:
            raise RuntimeError(f"Upload failed: {str(e)}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during upload: {str(e)}")

    def list_files(self, model_name: str, version: str) -> List[Dict]:
        """
        List files for a specific version of a model.

        Args:
            model_name (str): The unique identifier for the model.
            version (str): The version identifier for the model.

        Returns:
            List[Dict]: A list of dictionaries containing file information.

        Raises:
            RuntimeError: If the file listing fails.
        """
        if not self._hub_user:
            raise ValueError("User not authenticated")

        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions/{version}/files"
        headers = {"Authorization": f"Bearer {self._hub_user['idToken']}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result: list[dict] = response.json()
            return result

        except requests.RequestException as e:
            raise RuntimeError(f"File listing failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during file listing: {str(e)}")
