"""Model Hub for creating, versioning, and uploading ML models."""

import os
import time
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

# Firebase idTokens expire after 3600 seconds; refresh 60 s before expiry
_TOKEN_REFRESH_MARGIN_SEC = 60


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
        self._firebase_app = None
        self._hub_user = None
        self._token_expiry: float = 0.0

        if email is not None:
            self._firebase_app, self._hub_user = self._login(email, password)
            expires_in = int(self._hub_user.get("expiresIn", 3600))
            self._token_expiry = time.time() + expires_in

    @staticmethod
    def _login(email: str, password: Optional[str]):
        if not _FIREBASE_CONFIG.get("apiKey"):
            raise ValueError("Firebase API Key is missing in environment variables")

        firebase_app = firebase.initialize_app(_FIREBASE_CONFIG)
        user = firebase_app.auth().sign_in_with_email_and_password(email, password)
        return firebase_app, user

    def _get_auth_token(self) -> str:
        """Return a valid Firebase idToken, refreshing it if it has expired or is
        about to expire within ``_TOKEN_REFRESH_MARGIN_SEC`` seconds.

        Raises:
            ValueError: If the user is not authenticated.
        """
        if not self._hub_user:
            raise ValueError("User not authenticated")

        if time.time() >= self._token_expiry - _TOKEN_REFRESH_MARGIN_SEC:
            # Refresh the token using the stored refresh token
            refresh_token = self._hub_user.get("refreshToken")
            if not refresh_token or self._firebase_app is None:
                raise ValueError(
                    "Cannot refresh Firebase token: missing refresh token or Firebase app. "
                    "Please re-authenticate by creating a new ModelHub instance."
                )
            refreshed = self._firebase_app.auth().refresh(refresh_token)
            self._hub_user["idToken"] = refreshed["idToken"]
            self._hub_user["refreshToken"] = refreshed.get("refreshToken", refresh_token)
            expires_in = int(refreshed.get("expiresIn", 3600))
            self._token_expiry = time.time() + expires_in

        return str(self._hub_user["idToken"])  # cast Any->str for mypy [no-any-return]

    def create_model(self, model_name: str, model_desc: str, version: str = "1.00") -> ModelRepository:
        """
        Create a new model with the given model_name and model_desc, and a specified version.

        Args:
            model_name (str): The name of the model.
            model_desc (str): The description of the model.
            version (str): A label used in the initial version notes (default is "1.00").
                           Note: the actual version string is assigned by the server.

        Returns:
            ModelRepository: Object containing the model name and server-assigned version string.

        Raises:
            RuntimeError: If the model creation fails.
        """
        url = "https://api.opengradient.ai/api/v0/models/"
        headers = {"Authorization": f"Bearer {self._get_auth_token()}", "Content-Type": "application/json"}
        payload = {"name": model_name, "description": model_desc}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except requests.HTTPError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise RuntimeError(f"Model creation failed: {error_details}") from e

        json_response = response.json()
        created_name = json_response.get("name")
        if not created_name:
            raise Exception(f"Model creation response missing 'name'. Full response: {json_response}")

        # Create the initial version for the newly created model.
        # Pass `version` as release notes (e.g. "1.00") since the server assigns
        # its own version string — previously `version` was incorrectly passed as
        # the positional `notes` argument, producing the same result but confusingly.
        version_response = self.create_version(created_name, notes=f"Initial version {version}")

        return ModelRepository(created_name, version_response["versionString"])

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
        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions"
        headers = {"Authorization": f"Bearer {self._get_auth_token()}", "Content-Type": "application/json"}
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
            FileUploadResult: The processed result.

        Raises:
            RuntimeError: If the upload fails.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions/{version}/files"
        headers = {"Authorization": f"Bearer {self._get_auth_token()}"}

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
        url = f"https://api.opengradient.ai/api/v0/models/{model_name}/versions/{version}/files"
        headers = {"Authorization": f"Bearer {self._get_auth_token()}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result: list[dict] = response.json()
            return result

        except requests.RequestException as e:
            raise RuntimeError(f"File listing failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during file listing: {str(e)}")
