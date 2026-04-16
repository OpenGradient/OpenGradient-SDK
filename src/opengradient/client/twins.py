"""Digital twins chat via OpenGradient verifiable inference."""

from typing import Dict, List, Optional

import httpx

from ..types import TEE_LLM, TextGenerationOutput

TWINS_API_BASE_URL = "https://chat-api.memchat.io"


class Twins:
    """
    Digital twins chat namespace.

    Provides access to digital twin conversations backed by OpenGradient
    verifiable inference. Browse available twins at https://twin.fun.

    Usage:
        twins = og.Twins(api_key="your-api-key")
        response = await twins.chat(
            twin_id="0x1abd463fd6244be4a1dc0f69e0b70cd5",
            model=og.TEE_LLM.GROK_4_1_FAST_NON_REASONING,
            messages=[{"role": "user", "content": "What do you think about AI?"}],
            max_tokens=1000,
        )
        print(response.chat_output["content"])
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def chat(
        self,
        twin_id: str,
        model: TEE_LLM,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> TextGenerationOutput:
        """
        Chat with a digital twin.

        Args:
            twin_id: The unique identifier of the digital twin.
            model: The model to use for inference (e.g., TEE_LLM.GROK_4_1_FAST_NON_REASONING).
            messages: The conversation messages to send.
            temperature: Sampling temperature. Optional.
            max_tokens: Maximum number of tokens for the response. Optional.

        Returns:
            TextGenerationOutput: Generated text results including chat_output and finish_reason.

        Raises:
            RuntimeError: If the request fails.
        """
        url = f"{TWINS_API_BASE_URL}/api/v1/twins/{twin_id}/chat"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self._api_key,
        }

        payload: Dict = {
            "model": model.value,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            choices = result.get("choices")
            if not choices:
                raise RuntimeError(f"Invalid response: 'choices' missing or empty in {result}")

            return TextGenerationOutput(
                transaction_hash="",
                finish_reason=choices[0].get("finish_reason"),
                chat_output=choices[0].get("message"),
                payment_hash=None,
            )
        except RuntimeError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Twins chat request failed: {e.response.status_code} {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Twins chat request failed: {str(e)}")
