"""Use the OpenAI Python SDK with OpenGradient's OHTTP transport."""

import opengradient as og
from openai import OpenAI

from opengradient.client.confidential_llm import OHTTPXClient

auth = og.login_chat_account()  # Opens Google sign-in in the browser

client = OpenAI(
    api_key="unused",
    base_url=f"{auth.chat_api_base_url.rstrip('/')}/v1",
    http_client=OHTTPXClient(
        relay_url=auth.chat_api_base_url,
        auth_headers=auth.auth_headers,
    ),
)

response = client.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "what are you?"}],
    max_tokens=200,
)

print(response.choices[0].message.content)
