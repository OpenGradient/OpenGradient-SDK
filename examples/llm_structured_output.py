"""
Structured LLM output via JSON schema enforcement.

Constrains the model to return a response matching a predefined JSON schema,
ensuring predictable, machine-readable output.

Usage:
    export OG_PRIVATE_KEY="your_private_key"
    python examples/llm_structured_output.py
"""

import json
import os

import opengradient as og

client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
)
client.llm.ensure_opg_approval(opg_amount=2)

# Define a JSON schema for sentiment analysis output
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the classification",
                },
            },
            "required": ["sentiment", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    },
}

result = client.llm.chat(
    model=og.TEE_LLM.GPT_4O,
    messages=[
        {
            "role": "system",
            "content": "You are a sentiment analysis assistant. Analyze the sentiment of the given text.",
        },
        {
            "role": "user",
            "content": "I absolutely love this new feature, it makes everything so much easier!",
        },
    ],
    max_tokens=200,
    response_format=response_format,
)

# The response content is guaranteed to match the schema
output = json.loads(result.chat_output["content"])
print(f"Sentiment:  {output['sentiment']}")
print(f"Confidence: {output['confidence']}")
print(f"Reasoning:  {output['reasoning']}")
