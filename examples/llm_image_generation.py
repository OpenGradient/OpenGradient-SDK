import asyncio
import base64
import logging
import os
import re

import opengradient as og

logging.basicConfig()
logging.getLogger("opengradient").setLevel(logging.DEBUG)

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.*)$", re.DOTALL)


def save_data_uri(data_uri: str, path: str) -> None:
    """Decode a ``data:image/...;base64,...`` URI and write it to ``path``."""
    match = _DATA_URI_RE.match(data_uri)
    payload = match.group("data") if match else data_uri
    with open(path, "wb") as f:
        f.write(base64.b64decode(payload))


async def main():
    llm = og.LLM(private_key=os.environ.get("OG_PRIVATE_KEY"))
    llm.ensure_opg_approval(min_allowance=0.1)

    messages = [
        {"role": "user", "content": "Generate an image of a friendly robot reading a book under a tree."},
    ]

    # Image-output models ("nano banana") return generated images on the response.
    # The text caption (if any) is in chat_output["content"]; the generated images
    # are in result.images as data: URIs. Images travel out-of-band and are not part
    # of the signed output hash.
    result = await llm.chat(
        model=og.TEE_LLM.GEMINI_3_1_FLASH_IMAGE,
        messages=messages,
        max_tokens=1024,
    )

    if result.chat_output and result.chat_output.get("content"):
        print(result.chat_output["content"])

    images = result.images or []
    print(f"Generated {len(images)} image(s)")
    for index, image in enumerate(images):
        path = f"generated_image_{index + 1}.png"
        save_data_uri(image, path)
        print(f"Saved {path}")


asyncio.run(main())
