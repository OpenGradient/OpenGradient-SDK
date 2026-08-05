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

    # Step 1: Generate the original image of a car.
    #
    # Image-output models ("nano banana") return generated images on the response.
    # The text caption (if any) is in chat_output["content"]; the generated images
    # are in result.images as data: URIs. Images travel out-of-band and are not part
    # of the signed output hash.
    print("Generating a car...")
    result = await llm.chat(
        model=og.TEE_LLM.SEEDREAM_4_0,
        messages=[
            {"role": "user", "content": "Generate an image of a red sports car parked on an empty road."},
        ],
    )

    images = result.images or []
    print(f"Generated {len(images)} image(s)")
    if not images:
        print("No image was returned; cannot continue with the edit.")
        return

    car_image = images[0]
    save_data_uri(car_image, "generated_image_1.png")
    print("Saved generated_image_1.png")

    # Step 2: Send the generated image back inline and ask to add a palm tree.
    #
    # Input images use OpenAI-style multimodal content: a list of content blocks
    # mixing text and image_url. We pass the data: URI from step 1 straight back in
    # so the model edits the exact image it just produced.
    print("Adding a palm tree...")
    edit_result = await llm.chat(
        model=og.TEE_LLM.SEEDREAM_4_0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add a palm tree next to the car."},
                    {"type": "image_url", "image_url": {"url": car_image}},
                ],
            },
        ],
    )

    if edit_result.chat_output and edit_result.chat_output.get("content"):
        print(edit_result.chat_output["content"])

    edited_images = edit_result.images or []
    print(f"Generated {len(edited_images)} edited image(s)")
    for index, image in enumerate(edited_images):
        path = f"edited_image_{index + 1}.png"
        save_data_uri(image, path)
        print(f"Saved {path}")


asyncio.run(main())
