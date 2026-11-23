"""Smoke-test the transport against a live developer relay.

    DF_API_KEY=df_sk_... python examples/smoke.py                # non-streaming
    DF_API_KEY=df_sk_... python examples/smoke.py --stream       # streaming
    DF_API_KEY=df_sk_... python examples/smoke.py --async        # AsyncOpenAI
    DF_API_KEY=df_sk_... python examples/smoke.py --relay-url http://localhost:8000

Each request costs a little from the key's developer credits.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from openai import AsyncOpenAI, OpenAI

from opengradient_ohttp import DEFAULT_RELAY_URL, AsyncOHTTPTransport, OHTTPTransport

BASE_URL = "https://ohttp.invalid/v1"  # never contacted; the transport routes to the relay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--prompt", default="In one sentence: what is Oblivious HTTP?")
    parser.add_argument("--relay-url", default=DEFAULT_RELAY_URL)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--async", dest="use_async", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DF_API_KEY")
    if not api_key:
        sys.exit("Set DF_API_KEY to a df_sk_... developer API key.")

    if args.use_async:
        asyncio.run(run_async(args, api_key))
    else:
        run_sync(args, api_key)


def run_sync(args: argparse.Namespace, api_key: str) -> None:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(transport=OHTTPTransport(relay_url=args.relay_url)),
    )
    messages = [{"role": "user", "content": args.prompt}]
    if args.stream:
        for chunk in client.chat.completions.create(model=args.model, messages=messages, max_tokens=200, stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n[stream verified]")
        return

    raw = client.chat.completions.with_raw_response.create(model=args.model, messages=messages, max_tokens=200)
    completion = raw.parse()
    print(completion.choices[0].message.content)
    print(f"[verified={raw.headers.get('x-tee-verified')} tee_id={raw.headers.get('x-tee-id')} host={raw.headers.get('x-tee-host')}]")


async def run_async(args: argparse.Namespace, api_key: str) -> None:
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.AsyncClient(transport=AsyncOHTTPTransport(relay_url=args.relay_url)),
    )
    messages = [{"role": "user", "content": args.prompt}]
    if args.stream:
        stream = await client.chat.completions.create(model=args.model, messages=messages, max_tokens=200, stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n[stream verified]")
    else:
        completion = await client.chat.completions.create(model=args.model, messages=messages, max_tokens=200)
        print(completion.choices[0].message.content)
        print("[verified]")
    await client.close()


if __name__ == "__main__":
    main()
