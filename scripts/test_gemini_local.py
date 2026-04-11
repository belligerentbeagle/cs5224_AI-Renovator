#!/usr/bin/env python3
"""Standalone local test for Gemini room generation.

No AWS, no database, no running server needed.

Usage
─────
    # Basic test with your own room photo:
    uv run python scripts/test_gemini_local.py --image /path/to/room.jpg

    # With custom style and prompt:
    uv run python scripts/test_gemini_local.py \
        --image /path/to/room.jpg \
        --style modern \
        --prompt "bright and airy with lots of plants"

Output
──────
    Writes the generated image to  output_<style>.jpg  in the current directory.
    Also prints the raw Gemini response metadata for debugging.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run(image_path: str, style: str, prompt: str | None) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    img = Path(image_path)
    if not img.exists():
        print(f"ERROR: image file not found: {img}")
        sys.exit(1)

    print(f"[1/3] Reading image: {img} ({img.stat().st_size // 1024} KB)")
    photo_bytes = img.read_bytes()
    mime_type = "image/jpeg" if img.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    # ── Build the same prompt as the production service ────────────────────
    products_block = (
        "- SÖDERHAMN Sofa\n"
        "- KALLAX Shelf Unit\n"
        "- MALM Coffee Table\n"
        "- SYMFONISK Floor Lamp"
    )
    extra = f" {prompt.strip()}" if prompt else ""
    full_prompt = (
        f"You are a professional interior designer. "
        f"Transform this room into a {style} style interior.{extra}\n\n"
        f"Incorporate the following furniture pieces naturally into the redesign:\n"
        f"{products_block}\n\n"
        f"Requirements:\n"
        f"- Generate a photorealistic image of the redesigned room.\n"
        f"- Preserve the room's structural layout, windows, and dimensions.\n"
        f"- Replace existing furniture and decor with the listed items.\n"
        f"- Use a colour palette and lighting that fits the {style} aesthetic."
    )

    print(f"[2/3] Calling Gemini (model: gemini-2.0-flash-exp, style: {style}) ...")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(data=photo_bytes, mime_type=mime_type)),
                    types.Part(text=full_prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # ── Debug: show what came back ─────────────────────────────────────────
    if response.candidates:
        finish = response.candidates[0].finish_reason
        print(f"       finish_reason: {finish}")
        part_types = [
            "IMAGE" if (p.inline_data and p.inline_data.data) else "TEXT"
            for p in response.candidates[0].content.parts or []
        ]
        print(f"       parts returned: {part_types}")
    else:
        print("       WARNING: no candidates in response")

    # ── Extract image ──────────────────────────────────────────────────────
    image_bytes: bytes | None = None
    text_response: str | None = None

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
            elif part.text:
                text_response = part.text

    if text_response:
        print(f"       text response: {text_response[:200]}")

    if not image_bytes:
        print("\nERROR: Gemini returned no image.")
        print("This usually means:")
        print("  - The model doesn't support image generation yet for your API key tier")
        print("  - Try updating google-genai: uv add 'google-genai>=1.5.0'")
        print("  - Or check https://ai.google.dev for model availability")
        sys.exit(1)

    # ── Save output ────────────────────────────────────────────────────────
    out_path = Path(f"output_{style}.jpg")
    out_path.write_bytes(image_bytes)

    print(f"[3/3] Saved → {out_path.resolve()} ({len(image_bytes) // 1024} KB)")
    print("\nSUCCESS — open the output file to inspect the generated room image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini room generation locally")
    parser.add_argument("--image", required=True, help="Path to a room photo (JPEG or PNG)")
    parser.add_argument("--style", default="scandinavian", help="Interior style (default: scandinavian)")
    parser.add_argument("--prompt", default=None, help="Optional extra design note")
    args = parser.parse_args()
    run(args.image, args.style, args.prompt)
