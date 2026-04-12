#!/usr/bin/env python3
"""Standalone local test for Gemini room generation.

Mirrors the new production flow:
  room photo + furniture images → Gemini → redesigned room image

No AWS, no database, no running server needed.

Usage
─────
    # Room photo + one furniture image:
    uv run python scripts/test_gemini_local.py \
        --room scripts/house.jpg \
        --furniture scripts/sofa.png

    # Multiple furniture items:
    uv run python scripts/test_gemini_local.py \
        --room scripts/house.jpg \
        --furniture scripts/sofa.png scripts/chair.jpg \
        --style modern \
        --prompt "bright and airy with lots of natural light"

Output
──────
    Writes the generated image to  output_<style>.jpg  in the current directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def run(room_path: str, furniture_paths: list[str], style: str, prompt: str | None) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    # ── Load room photo ────────────────────────────────────────────────────
    room = Path(room_path)
    if not room.exists():
        print(f"ERROR: room image not found: {room}")
        sys.exit(1)
    room_bytes = room.read_bytes()
    room_mime = _MIME.get(room.suffix.lower(), "image/jpeg")
    print(f"[room]      {room.name}  ({room.stat().st_size // 1024} KB)")

    # ── Load furniture images ──────────────────────────────────────────────
    # Each entry: (bytes, mime_type, label)
    furniture_image_data: list[tuple[bytes, str, str]] = []
    for fp in furniture_paths:
        p = Path(fp)
        if not p.exists():
            print(f"WARNING: furniture image not found, skipping: {p}")
            continue
        mime = _MIME.get(p.suffix.lower(), "image/jpeg")
        furniture_image_data.append((p.read_bytes(), mime, p.stem.replace("_", " ").title()))
        print(f"[furniture] {p.name}  ({p.stat().st_size // 1024} KB)")

    print(f"\nCalling Gemini (model: gemini-2.5-flash-image, style: {style}, furniture: {len(furniture_image_data)}) ...")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # ── Build multi-image request (same logic as production service) ───────
    parts: list[types.Part] = []
    parts.append(types.Part(inline_data=types.Blob(data=room_bytes, mime_type=room_mime)))

    if furniture_image_data:
        parts.append(types.Part(
            text=(
                f"The image above is the current room. "
                f"The following {len(furniture_image_data)} image(s) are the furniture items "
                f"the user has selected to include in the redesign:"
            )
        ))
        for img_bytes, img_mime, label in furniture_image_data:
            parts.append(types.Part(text=f"Furniture: {label}"))
            parts.append(types.Part(inline_data=types.Blob(data=img_bytes, mime_type=img_mime)))

    extra = f" {prompt.strip()}" if prompt else ""
    instruction = (
        f"You are performing a precise furniture placement task — NOT a full room redesign.\n\n"
        f"ROOM BACKGROUND — strict rules:\n"
        f"- Use the EXACT room from the first photo as the background. Do not alter it in any way.\n"
        f"- Do NOT change walls, floor, ceiling, windows, doorways, skirting boards, or room dimensions.\n"
        f"- Do NOT change the camera angle, perspective, or crop.\n\n"
        + (
            f"FURNITURE TO ADD — the ONLY change allowed:\n"
            + "\n".join(f"- {label}" for _, _, label in furniture_image_data)
            + f"\nPlace these items into the room exactly as they appear in the product images above.\n\n"
            if furniture_image_data else ""
        )
        + f"STRICT PROHIBITIONS — do not add any of the following under any circumstances:\n"
        f"- Rugs or mats\n"
        f"- Coffee tables, side tables, or any additional tables\n"
        f"- Cushions or throws not already on the listed furniture\n"
        f"- Plants, flowers, or any greenery\n"
        f"- Lamps, floor lights, or ceiling lights\n"
        f"- Artwork, frames, mirrors, or wall decorations\n"
        f"- Shelves, cabinets, or storage not in the furniture list\n"
        f"- Any object, decoration, or furniture not explicitly listed above\n\n"
        f"Style finish: {style}.{extra}\n\n"
        f"The final image must show the original room with ONLY the listed furniture placed inside it — nothing else added, nothing removed."
    )
    parts.append(types.Part(text=instruction))

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    # ── Debug info ─────────────────────────────────────────────────────────
    if response.candidates:
        finish = response.candidates[0].finish_reason
        part_types = [
            "IMAGE" if (p.inline_data and p.inline_data.data) else "TEXT"
            for p in response.candidates[0].content.parts or []
        ]
        print(f"finish_reason: {finish}")
        print(f"parts returned: {part_types}")
    else:
        print("WARNING: no candidates in response")

    # ── Extract and save image ─────────────────────────────────────────────
    image_bytes: bytes | None = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
            elif part.text:
                print(f"text response: {part.text[:200]}")

    if not image_bytes:
        print("\nERROR: Gemini returned no image.")
        sys.exit(1)

    out_path = Path(f"output_{style}.jpg")
    out_path.write_bytes(image_bytes)
    print(f"\nSaved → {out_path.resolve()}  ({len(image_bytes) // 1024} KB)")
    print("SUCCESS — open the file to inspect the generated room image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini room generation locally")
    parser.add_argument("--room", required=True, help="Path to a room photo (JPEG or PNG)")
    parser.add_argument("--furniture", nargs="*", default=[], help="One or more furniture image paths")
    parser.add_argument("--style", default="scandinavian", help="Interior style (default: scandinavian)")
    parser.add_argument("--prompt", default=None, help="Optional extra design note")
    args = parser.parse_args()
    run(args.room, args.furniture, args.style, args.prompt)
