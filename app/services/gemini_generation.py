"""Gemini image generation service for room redesign (Spatial Sync).

Flow
────
  1. Download the original room photo bytes from S3.
  2. Build a multi-image Gemini request:
       [room photo] + [furniture image 1] + [furniture image 2] + ... + [prompt]
  3. Gemini visually places the selected furniture into the room.
  4. Upload the generated JPEG to S3 at generations/{design_id}/output.jpg.
  5. Return the S3 key.

Raises
──────
  ValueError   – GEMINI_API_KEY not configured.
  RuntimeError – Gemini returned no image in its response.
"""
from __future__ import annotations

import logging
import os

import boto3
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("S3_BUCKET", "roomstyle-cs5224")
_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
_MODEL = "gemini-2.5-flash-image"
_MAX_FURNITURE = 6   # cap to stay within Gemini's per-request image limits


def _s3():
    return boto3.client("s3", region_name=_REGION)


def generate_room_image(
    photo_s3_key: str,
    design_id: str,
    style_name: str,
    prompt_text: str | None,
    furniture_images: list[tuple[bytes, str, str]],
) -> str:
    """Generate a redesigned room image and store it in S3.

    Parameters
    ----------
    photo_s3_key:      S3 key of the original uploaded room photo.
    design_id:         UUID string of the DesignGeneration record.
    style_name:        e.g. "scandinavian", "modern", "industrial".
    prompt_text:       Optional free-text note from the user.
    furniture_images:  List of (image_bytes, mime_type, product_name) tuples
                       for the furniture items the user selected.
                       Capped at _MAX_FURNITURE items.

    Returns
    -------
    The S3 key of the generated output image.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    # ── Download original room photo from S3 ───────────────────────────────
    s3 = _s3()
    obj = s3.get_object(Bucket=_BUCKET, Key=photo_s3_key)
    room_bytes: bytes = obj["Body"].read()
    room_mime: str = obj.get("ContentType") or "image/jpeg"

    furniture_images = furniture_images[:_MAX_FURNITURE]

    logger.info(
        "gemini_call design_id=%s model=%s style=%s furniture_count=%d",
        design_id, _MODEL, style_name, len(furniture_images),
    )

    # ── Build multi-image request ──────────────────────────────────────────
    # Structure: room photo → labelled furniture images → instruction text
    parts: list[types.Part] = []

    parts.append(types.Part(
        inline_data=types.Blob(data=room_bytes, mime_type=room_mime)
    ))

    if furniture_images:
        parts.append(types.Part(
            text=(
                f"The image above is the current room. "
                f"The following {len(furniture_images)} image(s) are the furniture items "
                f"the user has selected to include in the redesign:"
            )
        ))
        for img_bytes, img_mime, label in furniture_images:
            parts.append(types.Part(text=f"Furniture: {label}"))
            parts.append(types.Part(
                inline_data=types.Blob(data=img_bytes, mime_type=img_mime)
            ))

    # Final instruction
    extra = f" {prompt_text.strip()}" if prompt_text else ""
    furniture_names = ", ".join(label for _, _, label in furniture_images)
    instruction = (
        f"You are performing a precise furniture placement task — NOT a full room redesign.\n\n"
        f"ROOM BACKGROUND — strict rules:\n"
        f"- Use the EXACT room from the first photo as the background. Do not alter it in any way.\n"
        f"- Do NOT change walls, floor, ceiling, windows, doorways, skirting boards, or room dimensions.\n"
        f"- Do NOT change the camera angle, perspective, or crop.\n\n"
        + (
            f"FURNITURE TO ADD — the ONLY change allowed:\n"
            + "\n".join(f"- {label}" for _, _, label in furniture_images)
            + f"\nPlace these items into the room exactly as they appear in the product images above.\n\n"
            if furniture_images else ""
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
        f"Style finish: {style_name}.{extra}\n\n"
        f"The final image must show the original room with ONLY the listed furniture placed inside it — nothing else added, nothing removed."
    )
    parts.append(types.Part(text=instruction))

    # ── Call Gemini ────────────────────────────────────────────────────────
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
    except Exception as e:
        logger.error("Gemini API call failed: %s", str(e))
        raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

    # ── Extract image bytes ────────────────────────────────────────────────
    image_bytes: bytes | None = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                break
        if image_bytes:
            break

    if not image_bytes:
        finish = (
            response.candidates[0].finish_reason
            if response.candidates else "unknown"
        )
        raise RuntimeError(
            f"Gemini returned no image for design_id={design_id}. "
            f"finish_reason={finish}"
        )

    # ── Upload generated image to S3 ───────────────────────────────────────
    output_key = f"generations/{design_id}/output.jpg"
    s3.put_object(
        Bucket=_BUCKET,
        Key=output_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    logger.info(
        "gemini_upload design_id=%s s3_key=%s bytes=%d",
        design_id, output_key, len(image_bytes),
    )
    return output_key
