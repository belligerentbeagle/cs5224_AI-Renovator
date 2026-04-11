"""Gemini image generation service for room redesign (Spatial Sync).

Flow
────
  1. Download the original room photo bytes from S3.
  2. Build a style + furniture prompt.
  3. Call Gemini 2.0 Flash (multimodal in → image out).
  4. Upload the generated JPEG to S3 at generations/{design_id}/output.jpg.
  5. Return the S3 key.

Raises
──────
  ValueError   – GEMINI_API_KEY not configured.
  RuntimeError – Gemini returned no image in its response.
  Any boto3 exception propagates; the caller should catch and mark the
  generation as failed.
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


def _s3():
    return boto3.client("s3", region_name=_REGION)


def generate_room_image(
    photo_s3_key: str,
    design_id: str,
    style_name: str,
    prompt_text: str | None,
    product_names: list[str],
) -> str:
    """Generate a redesigned room image and store it in S3.

    Parameters
    ----------
    photo_s3_key:  S3 key of the original uploaded room photo.
    design_id:     UUID string of the DesignGeneration record.
    style_name:    e.g. "scandinavian", "modern", "industrial".
    prompt_text:   Optional free-text designer note from the user.
    product_names: Names of furniture items selected for this generation.

    Returns
    -------
    The S3 key of the generated output image.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    # ── Download original room photo ────────────────────────────────────────
    s3 = _s3()
    obj = s3.get_object(Bucket=_BUCKET, Key=photo_s3_key)
    photo_bytes: bytes = obj["Body"].read()
    mime_type: str = obj.get("ContentType") or "image/jpeg"

    # ── Build the redesign prompt ───────────────────────────────────────────
    products_block = (
        "\n".join(f"- {name}" for name in product_names)
        if product_names
        else "appropriate modern furniture"
    )
    extra = f" {prompt_text.strip()}" if prompt_text else ""
    full_prompt = (
        f"You are a professional interior designer. "
        f"Transform this room into a {style_name} style interior.{extra}\n\n"
        f"Incorporate the following furniture pieces naturally into the redesign:\n"
        f"{products_block}\n\n"
        f"Requirements:\n"
        f"- Generate a photorealistic image of the redesigned room.\n"
        f"- Preserve the room's structural layout, windows, and dimensions.\n"
        f"- Replace existing furniture and decor with the listed items.\n"
        f"- Use a colour palette and lighting that fits the {style_name} aesthetic."
    )

    logger.info(
        "gemini_call design_id=%s model=%s style=%s products=%d",
        design_id, _MODEL, style_name, len(product_names),
    )

    # ── Call Gemini ─────────────────────────────────────────────────────────
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=photo_bytes, mime_type=mime_type)
                    ),
                    types.Part(text=full_prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # ── Extract image bytes from response ───────────────────────────────────
    image_bytes: bytes | None = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                break
        if image_bytes:
            break

    if not image_bytes:
        raise RuntimeError(
            f"Gemini returned no image for design_id={design_id}. "
            f"Response finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}"
        )

    # ── Upload generated image to S3 ────────────────────────────────────────
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
