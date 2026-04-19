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
_MODEL = "gemini-3-pro-image-preview"
_MAX_FURNITURE = 6   # cap to stay within Gemini's per-request image limits


def _s3():
    ak = os.environ.get("MY_AWS_ACCESS_KEY_ID")
    sk = os.environ.get("MY_AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        return boto3.client(
            "s3",
            region_name=_REGION,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        )
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
        f"You are a model specialised in room designing, performing a precise furniture placement task — NOT a full room redesign.\n\n"
        f"ROOM BACKGROUND — strict rules (MY JOB DEPENDS ON THIS SO PLEASE STRICTLY FOLLOW!):\n"
        f"- Use the EXACT room from the first photo as the background. Do not alter it in any way.\n"
        f"- Do NOT change walls, floor, ceiling, windows, doors, doorways, skirting boards, or room dimensions.\n"
        f"- Do NOT remove ANY existing structural items like built-in cabinets, shelves, or architectural details.\n"
        f"- Every single item originally present in the room image SHOULD NOT BE CHANGED. Nothing must be removed, modified, or painted over.\n"
        f"- Do NOT change the camera angle, perspective, ORIENTATION, or crop of the room's image!\n\n"
        f"- IMPORTANT: Ensure that the furniture you add looks natural and DOES NOT have unnatural lighting or shadows. It should naturally BLEND IN with the room.\n"
        + (
            f"FURNITURE TO ADD — the ONLY change allowed:\n"
            + "\n".join(f"- {label}" for _, _, label in furniture_images)
            + f"\nPlace these items into the room exactly as they appear in the product images above.\n\n"
            if furniture_images else ""
        )
        +
        f"Style finish: {style_name}.{extra}\n\n"
        f"The final image must show the original room with ONLY the listed furniture placed inside it — nothing else added, absolutely nothing else removed or changed."
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


_LIGHTING_PROMPTS: dict[str, str] = {
    "day": (
        "Transform the lighting to natural daytime. Simulate bright, even natural daylight streaming in through "
        "the windows. The room should look like midday — balanced white light, clear visibility, soft natural "
        "shadows, all artificial lights off. The outside view through windows should look bright."
    ),
    "afternoon": (
        "Transform the lighting to warm golden-hour afternoon. Simulate late-afternoon sunlight streaming through "
        "windows at a low angle, casting long soft diagonal shadows across the floor and furniture. The room "
        "should have warm amber and golden tones, a cozy glowing atmosphere."
    ),
    "night": (
        "Transform the lighting to nighttime. Make the view through any windows completely dark — it is night "
        "outside. Switch on all interior artificial lights: warm incandescent ceiling lights, floor lamps, table "
        "lamps. The room should feel intimate and warmly lit from within with no natural daylight."
    ),
    "cove": (
        "Transform the lighting to dramatic indirect cove lighting. Add soft LED light hidden in ceiling coves or "
        "troughs along the perimeter, creating a gentle upward wash of light across the ceiling and walls. "
        "No direct fixtures are visible — only a sophisticated diffused ambient glow. The mood is elegant and calm."
    ),
    "spot": (
        "Transform the lighting to recessed ceiling spotlights and downlights. Add focused pools of light cast "
        "downward from ceiling spots onto furniture pieces and the floor. Create high contrast between the brightly "
        "lit areas and the softer shadow zones between them. The room looks architectural and gallery-like."
    ),
}

LIGHTING_MODES = list(_LIGHTING_PROMPTS.keys())


def apply_lighting_image(
    photo_s3_key: str,
    design_id: str,
    lighting_type: str,
) -> str:
    """Apply an atmospheric lighting transformation to an existing room image.

    Parameters
    ----------
    photo_s3_key:   S3 key of the room image to transform.
    design_id:      UUID string of the new DesignGeneration record.
    lighting_type:  One of the keys in _LIGHTING_PROMPTS
                    ('day', 'afternoon', 'night', 'cove', 'spot').

    Returns
    -------
    The S3 key of the transformed output image.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    lighting_type = lighting_type.lower()
    if lighting_type not in _LIGHTING_PROMPTS:
        raise ValueError(
            f"Unknown lighting type '{lighting_type}'. Must be one of {LIGHTING_MODES}"
        )

    s3 = _s3()
    obj = s3.get_object(Bucket=_BUCKET, Key=photo_s3_key)
    room_bytes: bytes = obj["Body"].read()
    room_mime: str = obj.get("ContentType") or "image/jpeg"

    logger.info(
        "gemini_lighting design_id=%s model=%s lighting=%s",
        design_id, _MODEL, lighting_type,
    )

    parts: list[types.Part] = [
        types.Part(inline_data=types.Blob(data=room_bytes, mime_type=room_mime)),
        types.Part(text=(
            f"This is the current room design. Apply ONLY the following lighting transformation:\n\n"
            f"{_LIGHTING_PROMPTS[lighting_type]}\n\n"
            f"STRICT RULES — follow every one exactly:\n"
            f"- Do NOT move, add, or remove any furniture, decorations, rugs, plants, or objects.\n"
            f"- Do NOT change walls, floor material, ceiling structure, windows, or doors.\n"
            f"- Do NOT change the camera angle, perspective, or crop.\n"
            f"- ONLY change the lighting — colours, shadows, brightness, and light sources.\n"
            f"- Every piece of furniture must remain in exactly the same position as the original.\n\n"
            f"Output the room image with ONLY the lighting changed."
        )),
    ]

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
        logger.error("Gemini lighting API call failed: %s", str(e))
        raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

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
            f"Gemini returned no image for lighting design_id={design_id}. "
            f"finish_reason={finish}"
        )

    output_key = f"generations/{design_id}/output.jpg"
    s3.put_object(
        Bucket=_BUCKET,
        Key=output_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    logger.info(
        "gemini_lighting_upload design_id=%s s3_key=%s bytes=%d",
        design_id, output_key, len(image_bytes),
    )
    return output_key


def refine_room_image(
    photo_s3_key: str,
    design_id: str,
    style_name: str,
    prompt_text: str | None,
) -> str:
    """Apply a targeted refinement to an already-generated room image.

    Unlike generate_room_image, this function does not place furniture.
    It sends the existing design to Gemini with an edit instruction,
    asking it to apply only the specific change requested.

    Parameters
    ----------
    photo_s3_key:  S3 key of the previously generated room image.
    design_id:     UUID string of the new DesignGeneration record.
    style_name:    e.g. "scandinavian", "modern", "industrial".
    prompt_text:   The refinement instruction from the user.

    Returns
    -------
    The S3 key of the refined output image.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    s3 = _s3()
    obj = s3.get_object(Bucket=_BUCKET, Key=photo_s3_key)
    room_bytes: bytes = obj["Body"].read()
    room_mime: str = obj.get("ContentType") or "image/jpeg"

    logger.info(
        "gemini_refine design_id=%s model=%s style=%s",
        design_id, _MODEL, style_name,
    )

    change = prompt_text.strip() if prompt_text else f"Refine the design to better match the {style_name} style."

    parts: list[types.Part] = [
        types.Part(inline_data=types.Blob(data=room_bytes, mime_type=room_mime)),
        types.Part(text=(
            f"This is the current room design. Apply ONLY the following change:\n\n"
            f"{change}\n\n"
            f"Rules:\n"
            f"- Do NOT change the camera angle, perspective, or crop.\n"
            f"- If the change asks to MOVE a piece of furniture: place it in the new position and REMOVE it from its old position — it must not appear in both places.\n"
            f"- If the change asks to SWAP or REARRANGE furniture: each piece must appear exactly once in its new position; do not duplicate any item.\n"
            f"- Do NOT add new furniture or decorations that are not already in the image unless explicitly requested.\n"
            f"- Do NOT remove furniture unless the change explicitly asks for removal or a move.\n"
            f"- Keep the room background (walls, floor, ceiling, windows) exactly as shown.\n"
            f"- Do not add rugs, plants, lamps, artwork, or decorations unless explicitly requested.\n"
            f"Style: {style_name}.\n\n"
            f"Output the modified room image."
        )),
    ]

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
        logger.error("Gemini refine API call failed: %s", str(e))
        raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

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
            f"Gemini returned no image for refinement design_id={design_id}. "
            f"finish_reason={finish}"
        )

    output_key = f"generations/{design_id}/output.jpg"
    s3.put_object(
        Bucket=_BUCKET,
        Key=output_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    logger.info(
        "gemini_refine_upload design_id=%s s3_key=%s bytes=%d",
        design_id, output_key, len(image_bytes),
    )
    return output_key
