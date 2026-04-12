"""Generation routes: submit a room generation job (202) and poll its status.

New flow
────────
POST /generate/room
  - User provides: photo_id (their uploaded room photo) + furniture (list of
    products selected from /products search) + style_name
  - Returns 202 immediately; background task runs Gemini generation.

POST /generate/design-for-me
  - User provides: style_name only (no furniture selection).
  - Background task auto-searches IKEA, then runs Gemini generation.

GET /generations/{id}
  - Poll until status changes from "pending" to "done" or fails.
"""
from __future__ import annotations

import os
import uuid as _uuid
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.db import DB, SessionLocal
from app.models.orm import (
    DesignGeneration,
    GenerationProduct,
    GenerationStatus,
    Photo,
    Product as ProductORM,
    Project,
)
from app.models.schemas import GeneratedProduct, GenerationDone, GenerationPending
from app.services.auth import CurrentUser

router = APIRouter(tags=["Generation"])

_S3_BUCKET = os.environ.get("S3_BUCKET", "roomstyle-cs5224")
_AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")


# ── Request bodies ────────────────────────────────────────────────────────────

class FurnitureItem(BaseModel):
    """A single piece of furniture selected by the user from the product search."""
    name: str
    image_url: str | None = None          # IKEA CDN / product image URL
    product_id: UUID | None = None        # UUID from /products search response
    price: float = 0.0
    source: Literal["ikea", "taobao", "scraped"] = "ikea"
    buy_url: str | None = None


class GenerateRoomRequest(BaseModel):
    """Explicit flow: user picks photo + furniture + style."""
    project_id: UUID
    photo_id: UUID                         # required — must have uploaded a room photo
    furniture: list[FurnitureItem]         # selected from /products search
    style_name: str = "modern"
    prompt_text: str | None = None


class DesignForMeRequest(BaseModel):
    """Auto flow: backend searches IKEA and picks furniture automatically."""
    project_id: UUID
    photo_id: UUID | None = None           # optional — Gemini can generate without a base photo
    style_name: str = "scandinavian"
    prompt_text: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/generate/room", response_model=GenerationPending, status_code=202)
async def generate_room(
    body: GenerateRoomRequest,
    background_tasks: BackgroundTasks,
    db: DB,
    current_user: CurrentUser,
):
    _assert_project_owned(body.project_id, current_user, db)
    gen = DesignGeneration(
        project_id=body.project_id,
        input_photo_id=body.photo_id,
        style_name=body.style_name,
        prompt_text=body.prompt_text,
        status=GenerationStatus.pending,
    )
    db.add(gen)
    db.commit()
    db.refresh(gen)

    background_tasks.add_task(
        _run_generation,
        str(gen.design_id),
        body.style_name,
        [item.model_dump() for item in body.furniture],
        body.prompt_text,
    )
    return GenerationPending(generation_id=gen.design_id, status="pending")


@router.post("/generate/design-for-me", response_model=GenerationPending, status_code=202)
async def design_for_me(
    body: DesignForMeRequest,
    background_tasks: BackgroundTasks,
    db: DB,
    current_user: CurrentUser,
):
    _assert_project_owned(body.project_id, current_user, db)
    gen = DesignGeneration(
        project_id=body.project_id,
        input_photo_id=body.photo_id,
        style_name=body.style_name,
        prompt_text=body.prompt_text,
        status=GenerationStatus.pending,
    )
    db.add(gen)
    db.commit()
    db.refresh(gen)

    # No furniture passed — background task will search IKEA automatically.
    background_tasks.add_task(
        _run_generation,
        str(gen.design_id),
        body.style_name,
        [],
        body.prompt_text,
    )
    return GenerationPending(generation_id=gen.design_id, status="pending")


@router.get("/generations/{id}")
async def poll_generation(id: UUID, db: DB, current_user: CurrentUser):
    gen = db.query(DesignGeneration).filter(DesignGeneration.design_id == id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if gen.status == GenerationStatus.pending:
        return GenerationPending(generation_id=gen.design_id, status="pending")

    if gen.status == GenerationStatus.failed:
        raise HTTPException(status_code=500, detail="Generation failed")

    # Build completed response
    products: list[GeneratedProduct] = []
    total = 0.0
    for gp in gen.generation_products:
        product = db.query(ProductORM).filter(ProductORM.product_id == gp.product_id).first()
        if product:
            price = float(product.price or 0)
            total += price
            products.append(GeneratedProduct(
                product_id=product.product_id,
                name=product.name or "",
                price=price,
                source=product.external_source or "scraped",
                buy_url=product.product_url or "https://www.ikea.com/sg/en/",
            ))

    project = db.query(Project).filter(Project.project_id == gen.project_id).first()
    budget_limit = float(project.budget_limit) if project and project.budget_limit else None
    over_budget = total > budget_limit if budget_limit else False

    if gen.generated_photo_id:
        image_url = f"https://{_S3_BUCKET}.s3.{_AWS_REGION}.amazonaws.com/generations/{gen.design_id}/output.jpg"
    else:
        image_url = None

    return GenerationDone(
        generation_id=gen.design_id,
        status="done",
        image_url=image_url,
        over_budget=over_budget,
        total_cost=total,
        products=products,
    )


# ── Background task ───────────────────────────────────────────────────────────

async def _run_generation(
    design_id: str,
    style_name: str,
    furniture_items: list[dict],
    prompt_text: str | None,
) -> None:
    """Core generation pipeline — runs after the 202 response is sent.

    Steps
    ─────
    1. If no furniture was provided (design-for-me), search IKEA by style.
    2. Upsert Product rows + GenerationProduct join rows into the DB.
    3. Download each furniture item's product image from its CDN URL.
    4. Fetch the room photo S3 key from the DesignGeneration record.
    5. Call Gemini with [room photo] + [furniture images] + style prompt.
    6. Upload the generated image to S3, record a Photo row, link it.
    7. Mark the generation completed (or failed on any exception).
    """
    import asyncio
    import httpx
    from app.services.gemini_generation import generate_room_image

    db = SessionLocal()
    try:
        # ── Step 1: Auto-search IKEA if no furniture was selected ───────────
        if not furniture_items:
            from app.services.provider_registry import _ikea
            provider = _ikea()
            results = await provider.search(q=style_name, style=style_name)
            furniture_items = [
                {
                    "name": p.name,
                    "image_url": str(p.image_url) if p.image_url else None,
                    "product_id": str(p.product_id),
                    "price": p.price,
                    "source": p.source,
                    "buy_url": str(p.buy_url) if p.buy_url else None,
                }
                for p in results[:8]
            ]

        # ── Step 2: Upsert products + generation join rows ──────────────────
        saved: list[tuple[ProductORM, str | None]] = []  # (orm, image_url)
        for i, item in enumerate(furniture_items[:8]):
            external_id = str(item.get("product_id") or _uuid.uuid4())

            existing = db.query(ProductORM).filter(
                ProductORM.external_source == item.get("source", "ikea"),
                ProductORM.external_product_id == external_id,
            ).first()

            if not existing:
                existing = ProductORM(
                    external_source=item.get("source", "ikea"),
                    external_product_id=external_id,
                    name=item.get("name"),
                    product_url=item.get("buy_url"),
                    image_url=item.get("image_url"),
                    price=item.get("price", 0),
                    currency="SGD",
                )
                db.add(existing)
                db.flush()

            db.add(GenerationProduct(
                design_id=design_id,
                product_id=existing.product_id,
                x_position=float(i % 4) * 0.25,
                y_position=float(i // 4) * 0.5,
            ))
            saved.append((existing, item.get("image_url")))

        db.flush()

        # ── Step 3: Download furniture product images ───────────────────────
        # Each entry: (image_bytes, mime_type, product_name)
        furniture_image_data: list[tuple[bytes, str, str]] = []
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for product_orm, image_url in saved:
                if not image_url:
                    continue
                try:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    furniture_image_data.append((resp.content, mime, product_orm.name or "furniture"))
                except Exception:
                    pass  # skip items whose images can't be fetched

        # ── Step 4 & 5: Gemini generation ───────────────────────────────────
        gen = db.query(DesignGeneration).filter(DesignGeneration.design_id == design_id).first()
        if gen and gen.input_photo_id:
            photo = db.query(Photo).filter(Photo.photo_id == gen.input_photo_id).first()
            if photo:
                output_key = await asyncio.to_thread(
                    generate_room_image,
                    photo.s3_key,
                    design_id,
                    style_name,
                    prompt_text,
                    furniture_image_data,
                )

                # ── Step 6: Persist generated image ─────────────────────────
                gen_photo = Photo(
                    project_id=gen.project_id,
                    photo_type="generated",
                    s3_key=output_key,
                    file_name="output.jpg",
                    mime_type="image/jpeg",
                )
                db.add(gen_photo)
                db.flush()
                gen.generated_photo_id = gen_photo.photo_id

        if gen:
            gen.status = GenerationStatus.completed
        db.commit()

    except Exception:
        db.rollback()
        gen = db.query(DesignGeneration).filter(DesignGeneration.design_id == design_id).first()
        if gen:
            gen.status = GenerationStatus.failed
            db.commit()
    finally:
        db.close()


# ── Helper ────────────────────────────────────────────────────────────────────

def _assert_project_owned(project_id: UUID, user_id: UUID, db) -> None:
    project = db.query(Project).filter(
        Project.project_id == project_id,
        Project.user_id == user_id,
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
