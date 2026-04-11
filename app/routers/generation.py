"""Generation routes: submit a room generation job (202) and poll its status."""
from __future__ import annotations

import os
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

class GenerateRoomRequest(BaseModel):
    project_id: UUID
    photo_id: UUID | None = None
    style_name: str = "modern"
    prompt_text: str | None = None


class DesignForMeRequest(BaseModel):
    project_id: UUID
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
    background_tasks.add_task(_run_product_search, str(gen.design_id), body.style_name)
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
        style_name=body.style_name,
        prompt_text=body.prompt_text,
        status=GenerationStatus.pending,
    )
    db.add(gen)
    db.commit()
    db.refresh(gen)
    background_tasks.add_task(_run_product_search, str(gen.design_id), body.style_name)
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
                buy_url=product.product_url or f"https://www.ikea.com/sg/en/",
            ))

    project = db.query(Project).filter(Project.project_id == gen.project_id).first()
    budget_limit = float(project.budget_limit) if project and project.budget_limit else None
    over_budget = total > budget_limit if budget_limit else False

    # generated_photo_id is set when image generation is wired up; use placeholder until then
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

async def _run_product_search(design_id: str, style_name: str) -> None:
    """Search IKEA for products, then call Gemini to generate the room image.

    Runs after the HTTP response is sent (FastAPI BackgroundTasks).
    Creates its own DB session since the request session is already closed.

    Steps
    ─────
    1. Search IKEA for furniture matching the style.
    2. Upsert Product rows and create GenerationProduct join rows.
    3. If the generation has an input photo, call Gemini with the room
       photo + product names to generate a redesigned image.
    4. Upload the result to S3 and record a Photo row for it.
    5. Mark the DesignGeneration as completed (or failed on any error).
    """
    import asyncio
    from app.services.provider_registry import _ikea  # avoid circular import at module level
    from app.services.gemini_generation import generate_room_image

    db = SessionLocal()
    try:
        # ── Step 1 & 2: IKEA product search ────────────────────────────────
        provider = _ikea()
        results = await provider.search(q=style_name, style=style_name)

        product_names: list[str] = []
        for i, product in enumerate(results[:8]):
            # Upsert: avoid duplicate product rows for the same external item
            existing = db.query(ProductORM).filter(
                ProductORM.external_source == product.source,
                ProductORM.external_product_id == str(product.product_id),
            ).first()

            if not existing:
                existing = ProductORM(
                    external_source=product.source,
                    external_product_id=str(product.product_id),
                    name=product.name,
                    product_url=str(product.buy_url) if product.buy_url else None,
                    image_url=str(product.image_url) if product.image_url else None,
                    price=product.price,
                    currency="SGD",
                )
                db.add(existing)
                db.flush()

            gp = GenerationProduct(
                design_id=design_id,
                product_id=existing.product_id,
                x_position=float(i % 4) * 0.25,
                y_position=float(i // 4) * 0.5,
            )
            db.add(gp)
            product_names.append(product.name)

        db.flush()

        # ── Step 3 & 4: Gemini image generation ────────────────────────────
        gen = db.query(DesignGeneration).filter(DesignGeneration.design_id == design_id).first()
        if gen and gen.input_photo_id:
            photo = db.query(Photo).filter(Photo.photo_id == gen.input_photo_id).first()
            if photo:
                # generate_room_image is synchronous (boto3 + google-genai);
                # run it in a thread so we don't block the event loop.
                output_key = await asyncio.to_thread(
                    generate_room_image,
                    photo.s3_key,
                    design_id,
                    style_name,
                    gen.prompt_text,
                    product_names,
                )

                # Record the generated image as a Photo row
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
