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

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from app.db import DB
from app.models.schemas import GenerationPending
from app.services.auth import CurrentUser
from app.services.generation_service import GenerationService

router = APIRouter(tags=["Generation"])


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
    service = GenerationService()
    result = service.submit_room_generation(
        body.project_id,
        current_user,
        body.photo_id,
        body.style_name,
        [item.model_dump() for item in body.furniture],
        body.prompt_text,
        db
    )

    background_tasks.add_task(
        service.run_generation_pipeline,
        str(result.generation_id),
        body.style_name,
        [item.model_dump() for item in body.furniture],
        body.prompt_text,
    )
    return result


@router.post("/generate/design-for-me", response_model=GenerationPending, status_code=202)
async def design_for_me(
    body: DesignForMeRequest,
    background_tasks: BackgroundTasks,
    db: DB,
    current_user: CurrentUser,
):
    service = GenerationService()
    result = service.submit_design_for_me(
        body.project_id,
        current_user,
        body.photo_id,
        body.style_name,
        body.prompt_text,
        db
    )

    # No furniture passed — background task will search IKEA automatically
    background_tasks.add_task(
        service.run_generation_pipeline,
        str(result.generation_id),
        body.style_name,
        [],
        body.prompt_text,
    )
    return result


@router.get("/generations/{id}")
async def poll_generation(id: UUID, db: DB, current_user: CurrentUser):
    service = GenerationService()
    return service.get_generation_status(id, current_user, db)
