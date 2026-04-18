from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.db import DB
from app.models.schemas import CartResponse, TrackClickResponse
from app.services.auth import CurrentUser
from app.services.cart_service import CartService

router = APIRouter(tags=["Cart"])


class TrackClickRequest(BaseModel):
    project_id: UUID
    product_id: UUID
    user_id: UUID


@router.get("/projects/{id}/cart", response_model=CartResponse)
async def get_cart(
    id: UUID,
    db: DB,
    current_user: CurrentUser,
    design_id: UUID | None = Query(default=None, description="Filter to a specific design generation"),
):
    return CartService().get_project_cart(id, current_user, db, design_id)


@router.post("/cart/track", response_model=TrackClickResponse)
async def track_affiliate_click(body: TrackClickRequest, db: DB, current_user: CurrentUser):
    return CartService().track_affiliate_click(body.product_id, current_user, body.project_id, db)


@router.delete("/projects/{id}/cart/items/{product_id}", status_code=204)
async def remove_cart_item(id: UUID, product_id: UUID, db: DB, current_user: CurrentUser):
    CartService().remove_cart_item(id, product_id, current_user, db)
