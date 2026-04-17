from uuid import UUID

from fastapi import APIRouter
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
async def get_cart(id: UUID, db: DB, current_user: CurrentUser):
    service = CartService()
    return service.get_project_cart(id, current_user, db)


@router.post("/cart/track", response_model=TrackClickResponse)
async def track_affiliate_click(body: TrackClickRequest, db: DB, current_user: CurrentUser):
    service = CartService()
    return service.track_affiliate_click(body.product_id, current_user, db)
