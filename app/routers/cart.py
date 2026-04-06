from uuid import UUID
from fastapi import APIRouter, HTTPException
from app.models.schemas import CartResponse, TrackClickResponse
from pydantic import BaseModel

router = APIRouter(tags=["Cart"])


@router.get("/projects/{id}/cart", response_model=CartResponse)
async def get_cart(id: UUID):
    # TODO: fetch cart items for the project and attach affiliate links
    raise HTTPException(status_code=501, detail="Not implemented")


class TrackClickRequest(BaseModel):
    project_id: UUID
    product_id: UUID
    user_id: UUID


@router.post("/cart/track", response_model=TrackClickResponse)
async def track_affiliate_click(body: TrackClickRequest):
    # TODO: log click, return affiliate redirect URL
    raise HTTPException(status_code=501, detail="Not implemented")
