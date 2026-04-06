from uuid import UUID
from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import Product, ProductDetail, ScrapedProduct, Substitute
from pydantic import HttpUrl

router = APIRouter(prefix="/products", tags=["Products"])


@router.get("", response_model=dict)
async def search_products(
    q: str | None = Query(None, description="Free-text search"),
    style: str | None = Query(None, examples=["scandinavian"]),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    source: Literal["ikea", "taobao"] | None = Query(None),
    in_stock: bool | None = Query(None),
):
    # TODO: query product catalogue (DB / search index)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{id}", response_model=ProductDetail)
async def get_product(id: UUID):
    # TODO: fetch product by id
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{id}/substitutes", response_model=dict)
async def get_substitutes(
    id: UUID,
    max_price: float = Query(..., ge=0),
    reason: Literal["over_budget", "out_of_stock"] = Query(...),
):
    # TODO: find visually similar products within budget
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/from-url", response_model=ScrapedProduct)
async def index_product_from_url(body: dict):
    # TODO: scrape and normalise product from arbitrary URL
    # body: { url: str }
    raise HTTPException(status_code=501, detail="Not implemented")
