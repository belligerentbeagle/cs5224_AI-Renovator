from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl

from app.models.schemas import Product, ProductDetail, ScrapedProduct
from app.services.provider_registry import Providers
from app.services.scraping.errors import ExtractionError, NetworkError
from app.services.scraping.graph import run_indexer

router = APIRouter(prefix="/products", tags=["Products"])


class IndexFromUrlRequest(BaseModel):
    url: HttpUrl


@router.get("", response_model=list[Product])
async def search_products(
    providers: Providers,
    q: str | None = Query(None, description="Free-text search"),
    style: str | None = Query(None, examples=["scandinavian"]),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    source: Literal["ikea", "taobao"] | None = Query(None),
    in_stock: bool | None = Query(None),
):
    if not providers:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source!r}")

    results: list[Product] = []
    for provider in providers:
        results.extend(
            await provider.search(
                q=q,
                style=style,
                min_price=min_price,
                max_price=max_price,
                in_stock=in_stock,
            )
        )
    return results


@router.get("/{id}", response_model=ProductDetail)
async def get_product(
    id: UUID,
    providers: Providers,
):
    for provider in providers:
        detail = await provider.get_product(str(id))
        if detail is not None:
            return detail
    raise HTTPException(status_code=404, detail="Product not found")


@router.post("/from-url", response_model=ScrapedProduct, status_code=201)
async def index_product_from_url(body: IndexFromUrlRequest):
    """Scrape and normalise a product from any furniture store URL.

    Strategy:
      1. Fetch the page (httpx, browser-like UA).
      2. Try JSON-LD + OpenGraph structured extraction — free, instant.
      3. If name or price is missing, fall back to Gemini LLM extraction.
      4. Validate the result is a real product page, then normalise.

    Error responses:
      502 — the URL could not be fetched (timeout, DNS, non-2xx).
      422 — the page was fetched but is not a recognisable product page.
    """
    url = str(body.url)
    try:
        product_data = await run_indexer(url)
    except NetworkError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ScrapedProduct(**product_data)
