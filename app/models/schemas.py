from __future__ import annotations
from typing import Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, HttpUrl


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthResponse(BaseModel):
    user_id: UUID
    token: str


# ── Projects ──────────────────────────────────────────────────────────────────

class Project(BaseModel):
    project_id: UUID
    name: str
    budget_limit: float | None = None
    created_at: datetime
    photo_ids: list[UUID] = []
    generation_ids: list[UUID] = []


# ── Budget ────────────────────────────────────────────────────────────────────

class BudgetLineItem(BaseModel):
    product_id: UUID
    name: str
    price: float
    source: Literal["ikea", "taobao", "scraped"]


class BudgetState(BaseModel):
    limit: float
    spent: float
    remaining: float
    over_budget: bool
    items: list[BudgetLineItem] = []


# ── Generation ────────────────────────────────────────────────────────────────

class GenerationPending(BaseModel):
    generation_id: UUID
    status: Literal["pending"]


class GeneratedProduct(BaseModel):
    product_id: UUID
    name: str
    price: float
    source: Literal["ikea", "taobao", "scraped"]
    buy_url: HttpUrl


class GenerationDone(BaseModel):
    generation_id: UUID
    status: Literal["done"]
    image_url: HttpUrl | None = None  # populated once image generation is wired up
    over_budget: bool
    total_cost: float
    products: list[GeneratedProduct] = []


# ── Products ──────────────────────────────────────────────────────────────────

class Product(BaseModel):
    product_id: UUID
    name: str
    price: float
    source: Literal["ikea", "taobao", "scraped"]
    image_url: HttpUrl | None = None
    buy_url: HttpUrl | None = None
    in_stock: bool = True
    style_tags: list[str] = []


class ProductDimensions(BaseModel):
    w: float
    d: float
    h: float


class ProductDetail(Product):
    dimensions: ProductDimensions | None = None


class ScrapedProduct(Product):
    scraped: bool = True


class Substitute(BaseModel):
    product_id: UUID
    name: str
    price: float
    similarity_score: float
    buy_url: HttpUrl | None = None


# ── Cart ──────────────────────────────────────────────────────────────────────

class CartItem(BaseModel):
    product_id: UUID
    name: str
    price: float
    currency: str | None = None
    source: str | None = None
    image_url: HttpUrl | None = None
    affiliate_url: HttpUrl | None = None


class CartResponse(BaseModel):
    total: float
    over_budget: bool = False
    items: list[CartItem] = []


class TrackClickResponse(BaseModel):
    tracked: bool
    redirect_url: HttpUrl
