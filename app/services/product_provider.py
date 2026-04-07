"""Abstract contract for any product data source (IKEA, Taobao, scraped, …).

Routers and other consumers depend only on this interface — never on a
concrete provider — honouring Dependency Inversion (DIP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import Product, ProductDetail


class ProductProvider(ABC):
    """Defines the surface area every product source must expose."""

    # Concrete classes must declare their source tag, e.g. "ikea" | "taobao"
    source: str

    @abstractmethod
    async def search(
        self,
        q: str | None = None,
        style: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock: bool | None = None,
        limit: int = 24,
    ) -> list[Product]:
        """Return products matching the given criteria."""

    @abstractmethod
    async def get_product(self, product_id: str) -> ProductDetail | None:
        """Return full product detail or None if not found.

        ``product_id`` is the opaque string token returned by ``search``
        (a deterministic UUID-5 derived from the provider's internal ID).
        """
