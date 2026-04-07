"""IKEA product provider backed by the ``ikea_api`` library.

Uses ``ikea_api``'s async execution layer (``run_async``) so the FastAPI
event loop is never blocked.  A single guest token is lazily acquired and
cached for the lifetime of the provider singleton.

NOTE: ``IngkaItems`` is intentionally NOT used — its endpoint URL encodes
the ``language`` field as the IKEA retail unit code, which is always wrong
for non-Russian locales (e.g. language="en" → "retail unit EN not found").
Instead, all data for search results is extracted directly from the richer
``Search`` response JSON, which already carries name, image, and price.
``PipItem`` is used for single-item detail fetches.

NOTE: ``ikea_api`` is an archived library (frozen Oct 2024).  Treat all
network calls as potentially fragile and test against live responses.
"""
from __future__ import annotations

import uuid
from typing import Any

import ikea_api
from ikea_api.wrappers.parsers.pip_item import parse_pip_item

from app.models.schemas import Product, ProductDetail
from app.services.product_provider import ProductProvider

# Deterministic namespace — always yields the same UUID-5 for a given item code.
_IKEA_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_IKEA_BASE = "https://www.ikea.com"


def _item_code_to_uuid(item_code: str) -> uuid.UUID:
    return uuid.uuid5(_IKEA_NS, f"ikea:{item_code}")


class _SearchResponseParser:
    """Isolates raw-JSON field access for the Search endpoint response.

    The IKEA search API (sik.search.blue.cdtapps.com) is undocumented.
    All access is defensive — yields empty results instead of raising on
    shape changes.

    Observed response structure::

        {
          "searchResultPage": {
            "products": {
              "main": {
                "items": [
                  {
                    "product": {
                      "id": "39375040",
                      "name": "SÖDERHAMN",
                      "typeName": "Sofa section",
                      "mainImageUrl": "https://...",
                      "salesPrice": {"numeral": 1999, "currencyCode": "USD"},
                      "pipUrl": "https://www.ikea.com/us/en/p/..."
                    }
                  }
                ]
              }
            }
          }
        }
    """

    @staticmethod
    def extract_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return raw["searchResultPage"]["products"]["main"]["items"]
        except (KeyError, TypeError):
            return []

    @staticmethod
    def parse_price(product: dict[str, Any]) -> float:
        try:
            return float(product["salesPrice"]["numeral"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def build_buy_url(product: dict[str, Any]) -> str | None:
        pip_url = product.get("pipUrl")
        if pip_url:
            return pip_url if pip_url.startswith("http") else _IKEA_BASE + pip_url
        rel_url = product.get("url")
        if rel_url:
            return _IKEA_BASE + rel_url
        return None


class IkeaProductProvider(ProductProvider):
    """Concrete ProductProvider that speaks to IKEA via ``ikea_api``.

    Two in-memory caches are kept for the lifetime of the singleton:
    - ``_product_cache``: str(UUID) → Product (populated during search)
    - ``_item_code_map``: str(UUID) → IKEA item code (for PipItem lookups)
    """

    source = "ikea"

    def __init__(self, country: str = "us", language: str = "en") -> None:
        self._constants = ikea_api.Constants(country=country, language=language)
        self._token: str | None = None
        self._product_cache: dict[str, Product] = {}
        self._item_code_map: dict[str, str] = {}  # str(UUID) → item_code
        self._parser = _SearchResponseParser()

    # ── Token management ──────────────────────────────────────────────────────

    async def _ensure_token(self) -> str:
        if self._token is None:
            ep = ikea_api.Auth(self._constants).get_guest_token()
            self._token = await ikea_api.run_async(ep)
        return self._token

    # ── ProductProvider interface ─────────────────────────────────────────────

    async def search(
        self,
        q: str | None = None,
        style: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock: bool | None = None,
        limit: int = 24,
    ) -> list[Product]:
        query = " ".join(part for part in [q, style] if part) or "furniture"

        ep = ikea_api.Search(self._constants).search(query, limit=limit)
        raw: dict = await ikea_api.run_async(ep)

        products = self._parse_search_response(raw)

        if min_price is not None:
            products = [p for p in products if p.price >= min_price]
        if max_price is not None:
            products = [p for p in products if p.price <= max_price]

        return products

    async def get_product(self, product_id: str) -> ProductDetail | None:
        cached = self._product_cache.get(product_id)
        if cached is None:
            return None

        item_code = self._item_code_map.get(product_id)
        if item_code is None:
            return None

        # Refresh price + URL from PipItem while reusing cached name/image.
        price = cached.price
        buy_url = str(cached.buy_url) if cached.buy_url else None
        try:
            pip_ep = ikea_api.PipItem(self._constants).get_item(item_code)
            pip_raw: dict = await ikea_api.run_async(pip_ep)
            parsed = parse_pip_item(pip_raw)
            if parsed:
                price = float(parsed.price)
                buy_url = parsed.url
        except Exception:
            pass  # Fall through to cached values

        return ProductDetail(
            product_id=cached.product_id,
            name=cached.name,
            price=price,
            source="ikea",
            image_url=cached.image_url,
            buy_url=buy_url,
            in_stock=cached.in_stock,
            style_tags=cached.style_tags,
            dimensions=None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_search_response(self, raw: dict[str, Any]) -> list[Product]:
        products: list[Product] = []

        for item in self._parser.extract_items(raw):
            product_data: dict | None = item.get("product") if isinstance(item, dict) else None
            if not isinstance(product_data, dict):
                continue

            item_code: str | None = product_data.get("id")
            if not item_code:
                continue

            uid = _item_code_to_uuid(item_code)
            uid_str = str(uid)

            name = product_data.get("name") or product_data.get("typeName") or "Unknown"
            image_url = product_data.get("mainImageUrl") or product_data.get("contextualImageUrl")
            price = self._parser.parse_price(product_data)
            buy_url = self._parser.build_buy_url(product_data)

            product = Product(
                product_id=uid,
                name=name,
                price=price,
                source="ikea",
                image_url=image_url,
                buy_url=buy_url,
                in_stock=True,  # Stock endpoint available but adds N+1 calls
                style_tags=[],
            )
            self._product_cache[uid_str] = product
            self._item_code_map[uid_str] = item_code
            products.append(product)

        return products
