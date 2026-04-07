"""Provider registry — maps source names to ProductProvider instances.

New providers (e.g. Taobao) are registered here in one place; the router
never needs to change.  FastAPI consumes ``get_providers`` as a dependency.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.services.ikea_provider import IkeaProductProvider
from app.services.product_provider import ProductProvider


@lru_cache(maxsize=1)
def _ikea() -> IkeaProductProvider:
    """Singleton IKEA provider — preserves the token + item-code cache."""
    return IkeaProductProvider(country="us", language="en")


# Registry: add new providers here when they're implemented.
_REGISTRY: dict[str, ProductProvider] = {
    "ikea": _ikea(),
    # "taobao": TaobaoProductProvider(...),
}


def get_providers(source: str | None = None) -> list[ProductProvider]:
    """Return the provider(s) relevant to the requested source.

    - ``source=None``  → all registered providers (fan-out search)
    - ``source="ikea"`` → only the IKEA provider
    - Unknown source   → empty list (caller should 404)
    """
    if source is None:
        return list(_REGISTRY.values())
    provider = _REGISTRY.get(source)
    return [provider] if provider else []


# Convenience type alias for use in route signatures.
Providers = Annotated[list[ProductProvider], Depends(get_providers)]
