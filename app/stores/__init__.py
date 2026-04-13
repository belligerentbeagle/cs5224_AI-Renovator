"""Data access layer - all SQLAlchemy operations live here.

Stores use flush() to make IDs available but never commit().
Services control transaction boundaries.
"""

from app.stores.base import BaseStore
from app.stores.user_store import UserStore
from app.stores.project_store import ProjectStore
from app.stores.photo_store import PhotoStore
from app.stores.product_store import ProductStore
from app.stores.design_generation_store import DesignGenerationStore
from app.stores.generation_product_store import GenerationProductStore

__all__ = [
    "BaseStore",
    "UserStore",
    "ProjectStore",
    "PhotoStore",
    "ProductStore",
    "DesignGenerationStore",
    "GenerationProductStore",
]
