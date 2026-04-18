"""Product data access layer."""

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.orm import Product
from app.stores.base import BaseStore


class ProductStore(BaseStore[Product]):
    """Store for Product entity operations."""

    def __init__(self, db: Session):
        """Initialize ProductStore.

        Args:
            db: Database session
        """
        super().__init__(Product, db)

    def find_by_external_id(self, source: str, external_id: str) -> Optional[Product]:
        """Find product by external source and ID.

        Args:
            source: External source (e.g., 'ikea')
            external_id: External product ID

        Returns:
            Product instance or None if not found
        """
        return self.db.query(Product).filter(
            Product.external_source == source,
            Product.external_product_id == external_id
        ).first()

    def upsert_by_external_id(self, product: Product) -> Product:
        """Insert product or return existing one if external ID exists.

        Args:
            product: Product instance to insert

        Returns:
            Existing product or newly inserted product
        """
        existing = self.find_by_external_id(
            product.external_source,
            product.external_product_id
        )
        if existing:
            return existing

        return self.add(product)

    def get_by_ids(self, product_ids: list[UUID]) -> list[Product]:
        """Get multiple products by their IDs.

        Args:
            product_ids: List of product UUIDs

        Returns:
            List of Product instances
        """
        return self.db.query(Product).filter(Product.product_id.in_(product_ids)).all()

    def get_products_for_project(
        self,
        project_id: UUID,
        design_id: Optional[UUID] = None,
    ) -> list[Product]:
        """Get all unique products linked to a project's design generations.

        Uses a single JOIN query across design_generations → generation_products → products.
        Optionally filtered to a single design generation.

        Args:
            project_id: Project UUID
            design_id: Optional design generation UUID to filter by

        Returns:
            Deduplicated list of Product instances
        """
        from app.models.orm import DesignGeneration, GenerationProduct

        q = (
            self.db.query(Product)
            .join(GenerationProduct, GenerationProduct.product_id == Product.product_id)
            .join(DesignGeneration, DesignGeneration.design_id == GenerationProduct.design_id)
            .filter(DesignGeneration.project_id == project_id)
        )
        if design_id:
            q = q.filter(DesignGeneration.design_id == design_id)
        return q.distinct(Product.product_id).all()
