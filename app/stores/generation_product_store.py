"""GenerationProduct data access layer."""

from uuid import UUID
from sqlalchemy.orm import Session

from app.models.orm import GenerationProduct
from app.stores.base import BaseStore


class GenerationProductStore(BaseStore[GenerationProduct]):
    """Store for GenerationProduct entity operations."""

    def __init__(self, db: Session):
        """Initialize GenerationProductStore.

        Args:
            db: Database session
        """
        super().__init__(GenerationProduct, db)

    def add_batch(self, gen_products: list[GenerationProduct]) -> None:
        """Add multiple generation products in batch.

        Args:
            gen_products: List of GenerationProduct instances to add
        """
        for gp in gen_products:
            self.db.add(gp)
        self.db.flush()

    def list_by_design(self, design_id: UUID) -> list[GenerationProduct]:
        """List all products for a design generation.

        Args:
            design_id: Design generation UUID

        Returns:
            List of GenerationProduct instances
        """
        return self.db.query(GenerationProduct).filter(
            GenerationProduct.design_id == design_id
        ).all()
