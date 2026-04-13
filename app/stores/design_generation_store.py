"""DesignGeneration data access layer."""

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session, joinedload

from app.models.orm import DesignGeneration, GenerationStatus
from app.stores.base import BaseStore


class DesignGenerationStore(BaseStore[DesignGeneration]):
    """Store for DesignGeneration entity operations."""

    def __init__(self, db: Session):
        """Initialize DesignGenerationStore.

        Args:
            db: Database session
        """
        super().__init__(DesignGeneration, db)

    def get_with_products(self, design_id: UUID) -> Optional[DesignGeneration]:
        """Get design generation with products eagerly loaded.

        Args:
            design_id: Design generation UUID

        Returns:
            DesignGeneration instance with products loaded, or None if not found
        """
        return self.db.query(DesignGeneration).options(
            joinedload(DesignGeneration.generation_products)
        ).filter(DesignGeneration.design_id == design_id).first()

    def list_by_project(self, project_id: UUID) -> list[DesignGeneration]:
        """List all design generations for a project.

        Args:
            project_id: Project UUID

        Returns:
            List of DesignGeneration instances
        """
        return self.db.query(DesignGeneration).filter(
            DesignGeneration.project_id == project_id
        ).all()

    def update_status(self, design_id: UUID, status: GenerationStatus) -> None:
        """Update generation status.

        Args:
            design_id: Design generation UUID
            status: New status
        """
        generation = self.get_by_id(design_id)
        if generation:
            generation.status = status
            self.db.flush()

    def update_generated_photo(self, design_id: UUID, photo_id: UUID) -> None:
        """Update generated photo ID.

        Args:
            design_id: Design generation UUID
            photo_id: Photo UUID
        """
        generation = self.get_by_id(design_id)
        if generation:
            generation.generated_photo_id = photo_id
            self.db.flush()
