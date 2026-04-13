"""Photo data access layer."""

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.orm import Photo
from app.stores.base import BaseStore


class PhotoStore(BaseStore[Photo]):
    """Store for Photo entity operations."""

    def __init__(self, db: Session):
        """Initialize PhotoStore.

        Args:
            db: Database session
        """
        super().__init__(Photo, db)

    def list_by_project(self, project_id: UUID) -> list[Photo]:
        """List all photos for a project.

        Args:
            project_id: Project UUID

        Returns:
            List of Photo instances
        """
        return self.db.query(Photo).filter(Photo.project_id == project_id).all()
