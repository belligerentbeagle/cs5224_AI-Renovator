"""Project data access layer."""

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.orm import Project
from app.stores.base import BaseStore


class ProjectStore(BaseStore[Project]):
    """Store for Project entity operations."""

    def __init__(self, db: Session):
        """Initialize ProjectStore.

        Args:
            db: Database session
        """
        super().__init__(Project, db)

    def get_by_id_and_user(self, project_id: UUID, user_id: UUID) -> Optional[Project]:
        """Get project by ID and verify ownership.

        Args:
            project_id: Project UUID
            user_id: User UUID

        Returns:
            Project instance or None if not found or not owned by user
        """
        return self.db.query(Project).filter(
            Project.project_id == project_id,
            Project.user_id == user_id,
        ).first()

    def list_by_user(self, user_id: UUID) -> list[Project]:
        """List all projects for a user.

        Args:
            user_id: User UUID

        Returns:
            List of Project instances
        """
        return self.db.query(Project).filter(Project.user_id == user_id).all()

    def update_budget(self, project_id: UUID, budget_limit: float) -> Optional[Project]:
        """Update project budget limit.

        Args:
            project_id: Project UUID
            budget_limit: New budget limit

        Returns:
            Updated project or None if not found
        """
        project = self.get_by_id(project_id)
        if project:
            project.budget_limit = budget_limit
            self.db.flush()
        return project
