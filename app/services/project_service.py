"""Project business logic service.

This service orchestrates project management workflows including:
- Project CRUD operations
- Photo upload presigning
- Budget calculation and management
"""

import uuid
from datetime import datetime
from typing import Optional
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.orm import Photo, Product, Project
from app.models.schemas import BudgetLineItem, BudgetState
from app.models.schemas import Project as ProjectSchema
from app.stores.project_store import ProjectStore
from app.stores.photo_store import PhotoStore
from app.stores.product_store import ProductStore
from app.services.s3 import presign_upload


class ProjectService:
    """Service for project business logic."""

    def create_project(
        self,
        title: str,
        user_id: UUID,
        room_type: Optional[str],
        style_prompt: Optional[str],
        budget_limit: Optional[float],
        db: Session
    ) -> ProjectSchema:
        """Create a new project.

        Args:
            title: Project title
            user_id: Owner user ID
            room_type: Optional room type
            style_prompt: Optional style description
            budget_limit: Optional budget limit
            db: Database session

        Returns:
            ProjectSchema with created project details
        """
        project_store = ProjectStore(db)

        project = Project(
            user_id=user_id,
            title=title,
            room_type=room_type,
            style_prompt=style_prompt,
            budget_limit=budget_limit,
        )

        project_store.add(project)
        db.commit()
        db.refresh(project)

        return self._to_schema(project)

    def list_user_projects(self, user_id: UUID, db: Session) -> list[ProjectSchema]:
        """List all projects for a user.

        Args:
            user_id: User UUID
            db: Database session

        Returns:
            List of ProjectSchema instances
        """
        project_store = ProjectStore(db)
        projects = project_store.list_by_user(user_id)
        return [self._to_schema(p) for p in projects]

    def get_project(self, project_id: UUID, user_id: UUID, db: Session) -> ProjectSchema:
        """Get a project by ID.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            db: Database session

        Returns:
            ProjectSchema

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        project = self._get_owned_project(project_id, user_id, db)
        return self._to_schema(project)

    def delete_project(self, project_id: UUID, user_id: UUID, db: Session) -> None:
        """Delete a project.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            db: Database session

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        project_store = ProjectStore(db)
        project = self._get_owned_project(project_id, user_id, db)

        project_store.delete(project)
        db.commit()

    def create_upload_presign(
        self,
        project_id: UUID,
        user_id: UUID,
        file_name: str,
        content_type: str,
        db: Session
    ) -> dict:
        """Create presigned URL for photo upload.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            file_name: Name of file to upload
            content_type: MIME type of file
            db: Database session

        Returns:
            Dict with photo_id, upload_url, s3_key, expires_in

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        project = self._get_owned_project(project_id, user_id, db)
        photo_store = PhotoStore(db)

        # Generate S3 key
        s3_key = f"uploads/{project.project_id}/{uuid.uuid4()}/{file_name}"

        # Create photo record
        photo = Photo(
            project_id=project.project_id,
            photo_type="original",
            s3_key=s3_key,
            file_name=file_name,
            mime_type=content_type,
        )

        photo_store.add(photo)
        db.commit()
        db.refresh(photo)

        # Generate presigned URL
        upload_url = presign_upload(s3_key, content_type)

        return {
            "photo_id": photo.photo_id,
            "upload_url": upload_url,
            "s3_key": s3_key,
            "expires_in": 3600,
        }

    def get_budget(self, project_id: UUID, user_id: UUID, db: Session) -> BudgetState:
        """Get budget state for a project.

        Aggregates all products from all generations and calculates totals.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            db: Database session

        Returns:
            BudgetState with limit, spent, remaining, items

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        project = self._get_owned_project(project_id, user_id, db)
        return self._build_budget(project, db)

    def update_budget(
        self,
        project_id: UUID,
        user_id: UUID,
        budget_limit: float,
        db: Session
    ) -> BudgetState:
        """Update project budget limit.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            budget_limit: New budget limit
            db: Database session

        Returns:
            BudgetState with updated limit

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        project = self._get_owned_project(project_id, user_id, db)
        project_store = ProjectStore(db)

        project.budget_limit = budget_limit
        db.commit()
        db.refresh(project)

        return self._build_budget(project, db)

    def _get_owned_project(self, project_id: UUID, user_id: UUID, db: Session) -> Project:
        """Get project and verify ownership.

        Args:
            project_id: Project UUID
            user_id: User UUID
            db: Database session

        Returns:
            Project instance

        Raises:
            HTTPException: If not found or not owned (404)
        """
        project_store = ProjectStore(db)
        project = project_store.get_by_id_and_user(project_id, user_id)

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return project

    def _to_schema(self, project: Project) -> ProjectSchema:
        """Convert ORM Project to schema.

        Args:
            project: Project ORM instance

        Returns:
            ProjectSchema
        """
        return ProjectSchema(
            project_id=project.project_id,
            name=project.title or "",
            budget_limit=float(project.budget_limit) if project.budget_limit is not None else None,
            created_at=project.created_at or datetime.utcnow(),
            photo_ids=[p.photo_id for p in project.photos],
            generation_ids=[g.design_id for g in project.generations],
        )

    def _build_budget(self, project: Project, db: Session) -> BudgetState:
        """Calculate budget state from project generations.

        Args:
            project: Project ORM instance
            db: Database session

        Returns:
            BudgetState with aggregated products
        """
        product_store = ProductStore(db)
        items: list[BudgetLineItem] = []
        seen: set = set()

        # Aggregate unique products from all generations
        for gen in project.generations:
            for gp in gen.generation_products:
                if gp.product_id in seen:
                    continue
                seen.add(gp.product_id)

                product = product_store.get_by_id(gp.product_id)
                if product:
                    items.append(BudgetLineItem(
                        product_id=product.product_id,
                        name=product.name or "",
                        price=float(product.price or 0),
                        source=product.external_source or "scraped",
                    ))

        spent = sum(i.price for i in items)
        limit = float(project.budget_limit) if project.budget_limit is not None else 0.0

        return BudgetState(
            limit=limit,
            spent=spent,
            remaining=max(0.0, limit - spent),
            over_budget=spent > limit if limit > 0 else False,
            items=items,
        )
