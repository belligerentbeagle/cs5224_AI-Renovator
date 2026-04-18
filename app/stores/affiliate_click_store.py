"""AffiliateClick data access layer."""

from uuid import UUID
from sqlalchemy.orm import Session

from app.models.orm import AffiliateClick
from app.stores.base import BaseStore


class AffiliateClickStore(BaseStore[AffiliateClick]):
    """Store for AffiliateClick entity operations."""

    def __init__(self, db: Session):
        super().__init__(AffiliateClick, db)

    def log_click(self, user_id: UUID, project_id: UUID, product_id: UUID, redirect_url: str) -> AffiliateClick:
        """Record an affiliate link click.

        Args:
            user_id: User UUID
            project_id: Project UUID
            product_id: Product UUID that was clicked
            redirect_url: The affiliate URL the user is being sent to

        Returns:
            The created AffiliateClick instance (flushed, not committed)
        """
        click = AffiliateClick(
            user_id=user_id,
            project_id=project_id,
            product_id=product_id,
            redirect_url=redirect_url,
        )
        return self.add(click)
