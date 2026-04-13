"""Cart business logic service.

This service aggregates products from all design generations in a project
and provides cart functionality.
"""

from uuid import UUID
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.schemas import CartItem, CartResponse, TrackClickResponse
from app.stores.project_store import ProjectStore
from app.stores.design_generation_store import DesignGenerationStore
from app.stores.product_store import ProductStore


class CartService:
    """Service for cart business logic."""

    def get_project_cart(
        self,
        project_id: UUID,
        user_id: UUID,
        db: Session
    ) -> CartResponse:
        """Get cart for a project.

        Aggregates all unique products from all design generations.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            db: Database session

        Returns:
            CartResponse with items and total

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        # Verify project ownership
        project_store = ProjectStore(db)
        project = project_store.get_by_id_and_user(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get all design generations for project
        generation_store = DesignGenerationStore(db)
        generations = generation_store.list_by_project(project_id)

        # Aggregate unique products
        product_store = ProductStore(db)
        seen: set = set()
        items: list[CartItem] = []
        total = 0.0

        for gen in generations:
            for gp in gen.generation_products:
                if gp.product_id in seen:
                    continue
                seen.add(gp.product_id)

                product = product_store.get_by_id(gp.product_id)
                if product:
                    price = float(product.price or 0)
                    total += price
                    items.append(CartItem(
                        product_id=product.product_id,
                        name=product.name or "",
                        price=price,
                        affiliate_url=product.product_url,
                    ))

        return CartResponse(total=round(total, 2), items=items)

    def track_affiliate_click(
        self,
        product_id: UUID,
        user_id: UUID,
        db: Session
    ) -> TrackClickResponse:
        """Track affiliate click and return redirect URL.

        Args:
            product_id: Product UUID
            user_id: User UUID (for potential analytics)
            db: Database session

        Returns:
            TrackClickResponse with redirect URL

        Raises:
            HTTPException: If product not found (404)
        """
        product_store = ProductStore(db)
        product = product_store.get_by_id(product_id)

        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        redirect_url = product.product_url or "https://www.ikea.com/sg/en/"

        return TrackClickResponse(tracked=True, redirect_url=redirect_url)
