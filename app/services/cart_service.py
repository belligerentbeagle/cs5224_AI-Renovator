"""Cart business logic service."""

from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.schemas import CartItem, CartResponse, TrackClickResponse
from app.stores.affiliate_click_store import AffiliateClickStore
from app.stores.design_generation_store import DesignGenerationStore
from app.stores.generation_product_store import GenerationProductStore
from app.stores.product_store import ProductStore
from app.stores.project_store import ProjectStore

_UTM = {
    "utm_source": "roomstyle_ai",
    "utm_medium": "affiliate",
    "utm_campaign": "cart",
}


def _build_affiliate_url(raw_url) -> str | None:
    """Append UTM tracking params to a product URL without overwriting existing params."""
    if not raw_url:
        return None
    raw_url = str(raw_url)
    parsed = urlparse(raw_url)
    existing = parse_qs(parsed.query, keep_blank_values=True)
    merged = {**_UTM, **{k: v[0] for k, v in existing.items()}}
    return urlunparse(parsed._replace(query=urlencode(merged)))


class CartService:
    """Service for cart business logic. Controls all transactions (commit)."""

    def get_project_cart(
        self,
        project_id: UUID,
        user_id: UUID,
        db: Session,
        design_id: UUID | None = None,
    ) -> CartResponse:
        """Get cart for a project.

        Aggregates all unique products from design generations via a single JOIN query.
        Optionally filtered to a specific design generation.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            db: Database session
            design_id: Optional design generation UUID to filter by

        Returns:
            CartResponse with items, total, and over_budget flag

        Raises:
            HTTPException 404: If project not found or not owned by user
        """
        project = ProjectStore(db).get_by_id_and_user(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        products = ProductStore(db).get_products_for_project(project_id, design_id)

        items: list[CartItem] = []
        total = 0.0

        for product in products:
            price = float(product.price or 0)
            total += price
            items.append(CartItem(
                product_id=UUID(str(product.product_id)),
                name=str(product.name or ""),
                price=price,
                currency=str(product.currency) if product.currency else None,
                source=str(product.external_source) if product.external_source else None,
                image_url=str(product.image_url) if product.image_url else None,
                affiliate_url=_build_affiliate_url(product.product_url),
            ))

        budget = float(project.budget_limit) if project.budget_limit else None
        over_budget = budget is not None and round(total, 2) > budget

        return CartResponse(total=round(total, 2), over_budget=over_budget, items=items)

    def track_affiliate_click(
        self,
        product_id: UUID,
        user_id: UUID,
        project_id: UUID,
        db: Session,
    ) -> TrackClickResponse:
        """Record an affiliate click and return the redirect URL.

        Args:
            product_id: Product UUID that was clicked
            user_id: User UUID
            project_id: Project UUID
            db: Database session

        Returns:
            TrackClickResponse with redirect URL

        Raises:
            HTTPException 404: If product not found
        """
        product = ProductStore(db).get_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        redirect_url = _build_affiliate_url(product.product_url) or "https://www.ikea.com/sg/en/"

        AffiliateClickStore(db).log_click(
            user_id=user_id,
            project_id=project_id,
            product_id=product_id,
            redirect_url=redirect_url,
        )
        db.commit()

        return TrackClickResponse(tracked=True, redirect_url=redirect_url)

    def remove_cart_item(
        self,
        project_id: UUID,
        product_id: UUID,
        user_id: UUID,
        db: Session,
    ) -> None:
        """Remove a product from all generations in this project's cart.

        Args:
            project_id: Project UUID
            product_id: Product UUID to remove
            user_id: User UUID for ownership verification
            db: Database session

        Raises:
            HTTPException 404: If project not found or not owned by user
        """
        project = ProjectStore(db).get_by_id_and_user(project_id, user_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        design_ids = [
            gen.design_id
            for gen in DesignGenerationStore(db).list_by_project(project_id)
        ]

        GenerationProductStore(db).delete_by_product_and_designs(product_id, design_ids)
        db.commit()
