"""Generation business logic service.

This service orchestrates the room generation workflow including:
- Creating generation records
- Running the background generation pipeline
- Managing the status polling endpoint
"""

import os
import uuid as _uuid
from typing import Literal, Optional
from uuid import UUID
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.orm import (
    DesignGeneration,
    GenerationProduct,
    GenerationStatus,
    Photo,
    Product as ProductORM,
)
from app.models.schemas import GeneratedProduct, GenerationDone, GenerationPending
from app.stores.project_store import ProjectStore
from app.stores.design_generation_store import DesignGenerationStore
from app.stores.generation_product_store import GenerationProductStore
from app.stores.photo_store import PhotoStore
from app.stores.product_store import ProductStore


_S3_BUCKET = os.environ.get("S3_BUCKET", "roomstyle-cs5224")
_AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")


class GenerationService:
    """Service for generation business logic."""

    def submit_room_generation(
        self,
        project_id: UUID,
        user_id: UUID,
        photo_id: UUID,
        style_name: str,
        furniture_items: list[dict],
        prompt_text: Optional[str],
        db: Session
    ) -> GenerationPending:
        """Submit a room generation request with user-selected furniture.

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            photo_id: Input photo UUID
            style_name: Style name (e.g., "modern")
            furniture_items: List of furniture item dicts
            prompt_text: Optional custom prompt
            db: Database session

        Returns:
            GenerationPending response

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        # Verify project ownership
        self._verify_project_ownership(project_id, user_id, db)

        # Create generation record
        generation_store = DesignGenerationStore(db)
        gen = DesignGeneration(
            project_id=project_id,
            input_photo_id=photo_id,
            style_name=style_name,
            prompt_text=prompt_text,
            status=GenerationStatus.pending,
        )
        generation_store.add(gen)
        db.commit()
        db.refresh(gen)

        return GenerationPending(generation_id=gen.design_id, status="pending")

    def submit_design_for_me(
        self,
        project_id: UUID,
        user_id: UUID,
        photo_id: Optional[UUID],
        style_name: str,
        prompt_text: Optional[str],
        db: Session
    ) -> GenerationPending:
        """Submit a design-for-me request (auto-search furniture).

        Args:
            project_id: Project UUID
            user_id: User UUID for ownership verification
            photo_id: Optional input photo UUID
            style_name: Style name (e.g., "scandinavian")
            prompt_text: Optional custom prompt
            db: Database session

        Returns:
            GenerationPending response

        Raises:
            HTTPException: If project not found or not owned by user (404)
        """
        # Verify project ownership
        self._verify_project_ownership(project_id, user_id, db)

        # Create generation record
        generation_store = DesignGenerationStore(db)
        gen = DesignGeneration(
            project_id=project_id,
            input_photo_id=photo_id,
            style_name=style_name,
            prompt_text=prompt_text,
            status=GenerationStatus.pending,
        )
        generation_store.add(gen)
        db.commit()
        db.refresh(gen)

        return GenerationPending(generation_id=gen.design_id, status="pending")

    def get_generation_status(
        self,
        generation_id: UUID,
        user_id: UUID,
        db: Session
    ) -> GenerationPending | GenerationDone:
        """Get generation status (poll endpoint).

        Args:
            generation_id: Generation UUID
            user_id: User UUID (for future verification)
            db: Database session

        Returns:
            GenerationPending if still processing, GenerationDone if complete

        Raises:
            HTTPException: If generation not found (404) or failed (500)
        """
        generation_store = DesignGenerationStore(db)
        gen = generation_store.get_by_id(generation_id)

        if not gen:
            raise HTTPException(status_code=404, detail="Generation not found")

        if gen.status == GenerationStatus.pending:
            return GenerationPending(generation_id=gen.design_id, status="pending")

        if gen.status == GenerationStatus.failed:
            raise HTTPException(status_code=500, detail="Generation failed")

        # Build completed response
        product_store = ProductStore(db)
        project_store = ProjectStore(db)
        photo_store = PhotoStore(db)

        products: list[GeneratedProduct] = []
        total = 0.0

        for gp in gen.generation_products:
            product = product_store.get_by_id(gp.product_id)
            if product:
                price = float(product.price or 0)
                total += price
                products.append(GeneratedProduct(
                    product_id=product.product_id,
                    name=product.name or "",
                    price=price,
                    source=product.external_source or "scraped",
                    buy_url=product.product_url or "https://www.ikea.com/sg/en/",
                ))

        project = project_store.get_by_id(gen.project_id)
        budget_limit = float(project.budget_limit) if project and project.budget_limit else None
        over_budget = total > budget_limit if budget_limit else False

        if gen.generated_photo_id:
            photo = photo_store.get_by_id(gen.generated_photo_id)
            if photo:
                from app.services.s3 import presign_download
                image_url = presign_download(photo.s3_key)
            else:
                image_url = None
        else:
            image_url = None

        return GenerationDone(
            generation_id=gen.design_id,
            status="done",
            image_url=image_url,
            over_budget=over_budget,
            total_cost=total,
            products=products,
        )

    async def run_generation_pipeline(
        self,
        design_id: str,
        style_name: str,
        furniture_items: list[dict],
        prompt_text: Optional[str]
    ) -> None:
        """Run the complete generation pipeline in background.

        Steps:
        1. Auto-search IKEA if no furniture provided
        2. Upsert products and create GenerationProduct joins
        3. Download furniture images
        4. Fetch input photo
        5. Generate image via Gemini
        6. Upload generated image
        7. Mark status as completed (or failed on error)

        Args:
            design_id: Design generation ID
            style_name: Style name
            furniture_items: List of furniture item dicts
            prompt_text: Optional custom prompt
        """
        import asyncio
        import httpx
        from app.services.gemini_generation import generate_room_image

        db = SessionLocal()
        try:
            generation_store = DesignGenerationStore(db)
            product_store = ProductStore(db)
            gen_product_store = GenerationProductStore(db)
            photo_store = PhotoStore(db)

            # Step 1: Auto-search IKEA if no furniture
            if not furniture_items:
                from app.services.provider_registry import _ikea
                provider = _ikea()
                results = await provider.search(q=style_name, style=style_name)
                furniture_items = [
                    {
                        "name": p.name,
                        "image_url": str(p.image_url) if p.image_url else None,
                        "product_id": str(p.product_id),
                        "price": p.price,
                        "source": p.source,
                        "buy_url": str(p.buy_url) if p.buy_url else None,
                    }
                    for p in results[:8]
                ]

            # Step 2: Upsert products + generation join rows
            saved: list[tuple[ProductORM, str | None]] = []  # (orm, image_url)
            for i, item in enumerate(furniture_items[:8]):
                external_id = str(item.get("product_id") or _uuid.uuid4())

                # Upsert product
                existing = product_store.find_by_external_id(
                    item.get("source", "ikea"),
                    external_id
                )

                if not existing:
                    existing = ProductORM(
                        external_source=item.get("source", "ikea"),
                        external_product_id=external_id,
                        name=item.get("name"),
                        product_url=item.get("buy_url"),
                        image_url=item.get("image_url"),
                        price=item.get("price", 0),
                        currency="SGD",
                    )
                    product_store.add(existing)

                # Create generation product join
                gen_product = GenerationProduct(
                    design_id=design_id,
                    product_id=existing.product_id,
                    x_position=float(i % 4) * 0.25,
                    y_position=float(i // 4) * 0.5,
                )
                gen_product_store.add(gen_product)

                saved.append((existing, item.get("image_url")))

            db.flush()

            # Step 3: Download furniture product images
            furniture_image_data: list[tuple[bytes, str, str]] = []
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                for product_orm, image_url in saved:
                    if not image_url:
                        continue
                    try:
                        resp = await client.get(image_url)
                        resp.raise_for_status()
                        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                        furniture_image_data.append((resp.content, mime, product_orm.name or "furniture"))
                    except Exception:
                        pass  # skip items whose images can't be fetched

            # Step 4 & 5: Gemini generation
            gen = generation_store.get_by_id(design_id)
            if gen and gen.input_photo_id:
                photo = photo_store.get_by_id(gen.input_photo_id)
                if photo:
                    output_key = await asyncio.to_thread(
                        generate_room_image,
                        photo.s3_key,
                        design_id,
                        style_name,
                        prompt_text,
                        furniture_image_data,
                    )

                    # Step 6: Persist generated image
                    gen_photo = Photo(
                        project_id=gen.project_id,
                        photo_type="generated",
                        s3_key=output_key,
                        file_name="output.jpg",
                        mime_type="image/jpeg",
                    )
                    photo_store.add(gen_photo)
                    generation_store.update_generated_photo(design_id, gen_photo.photo_id)

            # Mark as completed
            if gen:
                generation_store.update_status(design_id, GenerationStatus.completed)
            db.commit()

        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("Background generation pipeline completely failed for design_id %s", design_id)
            db.rollback()
            gen = generation_store.get_by_id(design_id)
            if gen:
                generation_store.update_status(design_id, GenerationStatus.failed)
                db.commit()
        finally:
            db.close()

    def _verify_project_ownership(self, project_id: UUID, user_id: UUID, db: Session) -> None:
        """Verify project exists and is owned by user.

        Args:
            project_id: Project UUID
            user_id: User UUID
            db: Database session

        Raises:
            HTTPException: If project not found or not owned (404)
        """
        project_store = ProjectStore(db)
        project = project_store.get_by_id_and_user(project_id, user_id)

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
