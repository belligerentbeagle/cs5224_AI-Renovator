from __future__ import annotations

import enum
import uuid

from sqlalchemy import (
    Column,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    DateTime,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db import Base


class GenerationStatus(str, enum.Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    user_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name          = Column(String(255))
    created_at    = Column(DateTime(timezone=True), default=datetime.utcnow)

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")


class Product(Base):
    __tablename__ = "products"

    product_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_source     = Column(String(100))
    external_product_id = Column(String(255))
    name                = Column(Text)
    product_url         = Column(Text)
    image_url           = Column(Text)
    price               = Column(Numeric(10, 2))
    currency            = Column(String(10))
    created_at          = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at          = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Project(Base):
    __tablename__ = "projects"

    project_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"))
    title        = Column(String(255))
    room_type    = Column(String(100))
    style_prompt = Column(Text)
    budget_limit = Column(Numeric(10, 2))
    created_at   = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at   = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    user        = relationship("User", back_populates="projects")
    photos      = relationship("Photo", back_populates="project", cascade="all, delete-orphan")
    generations = relationship("DesignGeneration", back_populates="project", cascade="all, delete-orphan")


class Photo(Base):
    __tablename__ = "photos"

    photo_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"))
    photo_type = Column(String(50))   # 'original' | 'generated'
    s3_key     = Column(Text, nullable=False)
    file_name  = Column(Text)
    mime_type  = Column(String(100))
    width      = Column(Integer)
    height     = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    project = relationship("Project", back_populates="photos")


class DesignGeneration(Base):
    __tablename__ = "design_generations"

    design_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id         = Column(UUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"))
    input_photo_id     = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="SET NULL"), nullable=True)
    generated_photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="SET NULL"), nullable=True)
    prompt_text        = Column(Text)
    style_name         = Column(String(100))
    status             = Column(SAEnum(GenerationStatus, name="generation_status", create_type=False), default=GenerationStatus.pending)
    created_at         = Column(DateTime(timezone=True), default=datetime.utcnow)

    project             = relationship("Project", back_populates="generations")
    generation_products = relationship("GenerationProduct", cascade="all, delete-orphan")


class GenerationProduct(Base):
    __tablename__ = "generation_products"

    design_id  = Column(UUID(as_uuid=True), ForeignKey("design_generations.design_id", ondelete="CASCADE"), primary_key=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.product_id", ondelete="CASCADE"), primary_key=True)
    x_position = Column(Float, primary_key=True)
    y_position = Column(Float, primary_key=True)
    width      = Column(Integer)
    height     = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    product = relationship("Product")


class AffiliateClick(Base):
    __tablename__ = "affiliate_clicks"

    click_id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    project_id   = Column(UUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    product_id   = Column(UUID(as_uuid=True), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    redirect_url = Column(Text, nullable=False)
    clicked_at   = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
