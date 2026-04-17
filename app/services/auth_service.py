"""Authentication business logic service.

This service orchestrates user registration and login workflows.
It coordinates between UserStore and auth utilities (JWT, password hashing).
"""

from typing import Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.orm import User
from app.models.schemas import AuthResponse
from app.stores.user_store import UserStore
from app.services.auth import create_token, hash_password, verify_password


class AuthService:
    """Service for authentication business logic."""

    def register_user(
        self,
        email: str,
        password: str,
        name: Optional[str],
        db: Session
    ) -> AuthResponse:
        """Register a new user.

        Args:
            email: User's email address
            password: Plain text password
            name: Optional user name
            db: Database session

        Returns:
            AuthResponse with user_id and JWT token

        Raises:
            HTTPException: If email is already registered (409)
        """
        user_store = UserStore(db)

        # Check if email already exists
        if user_store.exists_by_email(email):
            raise HTTPException(
                status_code=409,
                detail="Email already registered"
            )

        # Create new user
        user = User(
            email=email,
            name=name,
            password_hash=hash_password(password),
        )

        # Add user to database
        user_store.add(user)
        db.commit()
        db.refresh(user)

        # Generate JWT token
        token = create_token(user.user_id)

        return AuthResponse(user_id=user.user_id, token=token)

    def login_user(
        self,
        email: str,
        password: str,
        db: Session
    ) -> AuthResponse:
        """Authenticate user and generate token.

        Args:
            email: User's email address
            password: Plain text password
            db: Database session

        Returns:
            AuthResponse with user_id and JWT token

        Raises:
            HTTPException: If credentials are invalid (401)
        """
        user_store = UserStore(db)

        # Find user by email
        user = user_store.find_by_email(email)

        # Verify user exists and password is correct
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # Generate JWT token
        token = create_token(user.user_id)

        return AuthResponse(user_id=user.user_id, token=token)
