"""User data access layer."""

from typing import Optional
from sqlalchemy.orm import Session

from app.models.orm import User
from app.stores.base import BaseStore


class UserStore(BaseStore[User]):
    """Store for User entity operations."""

    def __init__(self, db: Session):
        """Initialize UserStore.

        Args:
            db: Database session
        """
        super().__init__(User, db)

    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email address.

        Args:
            email: Email address to search for

        Returns:
            User instance or None if not found
        """
        return self.db.query(User).filter(User.email == email).first()

    def exists_by_email(self, email: str) -> bool:
        """Check if user exists with given email.

        Args:
            email: Email address to check

        Returns:
            True if user exists, False otherwise
        """
        return self.db.query(User).filter(User.email == email).first() is not None
