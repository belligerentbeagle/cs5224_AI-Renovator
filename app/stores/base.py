"""Base store providing common CRUD operations for all entity stores.

Stores are responsible for ALL SQLAlchemy operations. They use flush() to make
IDs available but NEVER commit(). Services control transactions.
"""

from typing import Generic, TypeVar, Type, Optional
from sqlalchemy.orm import Session

T = TypeVar('T')


class BaseStore(Generic[T]):
    """Base store with common CRUD operations.

    Each specific store inherits from this and adds:
    - Finder methods (e.g., find_by_email, find_by_user_id)
    - Relationship queries (e.g., get_project_with_generations)
    - Upsert operations (e.g., upsert_product_by_external_id)
    """

    def __init__(self, model: Type[T], db: Session):
        """Initialize store with model class and database session.

        Args:
            model: SQLAlchemy model class (e.g., User, Project)
            db: Database session
        """
        self.model = model
        self.db = db

    def add(self, entity: T) -> T:
        """Add entity to database and flush.

        Flush makes the ID available without committing the transaction.
        The service layer controls when to commit.

        Args:
            entity: Entity instance to add

        Returns:
            The added entity with ID populated
        """
        self.db.add(entity)
        self.db.flush()
        return entity

    def get_by_id(self, entity_id) -> Optional[T]:
        """Get entity by primary key ID.

        Args:
            entity_id: Primary key value

        Returns:
            Entity instance or None if not found
        """
        return self.db.query(self.model).filter(
            self.model.__mapper__.primary_key[0] == entity_id
        ).first()

    def delete(self, entity: T) -> None:
        """Delete entity from database.

        Does not commit - service layer controls transactions.

        Args:
            entity: Entity instance to delete
        """
        self.db.delete(entity)
        self.db.flush()
