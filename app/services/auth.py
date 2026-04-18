"""JWT auth helpers + FastAPI dependency for protected routes."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

_SECRET_KEY = os.environ.get("JWT_SECRET", "change-me-in-production")
_ALGORITHM = "HS256"
_EXPIRE_HOURS = 24

# Use pbkdf2_sha256 for new hashes so local MVP flows don't rely on a fragile
# bcrypt binary/backend combination, but keep bcrypt enabled for verifying
# existing hashes during a gradual migration.
_pwd = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt"],
    default="pbkdf2_sha256",
    deprecated=["bcrypt"],
)
_bearer = HTTPBearer()


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_token(user_id: UUID) -> str:
    expire = datetime.utcnow() + timedelta(hours=_EXPIRE_HOURS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, _SECRET_KEY, algorithm=_ALGORITHM)


def _get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> UUID:
    try:
        payload = jwt.decode(credentials.credentials, _SECRET_KEY, algorithms=[_ALGORITHM])
        return UUID(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Use as a route dependency: `current_user: CurrentUser`
CurrentUser = Annotated[UUID, Depends(_get_current_user_id)]
