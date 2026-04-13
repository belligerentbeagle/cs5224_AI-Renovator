from fastapi import APIRouter
from pydantic import BaseModel

from app.db import DB
from app.models.schemas import AuthResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(body: RegisterRequest, db: DB):
    service = AuthService()
    return service.register_user(body.email, body.password, body.name, db)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest, db: DB):
    service = AuthService()
    return service.login_user(body.email, body.password, db)
