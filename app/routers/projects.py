from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel

from app.db import DB
from app.models.schemas import BudgetState
from app.models.schemas import Project as ProjectSchema
from app.services.auth import CurrentUser
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["Projects"])


# ── Request bodies ────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    title: str
    room_type: str | None = None
    style_prompt: str | None = None
    budget_limit: float | None = None


class UpdateBudgetRequest(BaseModel):
    budget_limit: float


class PresignRequest(BaseModel):
    file_name: str
    content_type: str = "image/jpeg"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectSchema, status_code=201)
async def create_project(body: CreateProjectRequest, db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.create_project(
        body.title,
        current_user,
        body.room_type,
        body.style_prompt,
        body.budget_limit,
        db
    )


@router.get("", response_model=list[ProjectSchema])
async def list_projects(db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.list_user_projects(current_user, db)


@router.get("/{id}", response_model=ProjectSchema)
async def get_project(id: UUID, db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.get_project(id, current_user, db)


@router.delete("/{id}", status_code=204)
async def delete_project(id: UUID, db: DB, current_user: CurrentUser):
    service = ProjectService()
    service.delete_project(id, current_user, db)


@router.post("/{id}/uploads/presign")
async def presign_upload_url(id: UUID, body: PresignRequest, db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.create_upload_presign(
        id,
        current_user,
        body.file_name,
        body.content_type,
        db
    )


@router.get("/{id}/budget", response_model=BudgetState)
async def get_budget(id: UUID, db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.get_budget(id, current_user, db)


@router.put("/{id}/budget", response_model=BudgetState)
async def update_budget(id: UUID, body: UpdateBudgetRequest, db: DB, current_user: CurrentUser):
    service = ProjectService()
    return service.update_budget(id, current_user, body.budget_limit, db)
