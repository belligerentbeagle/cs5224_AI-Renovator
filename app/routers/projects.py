from uuid import UUID
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", status_code=201)
async def create_project():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("")
async def list_projects():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{id}")
async def get_project(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/{id}", status_code=204)
async def delete_project(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/{id}/uploads/presign")
async def presign_upload(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/{id}/budget")
async def get_budget(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.put("/{id}/budget")
async def update_budget(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")
