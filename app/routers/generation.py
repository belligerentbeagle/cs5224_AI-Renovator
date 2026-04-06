from uuid import UUID
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Generation"])


@router.post("/generate/room", status_code=202)
async def generate_room():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/generate/design-for-me", status_code=202)
async def design_for_me():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/generations/{id}")
async def poll_generation(id: UUID):
    raise HTTPException(status_code=501, detail="Not implemented")
