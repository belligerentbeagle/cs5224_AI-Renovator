from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", status_code=201)
async def register():
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/login")
async def login():
    raise HTTPException(status_code=501, detail="Not implemented")
