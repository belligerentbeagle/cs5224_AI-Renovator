from fastapi import FastAPI
from app.routers import auth, projects, generation, products, cart

app = FastAPI(
    title="RoomAI API",
    version="0.1.0",
    description="Backend API for the RoomAI interior design app.",
)

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(generation.router)
app.include_router(products.router)
app.include_router(cart.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
