from dotenv import load_dotenv

load_dotenv()  # loads .env for local dev; no-op on Lambda

from app.services.secrets import load_secrets
load_secrets()  # injects AWS Secrets Manager values into os.environ (Lambda only)

import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

# Configure basic logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(handler)

from app.db import Base, _engine
from app.models import orm  # noqa: F401 – ensures all models are registered before create_all
from app.routers import auth, cart, generation, products, projects


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("DATABASE_URL"):
        Base.metadata.create_all(bind=_engine(), checkfirst=True)
    yield


app = FastAPI(
    title="RoomStyle API",
    version="0.1.0",
    description="Backend API for the RoomStyle interior design app.",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to CloudFront domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(generation.router)
app.include_router(products.router)
app.include_router(cart.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Lambda entry point (Mangum adapts ASGI → Lambda event format)
handler = Mangum(app, lifespan="off")
