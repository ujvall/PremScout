"""FastAPI entry point for the PremAI Football Intelligence Platform."""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import api_router
from backend.api.schemas.common import RootResponse

# Setup application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("premai")

# Allowed CORS origins for frontend development (configurable)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="PremAI Football Intelligence API",
    description=(
        "Production REST API for the PremAI Football Intelligence Platform.\n\n"
        "Integrates three specialized machine learning modules:\n"
        "1. **Transfer Value Predictor**: Continuous player transfer market valuation.\n"
        "2. **Match Outcome Predictor**: Pre-match 3-class probabilistic outcome forecasting (Home/Draw/Away).\n"
        "3. **Player Scouting & Replacement**: 16-dimensional vector similarity engine with position-aware recruitment filters."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Root"],
    summary="API Root Service Discovery",
    description="Returns platform identity, operational status, and interactive documentation link.",
)
def read_root() -> RootResponse:
    """Service discovery root endpoint."""
    return RootResponse(
        name="PremAI Football Intelligence API",
        status="running",
        version="1.0.0",
        docs_url="/docs",
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle application value errors cleanly with standard JSON structure."""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
