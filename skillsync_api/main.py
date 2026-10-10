"""FastAPI application exposing the SkillSync agentic analysis workflow."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from skillsync_api.models import AnalysisRequest, AnalysisResponse
from skillsync_api.security import (
    sanitize_analysis_request,
    verify_internal_auth,
)
from skillsync_api.service import (
    AnalysisService,
    ConfigurationError,
    InferenceError,
    MarketSearchError,
)

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SkillSync AI API",
    description="Live market-grounded skill gap analysis with zero-trust security and deterministic accuracy.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
)
allowed_origins = [
    origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """Reuse provider clients across requests."""
    return AnalysisService()


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.post(
    "/api/analyze-gap",
    response_model=AnalysisResponse,
    tags=["analysis"],
)
@limiter.limit("10/minute")
async def analyze_gap(
    request: Request,
    payload: AnalysisRequest,
    _auth: str = Security(verify_internal_auth),
) -> AnalysisResponse:
    """Search current market requirements and analyze a candidate's skill gap."""
    clean_request = sanitize_analysis_request(payload)
    try:
        return await get_analysis_service().analyze(clean_request)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
