"""Zero-Trust security controls, input sanitization, and prompt injection defense."""

import os
import re
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from skillsync_api.models import AnalysisRequest

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
EXPECTED_INTERNAL_KEY = os.getenv(
    "SKILLSYNC_INTERNAL_API_KEY", "skillsync-internal-secure-token-2026"
)

# Known prompt injection signatures and jailbreak token heuristics
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions?",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"disregard\s+(the\s+)?(previous|system|rules)",
    r"override\s+instructions?",
    r"drop\s+table",
    r"<script.*?>",
    r"jailbreak",
    r"act\s+as\s+an?\s+unrestricted",
    r"new\s+role\s*:",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def verify_internal_auth(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Enforces zero unauthorized access to backend worker endpoints."""
    expected = os.getenv("SKILLSYNC_INTERNAL_API_KEY", EXPECTED_INTERNAL_KEY).strip()
    if not api_key or api_key.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden: Invalid or missing service credential.",
        )
    return api_key


def sanitize_user_input(text: str, max_chars: int = 80) -> str:
    """Strips potential injection patterns and limits token bloat."""
    if not text:
        return ""

    # Strip dangerous format and shell/meta characters: <, >, {, }, [, ], \
    cleaned = re.sub(r"[<>{}[\]\\]", "", text).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip()

    # Heuristic prompt injection detection
    if INJECTION_REGEX.search(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malicious prompt injection vector detected in input payload.",
        )

    return cleaned


def sanitize_analysis_request(request: AnalysisRequest) -> AnalysisRequest:
    """Validates and sanitizes every user input field in the AnalysisRequest."""
    clean_role = sanitize_user_input(request.current_role, max_chars=80)
    clean_level = sanitize_user_input(request.current_level, max_chars=50)
    clean_target = sanitize_user_input(request.target_role, max_chars=80)

    clean_skills = [
        sanitize_user_input(s, max_chars=60)
        for s in request.user_skills
        if s and s.strip()
    ]

    if not clean_skills:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one valid skill is required after sanitization.",
        )

    return AnalysisRequest(
        current_role=clean_role,
        current_level=clean_level,
        user_skills=clean_skills[:50],  # Bound maximum 50 skills
        target_role=clean_target,
    )
