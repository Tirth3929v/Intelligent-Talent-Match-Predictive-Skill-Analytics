"""Validated request and response models for the SkillSync API."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    """Candidate profile and target role to analyze."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    current_role: str = Field(min_length=1, max_length=200)
    current_level: str = Field(min_length=1, max_length=100)
    user_skills: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        min_length=1, max_length=100
    )
    target_role: str = Field(min_length=1, max_length=200)


class SkillBreakdown(BaseModel):
    """Categorized skills for the candidate's target role."""

    model_config = ConfigDict(extra="forbid", strict=True)

    relevant_matched: list[str]
    transferable_or_secondary: list[str]
    missing_critical: list[str]


class LearningRoadmapStep(BaseModel):
    """One actionable phase in a skill-development roadmap."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    phase: str = Field(min_length=1)
    skills: list[str]
    recommended_project: str = Field(min_length=1)


class TavilySourceItem(BaseModel):
    """A verified live market grounding resource returned from Tavily Search."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(default="Verified Resource")
    url: str = Field(min_length=1)
    snippet: str = Field(default="")
    resource_type: str = Field(default="Documentation / Live Market")


class TierExecutionTelemetry(BaseModel):
    """Telemetry captured for a specific Nemotron model tier execution."""

    model_config = ConfigDict(extra="ignore")

    tier: str
    model_used: str
    latency_ms: float
    tokens_used: int
    tokens_per_sec: float = 0.0
    estimated_savings_pct: int = 0
    request_id: str | None = None


class RouterTelemetry(BaseModel):
    """Inference and cost telemetry across Nebius Token Factory model tiers."""

    model_config = ConfigDict(extra="ignore")

    active_tier: str = Field(default="reasoning")
    model_used: str = Field(default="")
    total_latency_ms: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    average_tokens_per_sec: float = Field(default=0.0)
    estimated_cost_savings_pct: int = Field(default=0)
    tiers_executed: list[TierExecutionTelemetry] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Structured market-grounded skill-gap analysis."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    match_score_percentage: int = Field(ge=0, le=100)
    market_benchmarks_summary: str = Field(min_length=1)
    skill_breakdown: SkillBreakdown
    step_by_step_learning_roadmap: list[LearningRoadmapStep]
    interview_prep_questions: list[str]
    search_sources: list[str]
    model_provenance: str = Field(default="")
    router_telemetry: RouterTelemetry | None = Field(default=None)
    grounded_resources: list[TavilySourceItem] = Field(default_factory=list)


# ── Feature 1: Capstone Project Blueprint Models ──────────────────────────────
class CapstoneBlueprintRequest(BaseModel):
    """Request to auto-generate a tailored portfolio starter repo."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    target_role: str = Field(min_length=1, max_length=200)
    missing_skills: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        min_length=1, max_length=30
    )
    matched_skills: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        default_factory=list, max_length=30
    )
    project_focus: str | None = Field(default=None, max_length=200)


class CapstoneBlueprintResponse(BaseModel):
    """Generated portfolio starter blueprint with README, architecture, and code."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    project_title: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    architecture_summary: str = Field(min_length=1)
    bridged_skills: list[str] = Field(default_factory=list)
    readme_markdown: str = Field(min_length=1)
    starter_code_py: str = Field(min_length=1)
    requirements_txt: str = Field(min_length=1)
    model_tier_used: str = Field(default="reasoning")
    latency_ms: float = Field(default=0.0)


# ── Feature 2: Live Job Match & Resume Bullet Rewriter Models ─────────────────
class LiveJobOpening(BaseModel):
    """A real-time 2026 job posting retrieved via Tavily Search API."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    company: str = Field(default="Industry Employer")
    url: str = Field(min_length=1)
    location: str = Field(default="Remote / Hybrid")
    snippet: str = Field(default="")
    required_skills: list[str] = Field(default_factory=list)


class RewrittenBullet(BaseModel):
    """A candidate resume bullet rewritten with STAR format and live keywords."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    original_bullet: str = Field(min_length=1)
    rewritten_bullet: str = Field(min_length=1)
    target_skills_highlighted: list[str] = Field(default_factory=list)
    impact_rationale: str = Field(default="")


class JobMatchAndRewriteRequest(BaseModel):
    """Request to fetch live openings and rewrite resume bullets to match them."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    target_role: str = Field(min_length=1, max_length=200)
    current_role: str = Field(min_length=1, max_length=200)
    user_skills: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        min_length=1, max_length=100
    )
    existing_bullets: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        min_length=1, max_length=10
    )


class JobMatchAndRewriteResponse(BaseModel):
    """Result containing live openings and tailored resume bullets."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    target_role: str
    live_openings: list[LiveJobOpening] = Field(default_factory=list)
    rewritten_bullets: list[RewrittenBullet] = Field(default_factory=list)
    market_demand_notes: str = Field(default="")
    model_tier_used: str = Field(default="balanced")
    latency_ms: float = Field(default=0.0)
