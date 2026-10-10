"""Validated request and response models for the SkillSync API."""

from typing import Annotated

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


class AnalysisResponse(BaseModel):
    """Structured market-grounded skill-gap analysis."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    match_score_percentage: int = Field(ge=0, le=100)
    market_benchmarks_summary: str = Field(min_length=1)
    skill_breakdown: SkillBreakdown
    step_by_step_learning_roadmap: list[LearningRoadmapStep]
    interview_prep_questions: list[str]
    search_sources: list[str]
