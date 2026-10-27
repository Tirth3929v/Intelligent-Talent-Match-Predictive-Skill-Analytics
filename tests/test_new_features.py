"""Unit tests for Capstone Blueprint Generator and Live Job Match & Bullet Rewriter."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from skillsync_api.models import (
    CapstoneBlueprintRequest,
    CapstoneBlueprintResponse,
    JobMatchAndRewriteRequest,
    JobMatchAndRewriteResponse,
)
from skillsync_api.service import AnalysisService, ServiceSettings


class NewFeatureAgenticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = ServiceSettings(
            tavily_api_key="test-tavily-key",
            nebius_api_key="test-nebius-key",
            nebius_model="nvidia/Nemotron-3-Ultra-550B",
            fast_model="nvidia/Nemotron-3-Nano-30B",
            super_model="nvidia/Nemotron-3-Super-120B",
            ultra_model="nvidia/Nemotron-3-Ultra-550B",
        )
        self.tavily = Mock()
        self.openai = Mock()

    async def test_generate_capstone_blueprint_returns_valid_structure(self) -> None:
        mock_blueprint = {
            "project_title": "AI Agent Orchestration Hub",
            "problem_statement": "Bridge competencies in LangGraph and Vector Databases.",
            "architecture_summary": "Stateful agentic graph with Qdrant vector retrieval.",
            "bridged_skills": ["LangGraph", "Vector Databases"],
            "readme_markdown": "# AI Agent Hub\n\nSetup instructions.",
            "starter_code_py": "print('Agent Hub Ready')",
            "requirements_txt": "fastapi>=0.110.0\nlanggraph>=0.1.0\n",
        }
        mock_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(mock_blueprint))
                )
            ],
            model="nvidia/Nemotron-3-Ultra-550B",
            id="chatcmpl-ultra-blueprint-1",
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=220, total_tokens=370),
        )
        self.openai.chat.completions.create = AsyncMock(return_value=mock_completion)
        service = AnalysisService(self.settings, self.tavily, self.openai)

        req = CapstoneBlueprintRequest(
            target_role="AI Engineer",
            missing_skills=["LangGraph", "Vector Databases"],
            matched_skills=["Python", "FastAPI"],
        )
        blueprint = await service.generate_capstone_blueprint(req)

        self.assertIsInstance(blueprint, CapstoneBlueprintResponse)
        self.assertEqual(blueprint.project_title, "AI Agent Orchestration Hub")
        self.assertIn("LangGraph", blueprint.bridged_skills)
        self.assertIn("fastapi", blueprint.requirements_txt)
        self.assertEqual(blueprint.model_tier_used, "nvidia/Nemotron-3-Ultra-550B")

    async def test_match_jobs_and_rewrite_bullets(self) -> None:
        self.tavily.search.return_value = {
            "results": [
                {
                    "title": "Staff AI Engineer at Tech Corp",
                    "url": "https://example.com/job1",
                    "content": "Looking for builders with FastAPI, RAG, and multi-agent systems.",
                }
            ]
        }
        mock_rewrite_payload = {
            "rewritten_bullets": [
                {
                    "original_bullet": "Built REST APIs with FastAPI.",
                    "rewritten_bullet": "Architected low-latency FastAPI microservices integrating RAG pipelines, reducing endpoint response time by 35%.",
                    "target_skills_highlighted": ["FastAPI", "RAG"],
                    "impact_rationale": "Quantified latency reduction and aligned with live 2026 hiring specifications.",
                }
            ],
            "market_demand_notes": "Strong market demand for RAG and agentic orchestration.",
        }
        mock_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(mock_rewrite_payload))
                )
            ],
            model="nvidia/Nemotron-3-Super-120B",
            id="chatcmpl-super-rewrite-1",
            usage=SimpleNamespace(prompt_tokens=180, completion_tokens=120, total_tokens=300),
        )
        self.openai.chat.completions.create = AsyncMock(return_value=mock_completion)
        service = AnalysisService(self.settings, self.tavily, self.openai)

        req = JobMatchAndRewriteRequest(
            target_role="AI Engineer",
            current_role="Junior Python Dev",
            user_skills=["Python", "FastAPI", "SQL"],
            existing_bullets=["Built REST APIs with FastAPI."],
        )
        result = await service.match_jobs_and_rewrite(req)

        self.assertIsInstance(result, JobMatchAndRewriteResponse)
        self.assertEqual(len(result.rewritten_bullets), 1)
        self.assertIn("RAG pipelines", result.rewritten_bullets[0].rewritten_bullet)
        self.assertGreaterEqual(len(result.live_openings), 1)

    def test_get_demo_presets_returns_three_profiles(self) -> None:
        service = AnalysisService(self.settings, self.tavily, self.openai)
        presets = service.get_presets()
        self.assertEqual(len(presets), 3)
        self.assertEqual(presets[0]["id"], "preset_ai_agent")
        self.assertEqual(presets[1]["id"], "preset_ml_infra")
        self.assertEqual(presets[2]["id"], "preset_cloud_architect")


if __name__ == "__main__":
    unittest.main()
