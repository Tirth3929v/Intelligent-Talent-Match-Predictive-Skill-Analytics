import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from skillsync_api.models import AnalysisRequest, AnalysisResponse
from skillsync_api.service import (
    AnalysisService,
    ConfigurationError,
    InferenceError,
    MarketSearchError,
    ServiceSettings,
    _parse_json_object,
)


class AnalysisServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = ServiceSettings(
            tavily_api_key="tavily-test-key",
            nebius_api_key="nebius-test-key",
            nebius_model="meta-llama/test-model",
        )
        self.request = AnalysisRequest(
            current_role="Data Analyst",
            current_level="Junior",
            user_skills=["Python", "SQL"],
            target_role="Machine Learning Engineer",
        )
        self.tavily = Mock()
        self.tavily.search.return_value = {
            "answer": "Hiring emphasizes Python and model deployment.",
            "results": [
                {
                    "title": "Role requirements",
                    "url": "https://example.com/roles",
                    "content": "Python, MLOps, and deployment are common.",
                }
            ],
        }
        model_payload = {
            "match_score_percentage": 45,
            "market_benchmarks_summary": "Python and model deployment are common.",
            "skill_breakdown": {
                "relevant_matched": ["Python"],
                "transferable_or_secondary": ["SQL"],
                "missing_critical": ["MLOps"],
            },
            "step_by_step_learning_roadmap": [
                {
                    "phase": "Foundations",
                    "skills": ["MLOps"],
                    "recommended_project": "Deploy a model with a CI pipeline.",
                }
            ],
            "interview_prep_questions": ["How would you monitor model drift?"],
            "search_sources": ["https://untrusted.example/"],
        }
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(model_payload))
                )
            ]
        )
        self.openai = Mock()
        self.openai.chat.completions.create = AsyncMock(return_value=completion)
        self.service = AnalysisService(self.settings, self.tavily, self.openai)

    async def test_analyze_uses_live_sources_and_validates_model_output(self) -> None:
        result = await self.service.analyze(self.request)

        self.assertIsInstance(result, AnalysisResponse)
        self.assertEqual(result.match_score_percentage, 57)
        self.assertEqual(result.search_sources, ["https://example.com/roles"])
        self.tavily.search.assert_called_once()
        self.openai.chat.completions.create.assert_awaited_once()
        call = self.openai.chat.completions.create.await_args
        self.assertEqual(call.kwargs["model"], "meta-llama/test-model")
        self.assertEqual(call.kwargs["response_format"], {"type": "json_object"})

    async def test_missing_credentials_are_reported(self) -> None:
        service = AnalysisService(
            ServiceSettings(None, "nebius-test-key", "test-model"),
            self.tavily,
            self.openai,
        )

        with self.assertRaisesRegex(ConfigurationError, "TAVILY_API_KEY"):
            await service.analyze(self.request)

        self.tavily.search.assert_not_called()

    async def test_market_search_failure_is_not_silently_ignored(self) -> None:
        self.tavily.search.side_effect = RuntimeError("provider unavailable")

        with self.assertRaisesRegex(
            MarketSearchError, "market research is currently unavailable"
        ):
            await self.service.analyze(self.request)

        self.openai.chat.completions.create.assert_not_awaited()

    async def test_invalid_model_json_is_reported(self) -> None:
        self.openai.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"bad":'))]
            )
        )

        with self.assertRaises(InferenceError):
            await self.service.analyze(self.request)

    async def test_client_uses_configured_openai_base_url(self) -> None:
        settings = ServiceSettings(
            "tavily-test-key",
            "test-gemini-key",
            "gemini-1.5-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
        fake_client = Mock()
        fake_client.chat.completions.create = AsyncMock(return_value=completion)

        with patch(
            "skillsync_api.service.AsyncOpenAI", return_value=fake_client
        ) as openai_constructor:
            await AnalysisService(settings)._request_analysis(
                AnalysisRequest(
                    current_role="Analyst",
                    current_level="Junior",
                    user_skills=["Python"],
                    target_role="Engineer",
                ),
                "Market data",
                [],
            )

        self.assertEqual(
            openai_constructor.call_args.kwargs["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )


class ModelAndParserTests(unittest.TestCase):
    def test_request_rejects_empty_skills(self) -> None:
        with self.assertRaises(ValidationError):
            AnalysisRequest(
                current_role="Analyst",
                current_level="Junior",
                user_skills=[],
                target_role="Engineer",
            )

    def test_request_rejects_blank_skill_entries(self) -> None:
        with self.assertRaises(ValidationError):
            AnalysisRequest(
                current_role="Analyst",
                current_level="Junior",
                user_skills=["  "],
                target_role="Engineer",
            )

    def test_parser_accepts_json_fenced_or_surrounded_by_text(self) -> None:
        self.assertEqual(_parse_json_object('```json\n{"score": 1}\n```'), {"score": 1})
        self.assertEqual(_parse_json_object('Result: {"score": 2} done'), {"score": 2})

    def test_environment_selects_gemini_openai_compatible_endpoint(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "NEBIUS_API_KEY": "test-gemini-key",
                "NEBIUS_MODEL": "gemini-1.5-flash",
            },
            clear=True,
        ):
            settings = ServiceSettings.from_environment()

        self.assertEqual(
            settings.openai_base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.assertEqual(settings.nebius_model, "gemini-1.5-flash")


if __name__ == "__main__":
    unittest.main()
