"""Tests for multi-tiered NVIDIA Nemotron routing on Nebius Token Factory."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from skillsync_api.nebius_router import NemotronTokenFactoryRouter


class NemotronTokenFactoryRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.router = NemotronTokenFactoryRouter(
            api_key="test-nebius-key",
            base_url="https://api.studio.nebius.ai/v1/",
            models={
                "fast": "nvidia/Nemotron-3-Nano-30B",
                "balanced": "nvidia/Nemotron-3-Super-120B",
                "reasoning": "nvidia/Nemotron-3-Ultra-550B",
            },
        )

    def test_tier_model_resolution(self) -> None:
        self.assertEqual(
            self.router.get_model_for_tier("fast"), "nvidia/Nemotron-3-Nano-30B"
        )
        self.assertEqual(
            self.router.get_model_for_tier("balanced"), "nvidia/Nemotron-3-Super-120B"
        )
        self.assertEqual(
            self.router.get_model_for_tier("reasoning"), "nvidia/Nemotron-3-Ultra-550B"
        )
        self.assertEqual(
            self.router.get_model_for_tier("unknown_tier"),
            "nvidia/Nemotron-3-Super-120B",
        )

    async def test_async_invocation_returns_telemetry_and_savings(self) -> None:
        fake_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"status": "verified"}))
                )
            ],
            model="nvidia/Nemotron-3-Nano-30B",
            id="chatcmpl-nano-123",
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=25, total_tokens=65),
        )
        mock_async_client = Mock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=fake_completion
        )
        self.router._async_client = mock_async_client

        res = await self.router.ainvoke(
            task_type="fast",
            system_prompt="Test sys",
            user_prompt="Test user",
        )

        self.assertEqual(res["model_used"], "nvidia/Nemotron-3-Nano-30B")
        self.assertEqual(res["tier"], "fast")
        self.assertEqual(res["tokens_used"], 65)
        self.assertEqual(res["prompt_tokens"], 40)
        self.assertEqual(res["completion_tokens"], 25)
        self.assertEqual(res["estimated_savings_pct"], 85)
        self.assertGreaterEqual(res["latency_ms"], 0.0)
        self.assertEqual(res["request_id"], "chatcmpl-nano-123")

    def test_sync_invocation_returns_telemetry(self) -> None:
        fake_completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Role summary output")
                )
            ],
            model="nvidia/Nemotron-3-Super-120B",
            id="chatcmpl-super-456",
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=50, total_tokens=130),
        )
        mock_sync_client = Mock()
        mock_sync_client.chat.completions.create = Mock(
            return_value=fake_completion
        )
        self.router._client = mock_sync_client

        res = self.router.invoke(
            task_type="balanced",
            system_prompt="System prompt",
            user_prompt="User prompt",
        )

        self.assertEqual(res["model_used"], "nvidia/Nemotron-3-Super-120B")
        self.assertEqual(res["tier"], "balanced")
        self.assertEqual(res["tokens_used"], 130)
        self.assertEqual(res["estimated_savings_pct"], 60)
        self.assertEqual(res["content"], "Role summary output")

    def test_calculate_cumulative_savings(self) -> None:
        executions = [
            {"tier": "fast", "tokens_used": 100, "estimated_savings_pct": 85},
            {"tier": "balanced", "tokens_used": 200, "estimated_savings_pct": 60},
            {"tier": "reasoning", "tokens_used": 300, "estimated_savings_pct": 0},
        ]
        # (100*85 + 200*60 + 300*0) / 600 = (8500 + 12000) / 600 = 20500 / 600 = 34.16%
        savings = NemotronTokenFactoryRouter.calculate_cumulative_savings(executions)
        self.assertEqual(savings, 34)

    def test_calculate_cumulative_savings_empty(self) -> None:
        self.assertEqual(NemotronTokenFactoryRouter.calculate_cumulative_savings([]), 0)


if __name__ == "__main__":
    unittest.main()
