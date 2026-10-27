"""Multi-tiered NVIDIA Nemotron router on Nebius Token Factory.

Routes fast parsing/extraction to Nemotron 3 Nano/Super and deep career/skill
gap reasoning to Nemotron 3 Ultra (550B), with automatic fallback cascading.
"""

import logging
import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TierInvocationResult(BaseModel):
    """Execution telemetry for a single model tier invocation."""

    content: str
    model_used: str
    tier: str
    latency_ms: float
    tokens_used: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    estimated_savings_pct: int = 0
    request_id: str | None = None
    fallback_engaged: bool = False
    original_tier: str | None = None


# Relative compute cost baseline vs Nemotron Ultra (550B = 100% cost)
TIER_SAVINGS_MAP: dict[str, int] = {
    "fast": 85,       # Nano (30B) saves ~85% compute vs Ultra
    "balanced": 60,   # Super (120B) saves ~60% compute vs Ultra
    "reasoning": 0,   # Ultra (550B) baseline
}

# Cascade order when a higher tier is throttled or times out
TIER_FALLBACK_ORDER: dict[str, list[str]] = {
    "reasoning": ["balanced", "fast"],
    "balanced": ["fast"],
    "fast": [],
}


class NemotronTokenFactoryRouter:
    """Multi-tiered NVIDIA Nemotron router on Nebius Token Factory.

    Intelligently cascades tasks across NVIDIA Nemotron 3 model tiers:
    - fast (Nemotron-3-Nano-30B): Resume/profile skill normalization, keyword
      canonicalization, and Tavily search query generation.
    - balanced (Nemotron-3-Super-120B): Live market requirements summarization,
      hiring trend distillation, and AI Role Discovery.
    - reasoning (Nemotron-3-Ultra-550B): Deep multi-step 90-day learning roadmaps,
      project architecture recommendations, and complex skill gap synthesis.
    """

    DEFAULT_BASE_URL = "https://api.studio.nebius.ai/v1/"
    DEFAULT_FAST_MODEL = "nvidia/Nemotron-3-Nano-30B"
    DEFAULT_SUPER_MODEL = "nvidia/Nemotron-3-Super-120B"
    DEFAULT_ULTRA_MODEL = "nvidia/Nemotron-3-Ultra-550B"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        models: dict[str, str] | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        fallback_enabled: bool = True,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("NEBIUS_API_KEY", "").strip()
            or None
        )
        self.base_url = (
            base_url
            or os.getenv("NEBIUS_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).strip()
        if not self.base_url.endswith("/"):
            self.base_url += "/"

        # Generic override fallback if user specifies NEBIUS_MODEL
        env_fallback_model = os.getenv("NEBIUS_MODEL", "").strip() or None

        # Tiered NVIDIA Open Source Models on Nebius Token Factory
        self.models: dict[str, str] = models or {
            "fast": os.getenv(
                "NEMOTRON_FAST_MODEL",
                env_fallback_model or self.DEFAULT_FAST_MODEL,
            ).strip(),
            "balanced": os.getenv(
                "NEMOTRON_SUPER_MODEL",
                env_fallback_model or self.DEFAULT_SUPER_MODEL,
            ).strip(),
            "reasoning": os.getenv(
                "NEMOTRON_ULTRA_MODEL",
                env_fallback_model or self.DEFAULT_ULTRA_MODEL,
            ).strip(),
        }

        self._client = client
        self._async_client = async_client
        self.fallback_enabled = fallback_enabled

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "",
                timeout=45.0,
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "",
                timeout=60.0,
                max_retries=2,
            )
        return self._async_client

    def get_model_for_tier(self, task_type: str) -> str:
        return self.models.get(task_type, self.models.get("balanced", self.DEFAULT_SUPER_MODEL))

    def invoke(
        self,
        task_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Synchronous invocation with automatic fallback cascading."""
        tiers_to_try = [task_type]
        if self.fallback_enabled:
            tiers_to_try.extend(TIER_FALLBACK_ORDER.get(task_type, []))

        last_exc: Exception | None = None
        for current_tier in tiers_to_try:
            model_id = self.get_model_for_tier(current_tier)
            start_time = time.perf_counter()
            kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            try:
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                tokens_per_sec = (
                    round(completion_tokens / (latency_ms / 1000.0), 2)
                    if latency_ms > 0 and completion_tokens > 0
                    else 0.0
                )
                savings_pct = TIER_SAVINGS_MAP.get(current_tier, 0)
                served_model = getattr(response, "model", None) or model_id

                return {
                    "content": content,
                    "model_used": served_model,
                    "tier": current_tier,
                    "latency_ms": latency_ms,
                    "tokens_used": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "estimated_savings_pct": savings_pct,
                    "request_id": getattr(response, "id", None),
                    "fallback_engaged": current_tier != task_type,
                    "original_tier": task_type if current_tier != task_type else None,
                }
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Model tier '%s' (%s) failed: %s. Trying fallback tier if available.",
                    current_tier,
                    model_id,
                    exc,
                )
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("No model tiers were available to execute the request.")

    async def ainvoke(
        self,
        task_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Asynchronous invocation with automatic fallback cascading."""
        tiers_to_try = [task_type]
        if self.fallback_enabled:
            tiers_to_try.extend(TIER_FALLBACK_ORDER.get(task_type, []))

        last_exc: Exception | None = None
        for current_tier in tiers_to_try:
            model_id = self.get_model_for_tier(current_tier)
            start_time = time.perf_counter()
            kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            try:
                response = await self.async_client.chat.completions.create(**kwargs)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                tokens_per_sec = (
                    round(completion_tokens / (latency_ms / 1000.0), 2)
                    if latency_ms > 0 and completion_tokens > 0
                    else 0.0
                )
                savings_pct = TIER_SAVINGS_MAP.get(current_tier, 0)
                served_model = getattr(response, "model", None) or model_id

                return {
                    "content": content,
                    "model_used": served_model,
                    "tier": current_tier,
                    "latency_ms": latency_ms,
                    "tokens_used": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "estimated_savings_pct": savings_pct,
                    "request_id": getattr(response, "id", None),
                    "fallback_engaged": current_tier != task_type,
                    "original_tier": task_type if current_tier != task_type else None,
                }
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Async model tier '%s' (%s) failed: %s. Trying fallback tier if available.",
                    current_tier,
                    model_id,
                    exc,
                )
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("No model tiers were available to execute the request.")

    @staticmethod
    def calculate_cumulative_savings(tier_executions: list[dict[str, Any]]) -> int:
        """Calculate overall percentage of credits saved by routing calls away from Ultra."""
        if not tier_executions:
            return 0
        total_tokens = sum(e.get("tokens_used", 0) for e in tier_executions)
        if total_tokens == 0:
            return 0
        weighted_savings = sum(
            e.get("tokens_used", 0) * e.get("estimated_savings_pct", 0)
            for e in tier_executions
        )
        return int(round(weighted_savings / total_tokens))
