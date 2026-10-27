"""Diagnostic utility to verify multi-tiered NVIDIA Nemotron inference on Nebius Token Factory.

Reads strictly from .env without echoing or exposing credentials.
Tests all three tiers:
- Fast (Nemotron 3 Nano 30B)
- Balanced (Nemotron 3 Super 120B)
- Reasoning (Nemotron 3 Ultra 550B)
"""

import asyncio
import os
import sys
import time
from dotenv import load_dotenv

from skillsync_api.nebius_router import NemotronTokenFactoryRouter


async def test_live_nebius_router_inference() -> int:
    load_dotenv()

    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    base_url = (
        os.getenv("NEBIUS_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "https://api.studio.nebius.ai/v1/")
    ).strip()

    fast_model = os.getenv("NEMOTRON_FAST_MODEL", "nvidia/Nemotron-3-Nano-30B").strip()
    super_model = os.getenv("NEMOTRON_SUPER_MODEL", "nvidia/Nemotron-3-Super-120B").strip()
    ultra_model = os.getenv("NEMOTRON_ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550B").strip()

    print("=" * 70)
    print("NEBIUS × NVIDIA NEMOTRON MULTI-TIER ROUTER VERIFICATION PROBE")
    print("=" * 70)
    print(f"Target Base URL : {base_url}")
    print(f"Fast Tier       : {fast_model}")
    print(f"Balanced Tier   : {super_model}")
    print(f"Reasoning Tier  : {ultra_model}")
    print(f"API Key Status  : {'[CONFIGURED]' if api_key else '[MISSING]'}")

    if not api_key:
        print("\n[ERROR] NEBIUS_API_KEY is not set in your .env file.")
        print("Please add your Nebius Token Factory key to .env without pasting in chat.")
        return 1

    router = NemotronTokenFactoryRouter(
        api_key=api_key,
        base_url=base_url,
        models={
            "fast": fast_model,
            "balanced": super_model,
            "reasoning": ultra_model,
        },
    )

    tiers_to_test = [
        ("fast", "Normalize candidate skills (low-latency parsing)"),
        ("balanced", "Summarize 2026 hiring demand signals"),
        ("reasoning", "Synthesize 90-day technical learning roadmap"),
    ]

    total_success = 0
    executions = []

    print("\nProbing Nebius Token Factory model tiers...\n")

    for tier, description in tiers_to_test:
        model_name = router.get_model_for_tier(tier)
        print(f"--> [{tier.upper()}] Probing {model_name}...")
        print(f"    Task: {description}")
        start = time.perf_counter()
        try:
            result = await router.ainvoke(
                task_type=tier,
                system_prompt="You are an AI diagnostic probe for SkillSync AI. Reply in JSON.",
                user_prompt=f'{{"probe_tier": "{tier}", "status": "ping"}}',
                temperature=0.0,
                max_tokens=60,
                response_format={"type": "json_object"},
            )
            elapsed_ms = result["latency_ms"]
            tokens = result["tokens_used"]
            tok_per_sec = result["tokens_per_sec"]
            savings = result["estimated_savings_pct"]
            served = result["model_used"]

            print(f"    [OK] Served Model: {served}")
            print(f"    Latency: {elapsed_ms}ms | Tokens: {tokens} | Throughput: {tok_per_sec} tok/s")
            print(f"    Credit Efficiency: {savings}% savings vs Ultra baseline")
            print(f"    Payload: {result['content'][:80]}...\n")
            total_success += 1
            executions.append(result)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            print(f"    [FAIL] Tier failed after {elapsed_ms}ms:")
            print(f"    Error: {type(exc).__name__} - {exc}\n")

    print("-" * 70)
    cumulative_savings = router.calculate_cumulative_savings(executions)
    print(f"Router Health: {total_success}/{len(tiers_to_test)} Tiers Verified")
    print(f"Cascaded Credit Savings: {cumulative_savings}% across multi-tier workflow")
    print("=" * 70)

    return 0 if total_success > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(test_live_nebius_router_inference()))
