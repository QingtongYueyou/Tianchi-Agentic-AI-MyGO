"""A/B evaluation: compare strategy variants on identical seeds.

Usage:
    python -m agent.ab_eval --config demo/agent/configs/rl_config.yaml --seeds 42,123,456
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger("agent.ab_eval")

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGY_GROUPS = [
    "heuristic",
    "maskable_integrated",
    "maskable_per_driver",
]


def _run_episode(
    env: Any,
    strategy: str,
    model_dir: str | None = None,
    driver_id: str | None = None,
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Run a single episode with the given strategy. Returns metrics dict."""
    metrics: dict[str, Any] = {
        "total_income": 0.0,
        "net_income": 0.0,
        "penalties": 0.0,
        "deadhead_km": 0.0,
        "accepted_orders": 0,
        "wait_count": 0,
        "wait_duration_min": 0.0,
        "force_rest_count": 0,
        "positive_candidate_wait_count": 0,
        "fallback_used": 0,
        "steps": 0,
    }

    # Load RL layer if strategy needs it
    rl_layer = None
    if strategy in ("maskable_integrated", "maskable_per_driver"):
        from agent.maskable_rl_integration import MaskableRLDecisionLayer
        if model_dir:
            if strategy == "maskable_per_driver" and driver_id:
                # Per-driver model: models/{driver_id}/maskable/policy_maskable_final.zip
                pd_path = Path(model_dir) / driver_id / "maskable" / "policy_maskable_final.zip"
                if pd_path.exists():
                    rl_layer = MaskableRLDecisionLayer(model_path=str(pd_path))
            if rl_layer is None or not rl_layer.is_loaded:
                rl_layer = MaskableRLDecisionLayer(model_dir=Path(model_dir))

    obs, info = env.reset()
    step = -1
    for step in range(max_steps):
        action, fallback_used = _select_action(obs, info, strategy, rl_layer, env)
        obs, reward, terminated, truncated, info = env.step(action)
        _update_metrics(metrics, info, action, fallback_used=fallback_used)
        if terminated or truncated:
            break

    metrics["steps"] = step + 1
    metrics["strategy"] = strategy
    metrics["driver_id"] = driver_id or ""
    return metrics


def _select_action(
    obs: np.ndarray,
    info: dict,
    strategy: str,
    rl_layer: Any,
    env: Any,
) -> tuple[int, bool]:
    """Select action based on strategy."""
    if strategy == "heuristic":
        mask = info.get("action_mask")
        if mask is None and hasattr(env, "action_masks"):
            mask = env.action_masks()
        if mask is not None:
            from agent.heuristic_teacher import choose_teacher_action
            return int(choose_teacher_action(env, mask)), False
        return 0, False

    if rl_layer is not None and rl_layer.is_loaded:
        try:
            candidates = getattr(env, "_current_cargo_list", [])
            status = getattr(env, "_current_status", {}) or {}
            state_tracker = env
            constraints = getattr(env, "constraints", [])
            driver_config = getattr(env, "driver_config", None)

            result = rl_layer.decide(candidates, status, state_tracker, constraints, driver_config)
            if result is not None:
                return int(result.get("action_idx", 0)), False
        except Exception as exc:
            _logger.debug("RL decide failed: %s", exc)

    mask = info.get("action_mask")
    if mask is None and hasattr(env, "action_masks"):
        mask = env.action_masks()
    if mask is not None:
        from agent.heuristic_teacher import choose_teacher_action
        return int(choose_teacher_action(env, mask)), True
    return 0, True


def _update_metrics(metrics: dict, info: dict, action: int, fallback_used: bool = False) -> None:
    """Update metrics from step info."""
    from agent.rl_env import ACTION_FORCE_REST

    status = info.get("status", {})
    metrics["total_income"] = float(
        info.get("total_income", status.get("total_income", metrics["total_income"]))
    )
    metrics["net_income"] = float(
        info.get("net_income", status.get("net_income", metrics["net_income"]))
    )
    metrics["penalties"] = float(
        info.get("total_penalty", status.get("total_penalties", metrics["penalties"]))
    )
    metrics["deadhead_km"] = float(
        info.get("total_deadhead", status.get("total_deadhead_km", metrics["deadhead_km"]))
    )
    metrics["accepted_orders"] = int(
        info.get("total_orders", status.get("total_orders", metrics["accepted_orders"]))
    )

    if action in (0, 1, 2):  # wait actions
        metrics["wait_count"] += 1
        durations = {0: 15, 1: 30, 2: 60}
        metrics["wait_duration_min"] += durations.get(action, 30)
    if action == ACTION_FORCE_REST:
        metrics["force_rest_count"] += 1
        metrics["wait_count"] += 1
        metrics["wait_duration_min"] += 240

    reward_breakdown = info.get("reward_breakdown", {})
    if reward_breakdown.get("opportunity_wait_penalty", 0.0) < 0:
        metrics["positive_candidate_wait_count"] += 1

    if fallback_used or info.get("fallback_used", False):
        metrics["fallback_used"] += 1


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_ab_eval(
    config: dict[str, Any],
    seeds: list[int],
    strategies: list[str] | None = None,
    max_steps: int = 2000,
) -> dict[str, Any]:
    """Run A/B evaluation across seeds and strategies."""
    from agent.train import build_envs, load_config, resolve_project_path, get_config_value

    model_dir = resolve_project_path(get_config_value(
        config, "model_dir", "paths.models_dir", default="demo/agent/models"
    ))
    driver_ids = ["D001", "D002", "D003", "D005", "D006", "D007", "D008", "D010"]
    strategies = strategies or STRATEGY_GROUPS

    results: dict[str, Any] = {
        "seeds": seeds,
        "strategies": strategies,
        "driver_ids": driver_ids,
        "per_seed": {},
        "summary": {},
    }

    for seed in seeds:
        _logger.info("Evaluating seed=%d", seed)
        seed_results: dict[str, dict] = {}

        for strategy in strategies:
            _logger.info("  Strategy: %s", strategy)
            strategy_metrics = []

            for driver_id in driver_ids:
                try:
                    envs = build_envs(config, driver_ids=[driver_id])
                    if not envs:
                        continue
                    env = envs[0]
                    # Set seed if env supports it
                    if hasattr(env, "reset"):
                        env.reset(seed=seed)

                    metrics = _run_episode(
                        env, strategy,
                        model_dir=str(model_dir),
                        driver_id=driver_id,
                        max_steps=max_steps,
                    )
                    metrics["seed"] = seed
                    strategy_metrics.append(metrics)

                    if hasattr(env, "close"):
                        env.close()
                except Exception as exc:
                    _logger.warning("  Driver %s failed: %s", driver_id, exc)

            seed_results[strategy] = strategy_metrics

        results["per_seed"][str(seed)] = seed_results

    # Compute summary
    results["summary"] = _compute_summary(results, strategies)

    # Check acceptance criteria
    results["acceptance"] = _check_acceptance(results)

    return results


def _compute_summary(results: dict, strategies: list[str]) -> dict:
    """Compute per-strategy aggregate metrics."""
    summary = {}
    for strategy in strategies:
        all_metrics = []
        for seed_data in results["per_seed"].values():
            all_metrics.extend(seed_data.get(strategy, []))

        if not all_metrics:
            continue

        incomes = [m["net_income"] for m in all_metrics]
        penalties = [m["penalties"] for m in all_metrics]
        fallback_rates = [
            float(m["fallback_used"]) / max(1, int(m.get("steps", 0) or 0))
            for m in all_metrics
        ]
        wait_rates = [
            float(m.get("wait_count", 0)) / max(1, int(m.get("steps", 0) or 0))
            for m in all_metrics
        ]
        force_rest_rates = [
            float(m.get("force_rest_count", 0)) / max(1, int(m.get("steps", 0) or 0))
            for m in all_metrics
        ]
        positive_wait_rates = [
            float(m.get("positive_candidate_wait_count", 0)) / max(1, int(m.get("steps", 0) or 0))
            for m in all_metrics
        ]
        orders = [m["accepted_orders"] for m in all_metrics]

        summary[strategy] = {
            "episodes": len(all_metrics),
            "avg_net_income": float(np.mean(incomes)),
            "std_net_income": float(np.std(incomes)),
            "min_net_income": float(np.min(incomes)),
            "max_net_income": float(np.max(incomes)),
            "avg_penalties": float(np.mean(penalties)),
            "avg_orders": float(np.mean(orders)),
            "min_orders": int(np.min(orders)),
            "avg_fallback_rate": float(np.mean(fallback_rates)) if fallback_rates else 0.0,
            "avg_wait_rate": float(np.mean(wait_rates)) if wait_rates else 0.0,
            "avg_force_rest_rate": float(np.mean(force_rest_rates)) if force_rest_rates else 0.0,
            "avg_positive_candidate_wait_rate": float(np.mean(positive_wait_rates)) if positive_wait_rates else 0.0,
        }
    return summary


def _check_acceptance(results: dict) -> dict:
    """Check acceptance criteria against baseline (heuristic)."""
    summary = results.get("summary", {})
    baseline = summary.get("heuristic", {})

    if not baseline:
        return {"status": "no_baseline", "criteria": {}}

    baseline_avg = baseline.get("avg_net_income", 0)
    baseline_min = baseline.get("min_net_income", 0)
    baseline_penalties = baseline.get("avg_penalties", 0)
    baseline_orders = baseline.get("avg_orders", 0)
    baseline_wait_rate = baseline.get("avg_wait_rate", 1.0)

    acceptance = {}
    for strategy, stats in summary.items():
        if strategy == "heuristic":
            continue

        avg_income = stats.get("avg_net_income", 0)
        min_income = stats.get("min_net_income", 0)
        avg_penalties = stats.get("avg_penalties", 0)
        fallback_rate = stats.get("avg_fallback_rate", 0)
        avg_orders = stats.get("avg_orders", 0)
        min_orders = stats.get("min_orders", 0)
        wait_rate = stats.get("avg_wait_rate", 0)
        force_rest_rate = stats.get("avg_force_rest_rate", 0)
        positive_wait_rate = stats.get("avg_positive_candidate_wait_rate", 0)

        criteria = {
            "avg_income_target": baseline_avg * 1.08,
            "avg_income_actual": avg_income,
            "avg_income_pass": avg_income >= baseline_avg * 1.08 if baseline_avg > 0 else None,
            "worst_seed_target": baseline_min * 0.98,
            "worst_seed_actual": min_income,
            "worst_seed_pass": min_income >= baseline_min * 0.98 if baseline_min > 0 else None,
            "penalty_target": baseline_penalties * 1.10,
            "penalty_actual": avg_penalties,
            "penalty_pass": avg_penalties <= baseline_penalties * 1.10,
            "fallback_target": 0.35,
            "fallback_actual": fallback_rate,
            "fallback_pass": fallback_rate < 0.35,
            "avg_orders_target": baseline_orders * 0.90,
            "avg_orders_actual": avg_orders,
            "avg_orders_pass": avg_orders >= baseline_orders * 0.90 if baseline_orders > 0 else None,
            "min_orders_target": 1,
            "min_orders_actual": min_orders,
            "min_orders_pass": min_orders >= 1,
            "wait_rate_target": min(0.90, baseline_wait_rate + 0.15),
            "wait_rate_actual": wait_rate,
            "wait_rate_pass": wait_rate <= min(0.90, baseline_wait_rate + 0.15),
            "force_rest_rate_target": 0.20,
            "force_rest_rate_actual": force_rest_rate,
            "force_rest_rate_pass": force_rest_rate <= 0.20,
            "positive_wait_rate_target": 0.05,
            "positive_wait_rate_actual": positive_wait_rate,
            "positive_wait_rate_pass": positive_wait_rate <= 0.05,
        }
        pass_values = [v for k, v in criteria.items() if k.endswith("_pass") and v is not None]
        criteria["overall_pass"] = bool(pass_values) and all(pass_values)
        acceptance[strategy] = criteria

    return acceptance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="A/B evaluation for RL strategies")
    parser.add_argument("--config", type=str, default=None, help="Path to rl_config.yaml")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--strategies", type=str, default=None, help="Comma-separated strategy names")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    from agent.train import load_config
    config = load_config(args.config)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")] if args.strategies else None

    results = run_ab_eval(config, seeds, strategies, args.max_steps)

    # Write output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("demo/results/rl_logs") / f"ab_eval_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    _logger.info("Results written to %s", output_path)

    # Print summary table
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a concise summary table."""
    summary = results.get("summary", {})
    acceptance = results.get("acceptance", {})

    print("\n" + "=" * 80)
    print("A/B EVALUATION SUMMARY")
    print("=" * 80)

    header = f"{'Strategy':<25} {'Avg Income':>12} {'Min Income':>12} {'Orders':>8} {'Wait':>8} {'Force':>8} {'Pass':>6}"
    print(header)
    print("-" * 80)

    for strategy, stats in summary.items():
        avg_inc = stats.get("avg_net_income", 0)
        min_inc = stats.get("min_net_income", 0)
        orders = stats.get("avg_orders", 0)
        wait_rate = stats.get("avg_wait_rate", 0)
        force_rest = stats.get("avg_force_rest_rate", 0)

        acc = acceptance.get(strategy, {})
        overall = acc.get("overall_pass")
        pass_str = "PASS" if overall else ("FAIL" if overall is not None else "-")

        print(f"{strategy:<25} {avg_inc:>12,.0f} {min_inc:>12,.0f} {orders:>8.1f} {wait_rate:>8.1%} {force_rest:>8.1%} {pass_str:>6}")

    print("=" * 80)

    # Print acceptance details for non-baseline strategies
    for strategy, criteria in acceptance.items():
        if strategy == "heuristic":
            continue
        print(f"\n{strategy} acceptance details:")
        for key in ("avg_income", "worst_seed", "penalty", "fallback",
                    "avg_orders", "min_orders", "wait_rate",
                    "force_rest_rate", "positive_wait_rate"):
            actual = criteria.get(f"{key}_actual", "?")
            target = criteria.get(f"{key}_target", "?")
            passed = criteria.get(f"{key}_pass")
            status = "PASS" if passed else ("FAIL" if passed is not None else "-")
            print(f"  {key}: {actual} (target: {target}) [{status}]")


if __name__ == "__main__":
    main()
