"""Revenue and risk oriented action masking for driver RL environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionMaskSpec:
    action_dim: int
    cargo_start: int
    top_k: int
    wait_actions: tuple[int, ...]
    force_rest_action: int


def apply_revenue_risk_mask(
    base_mask: np.ndarray,
    *,
    cargo_list: list[dict[str, Any]],
    spec: ActionMaskSpec,
    driver_config: dict[str, Any],
    total_deadhead_km: float,
    rest_today_min: float,
    consecutive_wait_min: float,
) -> tuple[np.ndarray, dict[int, str]]:
    """Apply hard revenue/risk filters on top of the legal action mask."""
    mask = np.asarray(base_mask, dtype=np.bool_).copy()
    reasons: dict[int, str] = {}

    max_deadhead = float(driver_config.get("max_deadhead_km", 0.0) or 0.0)
    required_rest = float(driver_config.get("required_rest_min", 240.0) or 240.0)
    rest_gap = max(0.0, required_rest - rest_today_min)

    for idx, item in enumerate(cargo_list[:spec.top_k]):
        action = spec.cargo_start + idx
        if action >= spec.action_dim or not mask[action]:
            continue
        reason = _cargo_block_reason(item, max_deadhead, total_deadhead_km, rest_gap)
        if reason:
            mask[action] = False
            reasons[action] = reason

    positive_cargo_available = _has_positive_available_cargo(mask, cargo_list, spec)

    if positive_cargo_available and spec.force_rest_action < spec.action_dim and mask[spec.force_rest_action]:
        mask[spec.force_rest_action] = False
        reasons[spec.force_rest_action] = "positive_cargo_available"

    wait_loop_threshold = 60 if positive_cargo_available else 180
    if consecutive_wait_min >= wait_loop_threshold:
        for action in spec.wait_actions:
            if action < spec.action_dim and mask[action]:
                mask[action] = False
                reasons[action] = "wait_loop"

    if not positive_cargo_available and rest_gap > 120 and spec.force_rest_action < spec.action_dim:
        mask[spec.force_rest_action] = True
        reasons.pop(spec.force_rest_action, None)

    if not mask.any():
        fallback = spec.force_rest_action if rest_gap > 0 else spec.wait_actions[0]
        if fallback < spec.action_dim:
            mask[fallback] = True
            reasons.pop(fallback, None)

    return mask, reasons


def _has_positive_available_cargo(
    mask: np.ndarray,
    cargo_list: list[dict[str, Any]],
    spec: ActionMaskSpec,
) -> bool:
    for idx, item in enumerate(cargo_list[:spec.top_k]):
        action = spec.cargo_start + idx
        if action >= spec.action_dim or not mask[action]:
            continue
        if _estimated_true_net(item) > 0 and _estimated_net_profit(item) > 0:
            return True
    return False


def _cargo_block_reason(
    item: dict[str, Any],
    max_deadhead_km: float,
    total_deadhead_km: float,
    rest_gap_min: float,
) -> str | None:
    true_net = _estimated_true_net(item)
    deadhead_km = float(item.get("distance_km", item.get("deadhead_km", 0.0)) or 0.0)
    total_minutes = _estimated_minutes(item)
    penalty = float(item.get("hard_penalty", item.get("penalty_score", 0.0)) or 0.0)

    if true_net < 0:
        return "negative_true_net"
    if total_minutes > 600 and true_net < 80:
        return "long_low_margin"
    if deadhead_km > 150 and true_net < 200:
        return "extreme_deadhead_low_margin"
    if penalty > 0 and true_net < penalty + 100:
        return "penalty_not_covered"
    if max_deadhead_km > 0 and total_deadhead_km + deadhead_km > max_deadhead_km and true_net < 300:
        return "deadhead_budget_risk"
    if rest_gap_min > 180 and total_minutes > 240:
        return "rest_deadline_risk"
    return None


def _estimated_true_net(item: dict[str, Any]) -> float:
    for key in ("true_net", "score", "net_profit", "profit_search_score"):
        value = item.get(key)
        if value is not None:
            return float(value or 0.0)
    return _estimated_net_profit(item)


def _estimated_net_profit(item: dict[str, Any]) -> float:
    value = item.get("net_profit")
    if value is not None:
        return float(value or 0.0)
    cargo = item.get("cargo", {})
    price = float(cargo.get("price", item.get("price", 0.0)) or 0.0)
    deadhead_km = float(item.get("distance_km", item.get("deadhead_km", 0.0)) or 0.0)
    haul_km = float(item.get("haul_km", 0.0) or 0.0)
    return price - (deadhead_km + haul_km) * 1.5


def _estimated_minutes(item: dict[str, Any]) -> int:
    cargo = item.get("cargo", {})
    return int(
        item.get(
            "total_minutes",
            cargo.get("cost_time_minutes", item.get("cost_time_minutes", 0)),
        )
        or 0
    )
