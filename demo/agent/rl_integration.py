"""RL model integration layer for ranking heuristic candidates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from agent.rl_models import (
    OrderScoringNetworkNumpy,
    PolicyNetworkNumpy,
    PositionValueNetworkNumpy,
)

_logger = logging.getLogger("agent.rl_integration")

_LONG_ORDER_REJECT_TRUE_NET = 80.0
_EXTREME_DEADHEAD_KM = 150.0
_DEADHEAD_REJECT_TRUE_NET = 200.0
_PENALTY_MARGIN = 200.0
_POLICY_SCORE_WEIGHT = 300.0
_WAIT_DOMINANCE_MARGIN = 0.15
_WAIT_MIN_PROB = 0.35


class RLDecisionLayer:
    """Use trained RL weights to gently rerank heuristic cargo candidates."""

    def __init__(
        self,
        policy_path: str | Path | None = None,
        value_path: str | Path | None = None,
        scorer_path: str | Path | None = None,
    ) -> None:
        self._policy = PolicyNetworkNumpy()
        self._value = PositionValueNetworkNumpy()
        self._scorer = OrderScoringNetworkNumpy()
        self._loaded = False

        if policy_path and Path(policy_path).exists():
            self._policy.load(policy_path)
            self._loaded = self._policy.is_loaded
        if value_path and Path(value_path).exists():
            self._value.load(value_path)
        if scorer_path and Path(scorer_path).exists():
            self._scorer.load(scorer_path)

        if self._loaded:
            _logger.info(
                "RLDecisionLayer loaded: policy=%s value=%s scorer=%s",
                policy_path,
                value_path,
                scorer_path,
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
        constraints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self._loaded or not candidates:
            return candidates

        try:
            from agent.rl_env import ACTION_CARGO_0, _TOP_K, encode_state

            policy_candidates = [self._candidate_as_env_item(c) for c in candidates[:_TOP_K]]
            state_vec = encode_state(status, state_tracker, policy_candidates, constraints)
            state_feat = state_vec[:20]
            policy_probs, policy_value = self._policy.forward(state_vec)

            for idx, candidate in enumerate(candidates):
                base_score = self._candidate_value(candidate)
                scorer_score = 0.0
                if self._scorer.is_loaded:
                    cargo_feat = self._extract_cargo_features(candidate, state_tracker)
                    scorer_score = float(self._scorer.forward(state_feat, cargo_feat))

                pos_value = 0.0
                cargo = self._candidate_cargo(candidate)
                end = cargo.get("end", {})
                end_lat = end.get("lat")
                end_lng = end.get("lng")
                if self._value.is_loaded and end_lat is not None and end_lng is not None:
                    pos_feat = self._position_features(
                        float(end_lat),
                        float(end_lng),
                        status,
                        state_tracker,
                    )
                    pos_value = float(self._value.forward(pos_feat))

                action_idx = ACTION_CARGO_0 + idx
                policy_prob = (
                    float(policy_probs[action_idx])
                    if idx < _TOP_K and action_idx < len(policy_probs)
                    else 0.0
                )
                rl_score = (
                    base_score
                    + scorer_score
                    + pos_value * 0.3
                    + policy_prob * _POLICY_SCORE_WEIGHT
                )
                candidate["rl_score"] = float(rl_score)
                candidate["rl_base_score"] = float(base_score)
                candidate["rl_policy_prob"] = policy_prob
                candidate["rl_policy_value"] = float(policy_value)
                candidate["rl_position_value"] = float(pos_value)

            candidates.sort(key=lambda x: x.get("rl_score", 0.0), reverse=True)
        except Exception as exc:
            _logger.warning("RL rank_candidates failed: %s", exc)

        return candidates

    def select_best(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self._loaded or not candidates:
            return None

        safe = [c for c in candidates if not c.get("has_soft_penalty", False)]
        penalty = [c for c in candidates if c.get("has_soft_penalty", False)]

        best_safe = self._best_by_rl_score(safe)
        best_penalty = self._best_by_rl_score(penalty)

        chosen = None
        if best_penalty and best_safe:
            pn = float(best_penalty.get("rl_score", self._candidate_value(best_penalty)))
            sn = float(best_safe.get("rl_score", self._candidate_value(best_safe)))
            chosen = best_penalty if pn > sn + _PENALTY_MARGIN else best_safe
        elif best_safe:
            chosen = best_safe
        elif best_penalty:
            chosen = best_penalty

        if chosen is None or not self._passes_guard(chosen):
            return None
        return chosen

    def select_wait_action(
        self,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
        constraints: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._loaded:
            return None

        try:
            from agent.rl_env import ACTION_CARGO_0, _TOP_K, _WAIT_ACTION_DURATIONS, encode_state

            policy_candidates = [self._candidate_as_env_item(c) for c in candidates[:_TOP_K]]
            state_vec = encode_state(status, state_tracker, policy_candidates, constraints)
            policy_probs, policy_value = self._policy.forward(state_vec)
            wait_indices = [
                idx for idx in _WAIT_ACTION_DURATIONS
                if idx < len(policy_probs)
            ]
            if not wait_indices:
                return None
            wait_idx = max(wait_indices, key=lambda idx: float(policy_probs[idx]))
            wait_prob = float(policy_probs[wait_idx])
            cargo_count = min(_TOP_K, len(candidates))
            best_cargo_prob = (
                float(np.max(policy_probs[ACTION_CARGO_0:ACTION_CARGO_0 + cargo_count]))
                if cargo_count > 0 else 0.0
            )
            if (
                wait_prob >= _WAIT_MIN_PROB
                and wait_prob >= best_cargo_prob + _WAIT_DOMINANCE_MARGIN
            ):
                return {
                    "action": "wait",
                    "params": {"duration_minutes": _WAIT_ACTION_DURATIONS[wait_idx]},
                    "rl_wait_prob": wait_prob,
                    "rl_best_cargo_prob": best_cargo_prob,
                    "rl_policy_value": float(policy_value),
                }
        except Exception as exc:
            _logger.warning("RL select_wait_action failed: %s", exc)

        return None

    def should_use_rl(
        self,
        driver_id: str,
        state_tracker: Any,
        candidates: list[dict[str, Any]],
    ) -> bool:
        return self._loaded and len(candidates) >= 1

    @classmethod
    def _best_by_rl_score(cls, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.get("rl_score", cls._candidate_value(x)))

    @staticmethod
    def _candidate_value(candidate: dict[str, Any]) -> float:
        return float(
            candidate.get(
                "profit_search_score",
                candidate.get(
                    "true_net",
                    candidate.get("score", candidate.get("net_profit", 0.0)),
                ),
            )
            or 0.0
        )

    @staticmethod
    def _candidate_cargo(candidate: dict[str, Any]) -> dict[str, Any]:
        cargo = candidate.get("cargo")
        if isinstance(cargo, dict) and cargo:
            return cargo
        return {
            "cargo_id": candidate.get("cargo_id", ""),
            "cargo_name": candidate.get("cargo_name", ""),
            "price": candidate.get("price", 0.0),
            "cost_time_minutes": candidate.get(
                "cost_time_minutes",
                candidate.get("total_minutes", 1),
            ),
            "start": candidate.get("start", {}),
            "end": candidate.get("end", {}),
            "load_time": candidate.get("load_time"),
        }

    @classmethod
    def _candidate_as_env_item(cls, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "distance_km": candidate.get("distance_km", candidate.get("deadhead_km", 0.0)),
            "haul_km": candidate.get("haul_km", 0.0),
            "cargo": cls._candidate_cargo(candidate),
        }

    @classmethod
    def _passes_guard(cls, candidate: dict[str, Any]) -> bool:
        true_net = float(candidate.get("true_net", candidate.get("score", 0.0)) or 0.0)
        value = max(true_net, cls._candidate_value(candidate))
        total_minutes = int(candidate.get("total_minutes", 0) or 0)
        deadhead_km = float(candidate.get("deadhead_km", candidate.get("distance_km", 0.0)) or 0.0)

        if total_minutes > 600 and value < _LONG_ORDER_REJECT_TRUE_NET:
            return False
        if deadhead_km > _EXTREME_DEADHEAD_KM and value < _DEADHEAD_REJECT_TRUE_NET:
            return False
        return True

    @staticmethod
    def _extract_cargo_features(candidate: dict[str, Any], state_tracker: Any) -> np.ndarray:
        cargo = RLDecisionLayer._candidate_cargo(candidate)
        price = float(cargo.get("price", 0.0) or 0.0)
        deadhead_km = float(candidate.get("distance_km", candidate.get("deadhead_km", 0.0)) or 0.0)
        haul_km = float(candidate.get("haul_km", 0.0) or 0.0)
        cost_min = max(1, int(cargo.get("cost_time_minutes", 1) or 1))
        net_profit = float(candidate.get("net_profit", price - (deadhead_km + haul_km) * 1.5) or 0.0)
        time_efficiency = float(candidate.get("time_efficiency", net_profit / cost_min) or 0.0)
        is_preferred = float(candidate.get("is_preferred_category", candidate.get("preference_match", False)))

        end = cargo.get("end", {})
        spatial_val = 0.0
        end_lat = end.get("lat")
        end_lng = end.get("lng")
        if end_lat is not None and end_lng is not None and state_tracker is not None:
            get_spatial_value = getattr(state_tracker, "get_spatial_value", None)
            if not callable(get_spatial_value):
                get_spatial_value = getattr(state_tracker, "_get_spatial_value", None)
            if callable(get_spatial_value):
                spatial_val = float(get_spatial_value(float(end_lat), float(end_lng)) or 0.0)

        return np.array(
            [
                min(price / 5000.0, 1.0),
                min(deadhead_km / 200.0, 1.0),
                min(haul_km / 1000.0, 1.0),
                min(cost_min / 480.0, 1.0),
                max(min(net_profit / 3000.0, 1.0), -1.0),
                max(min(spatial_val / 10000.0, 1.0), -1.0),
                max(min(time_efficiency / 50.0, 1.0), -1.0),
                is_preferred,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _position_features(
        lat: float,
        lng: float,
        status: dict[str, Any],
        state_tracker: Any,
    ) -> np.ndarray:
        sim_min = int(status.get("simulation_progress_minutes", 0))
        time_of_day = (sim_min % 1440) / 1440.0
        day_of_month = min(sim_min / 44640.0, 1.0)
        income = getattr(state_tracker, "total_income", 0.0) / 500000.0
        mileage = getattr(state_tracker, "total_mileage_km", 0.0) / 50000.0
        deadhead = getattr(state_tracker, "total_deadhead_km", 0.0) / 5000.0
        orders = getattr(state_tracker, "total_orders", 0) / 200.0
        max_rest = getattr(state_tracker, "get_max_continuous_rest_today", lambda _: 0)(sim_min) / 480.0
        rest_deficit = max(0.0, (240.0 - max_rest * 480.0) / 240.0) if max_rest < 0.5 else 0.0

        return np.array(
            [
                (lat - 18.0) / 35.0,
                (lng - 73.0) / 62.0,
                time_of_day,
                day_of_month,
                min(income, 1.0),
                min(mileage, 1.0),
                min(deadhead, 1.0),
                min(orders, 1.0),
                min(max_rest, 1.0),
                min(rest_deficit, 1.0),
            ],
            dtype=np.float32,
        )
