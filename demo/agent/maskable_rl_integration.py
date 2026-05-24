"""MaskablePPO-first RL integration layer for online inference.

Decision flow:
    build TopK candidates
    -> encode state
    -> compute action mask
    -> MaskablePPO predicts ranked actions
    -> veto unsafe actions
    -> execute first accepted action
    -> fallback to heuristic if needed
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger("agent.maskable_rl_integration")

# Veto thresholds (same as rl_integration.py)
_LONG_ORDER_MINUTES = 600
_LONG_ORDER_REJECT_TRUE_NET = 80.0
_EXTREME_DEADHEAD_KM = 150.0
_DEADHEAD_REJECT_TRUE_NET = 200.0
_NEGATIVE_NET_THRESHOLD = 0.0
_PENALTY_COVER_MARGIN = 100.0
_MAX_CONSECUTIVE_WAIT_MIN = 180.0


class MaskableRLDecisionLayer:
    """Use a trained MaskablePPO model as the primary decision maker.

    Falls back to heuristic on missing model, exception, or vetoed actions.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self._sb3_model: Any = None
        self._numpy_policy: Any = None
        self._model_type = "none"
        self._loaded = False

        if model_path is not None:
            self._try_load(str(model_path))
        elif model_dir is not None:
            self._auto_discover(Path(model_dir))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> str:
        return self._model_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
        constraints: list[dict[str, Any]],
        driver_config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Select an action using MaskablePPO.

        Returns a result dict on success, or None to trigger heuristic fallback.
        """
        if not self._loaded or not candidates:
            return None

        try:
            from agent.rl_env import (
                ACTION_CARGO_0,
                ACTION_FORCE_REST,
                ACTION_REPOSITION_HOME,
                ACTION_REPOSITION_HIGH_VALUE_DEST,
                ACTION_REPOSITION_HOTZONE,
                ACTION_REPOSITION_SUPPLY_ZONE,
                _TOP_K,
                encode_state,
            )

            # Build TopK env items and encode state
            env_candidates = [_as_env_item(c) for c in candidates[:_TOP_K]]
            state_vec = encode_state(status, state_tracker, env_candidates, constraints)

            # Compute action mask
            mask = self._compute_action_mask(
                candidates, status, state_tracker, constraints, driver_config,
            )

            # Get model prediction + probs
            action_idx, probs = self._predict(state_vec, mask)

            # Veto loop: try actions in probability order
            result = self._try_accept_action(
                action_idx, probs, mask, candidates, status, state_tracker,
            )
            if result is not None:
                return result

            # All actions vetoed
            _logger.info("All MaskablePPO actions vetoed, falling back to heuristic")
            return None

        except Exception as exc:
            _logger.warning("MaskableRLDecisionLayer.decide failed: %s", exc)
            return None

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
        constraints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Annotate candidates with MaskablePPO scores and re-sort.

        Compatible with upstream code that expects rank_candidates + select_best.
        """
        if not self._loaded or not candidates:
            return candidates

        try:
            from agent.rl_env import ACTION_CARGO_0, _TOP_K, encode_state

            env_candidates = [_as_env_item(c) for c in candidates[:_TOP_K]]
            state_vec = encode_state(status, state_tracker, env_candidates, constraints)

            mask = self._compute_action_mask(
                candidates, status, state_tracker, constraints, None,
            )
            probs, value = self._get_probs_and_value(state_vec, mask)

            for idx, candidate in enumerate(candidates):
                if idx >= _TOP_K:
                    candidate["rl_score"] = float(_candidate_value(candidate))
                    candidate["rl_base_score"] = candidate["rl_score"]
                    candidate["rl_policy_prob"] = 0.0
                    candidate["rl_policy_value"] = float(value)
                    continue
                action_idx = ACTION_CARGO_0 + idx
                prob = float(probs[action_idx]) if action_idx < len(probs) else 0.0
                base_score = _candidate_value(candidate)
                rl_score = base_score + prob * 300.0

                candidate["rl_score"] = float(rl_score)
                candidate["rl_base_score"] = float(base_score)
                candidate["rl_policy_prob"] = prob
                candidate["rl_policy_value"] = float(value)

            candidates.sort(key=lambda x: x.get("rl_score", 0.0), reverse=True)
        except Exception as exc:
            _logger.warning("MaskableRLDecisionLayer.rank_candidates failed: %s", exc)

        return candidates

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _auto_discover(self, model_dir: Path) -> None:
        """Search for model files in priority order."""
        # 1. sb3 .zip
        for name in ("policy_maskable_best.zip", "policy_maskable_final.zip"):
            p = model_dir / name
            if p.exists() and self._try_load(str(p)):
                return
        # 2. numpy .npz
        for name in ("policy_maskable_best_np.npz", "policy_maskable_final_np.npz",
                      "policy_maskable_best.npz", "policy_maskable_final.npz"):
            p = model_dir / name
            if p.exists() and self._try_load(str(p)):
                return

    def _try_load(self, path: str) -> bool:
        """Try loading as sb3 .zip first, then as .npz."""
        if path.endswith(".zip"):
            if self._try_load_sb3(path):
                return True
        if self._try_load_numpy(path):
            return True
        # For .zip, also try .npz fallback
        if path.endswith(".zip"):
            npz_path = path.replace(".zip", "_np.npz")
            if Path(npz_path).exists() and self._try_load_numpy(npz_path):
                return True
        return False

    def _try_load_sb3(self, path: str) -> bool:
        try:
            from sb3_contrib import MaskablePPO
            self._sb3_model = MaskablePPO.load(path)
            self._model_type = "sb3"
            self._loaded = True
            _logger.info("MaskableRLDecisionLayer loaded sb3 model from %s", path)
            return True
        except ImportError:
            _logger.debug("sb3-contrib not available, skipping .zip load")
        except Exception as exc:
            _logger.debug("Failed to load sb3 model from %s: %s", path, exc)
        return False

    def _try_load_numpy(self, path: str) -> bool:
        try:
            from agent.maskable_policy_numpy import MaskablePolicyNumpy
            policy = MaskablePolicyNumpy()
            policy.load(path)
            if policy.is_loaded:
                self._numpy_policy = policy
                self._model_type = "numpy"
                self._loaded = True
                _logger.info("MaskableRLDecisionLayer loaded numpy model from %s", path)
                return True
        except Exception as exc:
            _logger.debug("Failed to load numpy model from %s: %s", path, exc)
        return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(
        self, state_vec: np.ndarray, mask: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Get action and probs from the loaded model."""
        if self._model_type == "sb3":
            obs = state_vec.reshape(1, -1).astype(np.float32)
            action, _ = self._sb3_model.predict(
                obs, deterministic=True, action_masks=mask.reshape(1, -1),
            )
            # Also get probs for ranked fallback
            probs = self._sb3_probs(obs, mask)
            return int(action[0]), probs
        else:
            action = self._numpy_policy.get_action(state_vec, action_mask=mask, deterministic=True)
            probs = self._numpy_policy.get_action_probs(state_vec, action_mask=mask)
            return action, probs

    def _get_probs_and_value(
        self, state_vec: np.ndarray, mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Get masked probs and state value."""
        if self._model_type == "sb3":
            obs = state_vec.reshape(1, -1).astype(np.float32)
            probs = self._sb3_probs(obs, mask)
            # SB3 doesn't easily expose value; use numpy if available or 0
            return probs, 0.0
        else:
            probs = self._numpy_policy.get_action_probs(state_vec, action_mask=mask)
            _, value = self._numpy_policy.forward(state_vec)
            return probs, value

    def _sb3_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract masked action probabilities from the sb3 model."""
        import torch

        try:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                features = self._sb3_model.policy.extract_features(obs_t)
                policy_latent = self._sb3_model.policy.mlp_extractor.policy_net(features)
                logits = self._sb3_model.policy.action_net(policy_latent)
                logits_np = logits.cpu().numpy().flatten()

                # Apply mask: set masked logits to -inf
                mask_bool = np.array(mask, dtype=bool)
                logits_np[~mask_bool] = -1e9

                # Softmax with numerical stability
                logits_np = logits_np - logits_np.max()
                exp_logits = np.exp(logits_np)
                probs = exp_logits / exp_logits.sum()
                return probs
        except Exception:
            from agent.rl_models import _ACTION_DIM
            return np.ones(_ACTION_DIM) / _ACTION_DIM

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _compute_action_mask(
        self,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
        constraints: list[dict[str, Any]],
        driver_config: dict[str, Any] | None,
    ) -> np.ndarray:
        """Compute the action mask matching the environment's logic."""
        from agent.action_masking import ActionMaskSpec, apply_revenue_risk_mask
        from agent.rl_env import _ACTION_DIM, _TOP_K

        mask = np.zeros(_ACTION_DIM, dtype=bool)

        # Wait actions always available
        mask[0] = True  # wait_15
        mask[1] = True  # wait_30
        mask[2] = True  # wait_60

        # Cargo actions: available for TopK candidates
        cargo_count = min(len(candidates), _TOP_K)
        for i in range(cargo_count):
            mask[3 + i] = True

        # Reposition actions: check if targets exist
        if _has_home(status, state_tracker):
            mask[13] = True
        if _has_hotzone(status, state_tracker):
            mask[14] = True
        if _has_supply_zone(status, state_tracker):
            mask[15] = True
        if _has_high_value_dest(status, state_tracker):
            mask[16] = True

        # Force rest: if rest deficit
        rest_today = _get_rest_today(status, state_tracker)
        required_rest = _get_required_rest(status, driver_config)
        if rest_today < required_rest:
            mask[17] = True

        # Apply revenue/risk mask on top
        if driver_config is None:
            driver_config = {}
        spec = ActionMaskSpec(
            action_dim=_ACTION_DIM,
            cargo_start=3,
            top_k=_TOP_K,
            wait_actions=(0, 1, 2),
            force_rest_action=17,
        )
        env_items = [_as_env_item(c) for c in candidates[:_TOP_K]]
        total_deadhead = _get_total_deadhead(status, state_tracker)
        consecutive_wait = _get_consecutive_wait(status, state_tracker)
        mask, _reasons = apply_revenue_risk_mask(
            mask,
            cargo_list=env_items,
            spec=spec,
            driver_config=driver_config,
            total_deadhead_km=total_deadhead,
            rest_today_min=rest_today,
            consecutive_wait_min=consecutive_wait,
        )
        return mask

    # ------------------------------------------------------------------
    # Veto
    # ------------------------------------------------------------------

    def _try_accept_action(
        self,
        action_idx: int,
        probs: np.ndarray,
        mask: np.ndarray,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
        state_tracker: Any,
    ) -> dict[str, Any] | None:
        """Try to accept the selected action. If vetoed, try next-best."""
        from agent.rl_env import (
            ACTION_CARGO_0,
            ACTION_FORCE_REST,
            ACTION_REPOSITION_HOME,
            ACTION_REPOSITION_HIGH_VALUE_DEST,
            ACTION_REPOSITION_HOTZONE,
            ACTION_REPOSITION_SUPPLY_ZONE,
            _TOP_K,
            _WAIT_ACTION_DURATIONS,
        )

        # Build action priority list (highest prob first)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return None
        priority = valid_indices[np.argsort(probs[valid_indices])[::-1]]

        for idx in priority:
            idx = int(idx)
            veto_reason = None
            positive_cargo_available = _has_positive_candidate(candidates)

            # Wait actions
            if idx in _WAIT_ACTION_DURATIONS:
                consecutive_wait = _get_consecutive_wait(status, state_tracker)
                if positive_cargo_available:
                    veto_reason = "positive_cargo_available"
                elif consecutive_wait >= _MAX_CONSECUTIVE_WAIT_MIN:
                    veto_reason = f"wait_loop ({consecutive_wait:.0f}min)"
                else:
                    return _make_result(
                        action="wait",
                        action_idx=idx,
                        params={"duration_minutes": _WAIT_ACTION_DURATIONS[idx]},
                        prob=float(probs[idx]),
                        mask_reasons={},
                        veto_reason=None,
                        model_type=self._model_type,
                    )

            # Cargo actions
            elif _is_cargo_action(idx):
                cargo_idx = idx - ACTION_CARGO_0
                if cargo_idx < len(candidates):
                    candidate = candidates[cargo_idx]
                    if not _passes_guard(candidate):
                        veto_reason = "guard_reject"
                    elif _candidate_value(candidate) < _NEGATIVE_NET_THRESHOLD:
                        veto_reason = "negative_net"
                    else:
                        return _make_result(
                            action="cargo",
                            action_idx=idx,
                            params={"candidate": candidate},
                            prob=float(probs[idx]),
                            mask_reasons={},
                            veto_reason=None,
                            model_type=self._model_type,
                        )

            # Reposition actions
            elif idx == ACTION_REPOSITION_HOME:
                return _make_result("reposition_home", idx, {}, float(probs[idx]), {}, None, self._model_type)
            elif idx == ACTION_REPOSITION_HOTZONE:
                return _make_result("reposition_hotzone", idx, {}, float(probs[idx]), {}, None, self._model_type)
            elif idx == ACTION_REPOSITION_SUPPLY_ZONE:
                return _make_result("reposition_supply_zone", idx, {}, float(probs[idx]), {}, None, self._model_type)
            elif idx == ACTION_REPOSITION_HIGH_VALUE_DEST:
                return _make_result("reposition_high_value_dest", idx, {}, float(probs[idx]), {}, None, self._model_type)

            # Force rest
            elif idx == ACTION_FORCE_REST:
                if positive_cargo_available:
                    veto_reason = "positive_cargo_available"
                else:
                    return _make_result("force_rest", idx, {}, float(probs[idx]), {}, None, self._model_type)

            if veto_reason:
                _logger.debug("Action %d vetoed: %s", idx, veto_reason)

        return None


# ======================================================================
# Module-level helpers
# ======================================================================


def _make_result(
    action: str,
    action_idx: int,
    params: dict,
    prob: float,
    mask_reasons: dict,
    veto_reason: str | None,
    model_type: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "action_idx": action_idx,
        "params": params,
        "rl_prob": prob,
        "mask_reasons": mask_reasons,
        "veto_reason": veto_reason,
        "fallback_used": False,
        "model_type": model_type,
    }


def _is_cargo_action(action_idx: int) -> bool:
    from agent.rl_env import ACTION_CARGO_0, ACTION_CARGO_9
    return ACTION_CARGO_0 <= action_idx <= ACTION_CARGO_9


def _as_env_item(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize candidate to env format. Same as RLDecisionLayer._candidate_as_env_item."""
    from agent.rl_integration import RLDecisionLayer
    return RLDecisionLayer._candidate_as_env_item(candidate)


def _candidate_value(candidate: dict[str, Any]) -> float:
    """Extract best available score. Same as RLDecisionLayer._candidate_value."""
    from agent.rl_integration import RLDecisionLayer
    return RLDecisionLayer._candidate_value(candidate)


def _candidate_true_net(candidate: dict[str, Any]) -> float:
    return float(candidate.get("true_net", candidate.get("score", 0.0)) or 0.0)


def _candidate_net_profit(candidate: dict[str, Any]) -> float:
    value = candidate.get("net_profit")
    if value is not None:
        return float(value or 0.0)
    return _candidate_true_net(candidate)


def _has_positive_candidate(candidates: list[dict[str, Any]]) -> bool:
    for candidate in candidates:
        if candidate.get("hard_penalty", 0):
            continue
        if not _passes_guard(candidate):
            continue
        if _candidate_true_net(candidate) > 0 and _candidate_net_profit(candidate) > 0:
            return True
    return False


def _passes_guard(candidate: dict[str, Any]) -> bool:
    """Guard check. Same as RLDecisionLayer._passes_guard."""
    from agent.rl_integration import RLDecisionLayer
    return RLDecisionLayer._passes_guard(candidate)


def _first_non_none(*values: Any) -> Any:
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def _has_home(status: dict, state_tracker: Any) -> bool:
    return _first_non_none(status.get("home_base"), getattr(state_tracker, "home_base", None)) is not None


def _has_hotzone(status: dict, state_tracker: Any) -> bool:
    return _first_non_none(status.get("hotzone"), getattr(state_tracker, "hotzone", None)) is not None


def _has_supply_zone(status: dict, state_tracker: Any) -> bool:
    return _first_non_none(status.get("supply_zone"), getattr(state_tracker, "supply_zone", None)) is not None


def _has_high_value_dest(status: dict, state_tracker: Any) -> bool:
    return _first_non_none(status.get("high_value_dest"), getattr(state_tracker, "high_value_dest", None)) is not None


def _get_rest_today(status: dict, state_tracker: Any) -> float:
    val = _first_non_none(status.get("rest_today_min"), getattr(state_tracker, "rest_today_min", None))
    return float(val) if val is not None else 0.0


def _get_required_rest(status: dict, driver_config: dict | None) -> float:
    if driver_config:
        return float(driver_config.get("required_rest_min", 480))
    return 480.0


def _get_total_deadhead(status: dict, state_tracker: Any) -> float:
    val = _first_non_none(status.get("total_deadhead_km"), getattr(state_tracker, "total_deadhead_km", None))
    return float(val) if val is not None else 0.0


def _get_consecutive_wait(status: dict, state_tracker: Any) -> float:
    val = _first_non_none(status.get("consecutive_wait_min"), getattr(state_tracker, "consecutive_wait_min", None))
    return float(val) if val is not None else 0.0
