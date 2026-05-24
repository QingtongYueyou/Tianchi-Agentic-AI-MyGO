"""Heuristic teacher data and warm start utilities for MaskablePPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent.rl_env import (
    ACTION_CARGO_0,
    ACTION_CARGO_9,
    ACTION_FORCE_REST,
    ACTION_REPOSITION_HIGH_VALUE_DEST,
    ACTION_REPOSITION_SUPPLY_ZONE,
    ACTION_WAIT_30,
)


@dataclass
class TeacherSample:
    state: np.ndarray
    action: int
    action_mask: np.ndarray


def choose_teacher_action(env: Any, action_mask: np.ndarray, min_cargo_score: float = 0.0) -> int:
    """Choose a masked heuristic action from the current environment state."""
    mask = np.asarray(action_mask, dtype=np.bool_)

    best_action = None
    best_score = float("-inf")
    for idx, item in enumerate(getattr(env, "_current_cargo_list", [])[:10]):
        action = ACTION_CARGO_0 + idx
        if action > ACTION_CARGO_9 or action >= len(mask) or not mask[action]:
            continue
        score = _candidate_value(item)
        if score > best_score:
            best_score = score
            best_action = action

    if best_action is not None and best_score >= min_cargo_score:
        return int(best_action)

    if ACTION_FORCE_REST < len(mask) and mask[ACTION_FORCE_REST]:
        return ACTION_FORCE_REST
    for action in (ACTION_REPOSITION_HIGH_VALUE_DEST, ACTION_REPOSITION_SUPPLY_ZONE):
        if action < len(mask) and mask[action]:
            return action
    if ACTION_WAIT_30 < len(mask) and mask[ACTION_WAIT_30]:
        return ACTION_WAIT_30
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0


def collect_teacher_samples(
    envs: list[Any],
    *,
    max_steps: int,
    min_cargo_score: float = 0.0,
) -> list[TeacherSample]:
    samples: list[TeacherSample] = []
    for env in envs:
        obs, _ = env.reset()
        done = False
        for _ in range(max_steps):
            if done:
                break
            mask = env.get_action_mask()
            action = choose_teacher_action(env, mask, min_cargo_score=min_cargo_score)
            samples.append(
                TeacherSample(
                    state=np.asarray(obs, dtype=np.float32),
                    action=int(action),
                    action_mask=np.asarray(mask, dtype=np.bool_),
                )
            )
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return samples


def pretrain_maskable_policy(
    model: Any,
    samples: list[TeacherSample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float]:
    if not samples or epochs <= 0:
        return {"teacher_samples": float(len(samples)), "teacher_loss": 0.0}

    import torch

    device = model.policy.device
    states = torch.as_tensor(np.array([s.state for s in samples]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([s.action for s in samples], dtype=torch.long, device=device)
    masks = torch.as_tensor(np.array([s.action_mask for s in samples]), dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)

    n_items = int(states.shape[0])
    batch_size = max(1, min(batch_size, n_items))
    losses: list[float] = []
    for _ in range(epochs):
        indices = torch.randperm(n_items, device=device)
        for start in range(0, n_items, batch_size):
            idx = indices[start:start + batch_size]
            _, log_prob, entropy = model.policy.evaluate_actions(
                states[idx],
                actions[idx],
                action_masks=masks[idx],
            )
            entropy_bonus = entropy.mean() if entropy is not None else 0.0
            loss = -log_prob.mean() - 0.001 * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))

    return {
        "teacher_samples": float(len(samples)),
        "teacher_loss": float(np.mean(losses)) if losses else 0.0,
    }


def _candidate_value(item: dict[str, Any]) -> float:
    for key in ("profit_search_score", "true_net", "score", "net_profit"):
        value = item.get(key)
        if value is not None:
            return float(value or 0.0)
    cargo = item.get("cargo", {})
    price = float(cargo.get("price", item.get("price", 0.0)) or 0.0)
    deadhead_km = float(item.get("distance_km", item.get("deadhead_km", 0.0)) or 0.0)
    haul_km = float(item.get("haul_km", 0.0) or 0.0)
    return price - (deadhead_km + haul_km) * 1.5
