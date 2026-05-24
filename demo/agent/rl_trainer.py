"""PPO 训练循环：收集轨迹、GAE 优势估计、策略/价值更新。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent.rl_env import DriverRLEnv, _ACTION_DIM, _STATE_DIM

_logger = logging.getLogger("agent.rl_trainer")

# ─────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_epochs: int = 10
    total_episodes: int = 1000
    eval_interval: int = 50
    save_interval: int = 100
    model_dir: str = "demo/agent/models"
    curriculum_start_days: int = 5
    curriculum_end_days: int = 31
    curriculum_ramp_episodes: int = 200


# ─────────────────────────────────────────────────────────
# 轨迹存储
# ─────────────────────────────────────────────────────────


@dataclass
class TrajectoryStep:
    state: np.ndarray
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    action_mask: np.ndarray | None = None


@dataclass
class Trajectory:
    steps: list[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    final_income: float = 0.0


# ─────────────────────────────────────────────────────────
# PPO 训练器
# ─────────────────────────────────────────────────────────


class PPOTrainer:
    """PPO 训练主循环。"""

    def __init__(self, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        self._policy_net: Any = None  # PolicyNetwork (PyTorch)
        self._optimizer: Any = None
        self._episode_rewards: list[float] = []
        self._episode_incomes: list[float] = []

    def train(
        self,
        envs: list[DriverRLEnv] | None = None,
        eval_env: DriverRLEnv | None = None,
        env_factory: Callable[[], list[DriverRLEnv]] | None = None,
        eval_env_factory: Callable[[], list[DriverRLEnv]] | None = None,
    ) -> dict[str, Any]:
        """主训练入口。

        Args:
            envs: 训练环境列表（每个司机一个）。
            eval_env: 评估环境（可选）。

        Returns:
            训练统计信息。
        """
        try:
            import torch
            from agent.rl_models import PolicyNetwork
        except ImportError:
            raise RuntimeError("训练需要 PyTorch，请安装: pip install torch>=2.0.0")

        cfg = self.config
        if envs is None and env_factory is None:
            raise ValueError("Either envs or env_factory must be provided")
        model_dir = Path(cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 初始化网络
        self._policy_net = PolicyNetwork(state_dim=_STATE_DIM, action_dim=_ACTION_DIM)
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=cfg.learning_rate)

        _logger.info(
            "PPO training start episodes=%d envs=%s lr=%.1e",
            cfg.total_episodes,
            "factory" if env_factory is not None else len(envs or []),
            cfg.learning_rate,
        )

        best_income = float("-inf")
        stats: dict[str, Any] = {
            "episode_rewards": [],
            "episode_incomes": [],
            "eval_incomes": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "approx_kls": [],
            "clip_fractions": [],
            "best_income": 0.0,
            "model_dir": str(model_dir),
        }

        for episode in range(cfg.total_episodes):
            # 课程学习：逐步增加仿真天数
            sim_days = self._curriculum_days(episode)

            # 收集轨迹
            episode_envs = env_factory() if env_factory is not None else envs
            if not episode_envs:
                continue

            trajectories = []
            for env in episode_envs:
                traj = self._collect_trajectory(env, sim_days)
                trajectories.append(traj)

            # 计算 GAE 和 returns
            all_states, all_actions, all_old_log_probs, all_advantages, all_returns, all_masks = (
                [], [], [], [], [], []
            )
            for traj in trajectories:
                advantages = self._compute_gae(traj.steps)
                returns = self._compute_returns(traj.steps)
                for step, adv, ret in zip(traj.steps, advantages, returns):
                    all_states.append(step.state)
                    all_actions.append(step.action)
                    all_old_log_probs.append(step.log_prob)
                    all_advantages.append(adv)
                    all_returns.append(ret)
                    all_masks.append(step.action_mask)

            if not all_states:
                continue

            device = self._policy_net.device
            states_t = torch.FloatTensor(np.array(all_states)).to(device)
            actions_t = torch.LongTensor(all_actions).to(device)
            old_log_probs_t = torch.FloatTensor(all_old_log_probs).to(device)
            advantages_t = torch.FloatTensor(all_advantages).to(device)
            returns_t = torch.FloatTensor(all_returns).to(device)
            masks_t = self._masks_to_tensor(all_masks, device)

            # 标准化优势
            if len(advantages_t) > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # PPO 更新
            update_metrics = self._update_policy_minibatches(
                states_t,
                actions_t,
                old_log_probs_t,
                advantages_t,
                returns_t,
                masks_t,
            )

            # 记录统计
            avg_reward = np.mean([t.total_reward for t in trajectories])
            avg_income = np.mean([t.final_income for t in trajectories])
            self._episode_rewards.append(avg_reward)
            self._episode_incomes.append(avg_income)
            stats["episode_rewards"].append(avg_reward)
            stats["episode_incomes"].append(avg_income)
            for key, stat_key in (
                ("policy_loss", "policy_losses"),
                ("value_loss", "value_losses"),
                ("entropy", "entropies"),
                ("approx_kl", "approx_kls"),
                ("clip_fraction", "clip_fractions"),
            ):
                stats[stat_key].append(update_metrics.get(key, 0.0))

            if (episode + 1) % 10 == 0:
                _logger.info(
                    "episode %d/%d days=%d avg_reward=%.1f avg_income=%.0f kl=%.4f clip=%.3f entropy=%.3f",
                    episode + 1, cfg.total_episodes, sim_days, avg_reward, avg_income,
                    update_metrics.get("approx_kl", 0.0),
                    update_metrics.get("clip_fraction", 0.0),
                    update_metrics.get("entropy", 0.0),
                )

            # 评估
            should_eval = (
                (eval_env is not None or eval_env_factory is not None)
                and (
                    (episode + 1) % max(1, cfg.eval_interval) == 0
                    or episode + 1 == cfg.total_episodes
                )
            )
            if should_eval:
                if eval_env_factory is not None:
                    eval_income = self._evaluate_envs(eval_env_factory(), sim_days)
                else:
                    eval_income = self._evaluate(eval_env, sim_days)
                stats["eval_incomes"].append(eval_income)
                _logger.info("eval episode=%d income=%.0f", episode + 1, eval_income)
                if eval_income > best_income:
                    best_income = eval_income
                    stats["best_income"] = best_income
                    self._policy_net.export_numpy(model_dir / "policy_best.npz")
            elif eval_env is None and eval_env_factory is None and avg_income > best_income:
                best_income = avg_income
                stats["best_income"] = best_income
                self._policy_net.export_numpy(model_dir / "policy_best.npz")

            # 定期保存
            if (episode + 1) % cfg.save_interval == 0:
                self._policy_net.export_numpy(model_dir / f"policy_ep{episode + 1}.npz")

        # 最终保存
        self._policy_net.export_numpy(model_dir / "policy_final.npz")
        _logger.info("training done best_income=%.0f", best_income)
        return stats

    def _collect_trajectory(self, env: DriverRLEnv, max_days: int = 31) -> Trajectory:
        """运行一个 episode 收集轨迹。"""
        import torch

        traj = Trajectory()
        obs, _ = env.reset()
        done = False
        max_steps = max_days * 60  # 每天约 60 个决策步

        for _ in range(max_steps):
            if done:
                break

            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, value = self._policy_net(state_t)
            action_mask = self._env_action_mask(env)
            logits = self._apply_action_mask(logits, action_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = float(dist.log_prob(action).item())
            value = float(value.item())
            action_idx = int(action.item())

            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            traj.steps.append(TrajectoryStep(
                state=obs,
                action=action_idx,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
                action_mask=action_mask,
            ))
            traj.total_reward += reward
            obs = next_obs

        traj.final_income = info.get("net_income", info.get("total_income", 0.0)) if info else 0.0
        return traj

    def _compute_gae(self, steps: list[TrajectoryStep]) -> list[float]:
        """计算 GAE 优势函数。"""
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        advantages = []
        gae = 0.0

        for i in reversed(range(len(steps))):
            if i == len(steps) - 1:
                next_value = 0.0
            else:
                next_value = steps[i + 1].value

            delta = steps[i].reward + gamma * next_value * (1 - int(steps[i].done)) - steps[i].value
            gae = delta + gamma * lam * (1 - int(steps[i].done)) * gae
            advantages.insert(0, gae)

        return advantages

    def _compute_returns(self, steps: list[TrajectoryStep]) -> list[float]:
        """计算折扣回报。"""
        gamma = self.config.gamma
        returns = []
        G = 0.0
        for step in reversed(steps):
            G = step.reward + gamma * G * (1 - int(step.done))
            returns.insert(0, G)
        return returns

    @staticmethod
    def _env_action_mask(env: DriverRLEnv) -> np.ndarray:
        getter = getattr(env, "get_action_mask", None)
        if callable(getter):
            mask = np.asarray(getter(), dtype=np.bool_)
        else:
            mask = np.ones(_ACTION_DIM, dtype=np.bool_)
        if mask.shape != (_ACTION_DIM,):
            mask = np.ones(_ACTION_DIM, dtype=np.bool_)
        if not mask.any():
            mask[0:2] = True
        return mask

    @staticmethod
    def _masks_to_tensor(masks: list[np.ndarray | None], device: Any) -> Any:
        if not masks or all(mask is None for mask in masks):
            return None
        normalized = []
        for mask in masks:
            if mask is None:
                normalized.append(np.ones(_ACTION_DIM, dtype=np.bool_))
                continue
            arr = np.asarray(mask, dtype=np.bool_)
            if arr.shape != (_ACTION_DIM,) or not arr.any():
                arr = np.ones(_ACTION_DIM, dtype=np.bool_)
            normalized.append(arr)
        import torch

        return torch.as_tensor(np.array(normalized), dtype=torch.bool, device=device)

    @staticmethod
    def _apply_action_mask(logits: Any, action_mask: np.ndarray | Any | None) -> Any:
        if action_mask is None:
            return logits
        import torch

        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0).expand_as(logits)
        return logits.masked_fill(~mask_t, -1e9)

    def _update_policy_minibatches(
        self,
        states: Any,
        actions: Any,
        old_log_probs: Any,
        advantages: Any,
        returns: Any,
        action_masks: Any = None,
    ) -> dict[str, float]:
        import torch

        n_items = int(states.shape[0])
        if n_items <= 0:
            return {}
        batch_size = max(1, min(int(self.config.batch_size), n_items))
        metric_rows: list[dict[str, float]] = []

        for _ in range(self.config.n_epochs):
            indices = torch.randperm(n_items, device=states.device)
            for start in range(0, n_items, batch_size):
                idx = indices[start:start + batch_size]
                batch_masks = action_masks[idx] if action_masks is not None else None
                metrics = self._update_policy(
                    states[idx],
                    actions[idx],
                    old_log_probs[idx],
                    advantages[idx],
                    returns[idx],
                    batch_masks,
                )
                if isinstance(metrics, dict):
                    metric_rows.append(metrics)

        if not metric_rows:
            return {}
        return {
            key: float(np.mean([row.get(key, 0.0) for row in metric_rows]))
            for key in metric_rows[0]
        }

    def _update_policy(
        self,
        states: Any,
        actions: Any,
        old_log_probs: Any,
        advantages: Any,
        returns: Any,
        action_masks: Any = None,
    ) -> dict[str, float]:
        """PPO clip 更新。"""
        import torch
        import torch.nn.functional as F

        cfg = self.config
        logits, values = self._policy_net(states)
        logits = self._apply_action_mask(logits, action_masks)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), cfg.max_grad_norm)
        self._optimizer.step()
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean()
        return {
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "approx_kl": float(approx_kl.detach().item()),
            "clip_fraction": float(clip_fraction.detach().item()),
        }

    def _evaluate(self, env: DriverRLEnv, max_days: int = 31) -> float:
        """评估当前策略的月收入。"""
        import torch

        obs, _ = env.reset()
        done = False
        max_steps = max_days * 60
        income = 0.0

        for _ in range(max_steps):
            if done:
                break
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self._policy_net(state_t)
            logits = self._apply_action_mask(logits, self._env_action_mask(env))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            income = info.get("net_income", info.get("total_income", income)) if info else income

        return income

    def _evaluate_envs(self, envs: list[DriverRLEnv], max_days: int = 31) -> float:
        if not envs:
            return 0.0
        incomes = [self._evaluate(env, max_days) for env in envs]
        return float(np.mean(incomes))

    def _curriculum_days(self, episode: int) -> int:
        """课程学习：逐步增加仿真天数。"""
        cfg = self.config
        if episode >= cfg.curriculum_ramp_episodes:
            return cfg.curriculum_end_days
        progress = episode / max(1, cfg.curriculum_ramp_episodes)
        days = cfg.curriculum_start_days + int(
            progress * (cfg.curriculum_end_days - cfg.curriculum_start_days)
        )
        return days
