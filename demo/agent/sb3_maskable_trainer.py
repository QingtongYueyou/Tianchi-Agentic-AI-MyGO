"""MaskablePPO training path built on sb3-contrib."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_logger = logging.getLogger("agent.sb3_maskable_trainer")


@dataclass
class MaskablePPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 256
    n_epochs: int = 10
    n_steps: int = 2048
    total_timesteps: int = 2048
    eval_interval: int = 50
    save_interval: int = 100
    model_dir: str = "demo/agent/models"
    logs_dir: str = "demo/results/rl_logs"
    seed: int | None = None
    verbose: int = 1
    teacher_pretrain_steps: int = 0
    teacher_pretrain_epochs: int = 0
    teacher_batch_size: int = 64
    teacher_learning_rate: float = 1e-4
    teacher_min_cargo_score: float = 0.0
    base_model_path: str | None = None


class MaskablePPOTrainer:
    """Train and evaluate sb3-contrib MaskablePPO on DriverRLEnv instances."""

    def __init__(self, config: MaskablePPOConfig | None = None) -> None:
        self.config = config or MaskablePPOConfig()
        self._model: Any = None

    def train(
        self,
        env_factory: Callable[[], list[Any]],
        eval_env_factory: Callable[[], list[Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError as exc:
            raise RuntimeError(
                "MaskablePPO training requires sb3-contrib. "
                "Install project requirements in the minimind environment first."
            ) from exc

        cfg = self.config
        model_dir = Path(cfg.model_dir)
        logs_dir = Path(cfg.logs_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        train_env = self._make_vec_env(env_factory(), DummyVecEnv)
        eval_vec_env = None
        callback = None
        if eval_env_factory is not None:
            eval_vec_env = self._make_vec_env(eval_env_factory(), DummyVecEnv)
            eval_freq = max(1, cfg.eval_interval * cfg.n_steps)
            callback = MaskableEvalCallback(
                eval_vec_env,
                best_model_save_path=str(model_dir),
                log_path=str(logs_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                warn=False,
            )

        _logger.info(
            "MaskablePPO training start timesteps=%d n_steps=%d batch_size=%d",
            cfg.total_timesteps,
            cfg.n_steps,
            cfg.batch_size,
        )

        if cfg.base_model_path and Path(cfg.base_model_path).exists():
            _logger.info("Fine-tuning from base model: %s", cfg.base_model_path)
            self._model = MaskablePPO.load(cfg.base_model_path, env=train_env)
            self._model.learning_rate = cfg.learning_rate
            self._model.n_steps = cfg.n_steps
            self._model.batch_size = cfg.batch_size
            self._model.n_epochs = cfg.n_epochs
        else:
            self._model = MaskablePPO(
                "MlpPolicy",
                train_env,
                learning_rate=cfg.learning_rate,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_epsilon,
                ent_coef=cfg.entropy_coef,
                vf_coef=cfg.value_coef,
                max_grad_norm=cfg.max_grad_norm,
                tensorboard_log=str(logs_dir),
                verbose=cfg.verbose,
                seed=cfg.seed,
            )
        teacher_metrics = self._run_teacher_pretrain(env_factory) if cfg.teacher_pretrain_steps > 0 else {}
        self._model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

        final_path = model_dir / "policy_maskable_final"
        self._model.save(str(final_path))

        eval_envs = eval_env_factory() if eval_env_factory is not None else env_factory()
        final_income = self._evaluate_envs(eval_envs)
        best_income = final_income
        best_path = model_dir / "policy_maskable_best"
        self._model.save(str(best_path))

        if train_env is not None:
            train_env.close()
        if eval_vec_env is not None:
            eval_vec_env.close()

        stats = {
            "episode_rewards": [],
            "episode_incomes": [final_income],
            "eval_incomes": [final_income],
            "best_income": best_income,
            "model_dir": str(model_dir),
            "algo": "maskable_ppo",
            "total_timesteps": cfg.total_timesteps,
            "final_model_path": str(final_path.with_suffix(".zip")),
            "best_model_path": str(best_path.with_suffix(".zip")),
            "teacher_metrics": teacher_metrics,
        }
        _logger.info("MaskablePPO training done final_income=%.0f", final_income)
        return stats

    @staticmethod
    def _make_vec_env(envs: list[Any], dummy_vec_env_cls: Any) -> Any:
        if not envs:
            raise ValueError("env_factory returned no environments")
        return dummy_vec_env_cls([lambda env=env: env for env in envs])

    def _run_teacher_pretrain(self, env_factory: Callable[[], list[Any]]) -> dict[str, float]:
        from agent.heuristic_teacher import collect_teacher_samples, pretrain_maskable_policy

        cfg = self.config
        envs = env_factory()
        samples = collect_teacher_samples(
            envs,
            max_steps=cfg.teacher_pretrain_steps,
            min_cargo_score=cfg.teacher_min_cargo_score,
        )
        metrics = pretrain_maskable_policy(
            self._model,
            samples,
            epochs=max(1, cfg.teacher_pretrain_epochs),
            batch_size=cfg.teacher_batch_size,
            learning_rate=cfg.teacher_learning_rate,
        )
        _logger.info(
            "teacher pretrain samples=%d loss=%.4f",
            int(metrics.get("teacher_samples", 0.0)),
            metrics.get("teacher_loss", 0.0),
        )
        return metrics

    def _evaluate_envs(self, envs: list[Any], max_steps: int = 2000) -> float:
        if not envs:
            return 0.0
        incomes = [self._evaluate(env, max_steps=max_steps) for env in envs]
        return float(np.mean(incomes))

    def _evaluate(self, env: Any, max_steps: int = 2000) -> float:
        from sb3_contrib.common.maskable.utils import get_action_masks

        obs, _ = env.reset()
        income = 0.0
        done = False
        for _ in range(max_steps):
            if done:
                break
            action_masks = get_action_masks(env)
            action, _ = self._model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if info:
                income = float(info.get("net_income", info.get("total_income", income)) or income)
        return income
