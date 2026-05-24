"""RL training entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

_logger = logging.getLogger("agent.train")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load rl_config.yaml."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "configs" / "rl_config.yaml")
    path = Path(config_path)
    if not path.exists():
        _logger.warning("config not found: %s, using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_value(config: dict[str, Any], *paths: str, default: Any = None) -> Any:
    """Read the first present value from nested config paths."""
    for path in paths:
        current: Any = config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return default


def resolve_project_path(path_value: str) -> str:
    """Resolve config paths from the repository root, independent of cwd."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((DEMO_DIR.parent / path).resolve())


def json_safe(value: Any) -> Any:
    """Convert common numeric/path objects into JSON-serializable values."""
    try:
        import numpy as np
    except Exception:
        np = None

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def summarize_training_stats(stats: dict[str, Any]) -> dict[str, Any]:
    incomes = [float(x) for x in stats.get("episode_incomes", [])]
    rewards = [float(x) for x in stats.get("episode_rewards", [])]
    summary = {
        "episode_count": len(incomes),
        "best_income": float(stats.get("best_income", 0.0) or 0.0),
        "final_income": incomes[-1] if incomes else None,
        "max_income": max(incomes) if incomes else None,
        "min_income": min(incomes) if incomes else None,
        "mean_income": (sum(incomes) / len(incomes)) if incomes else None,
        "final_reward": rewards[-1] if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "min_reward": min(rewards) if rewards else None,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
    }
    if incomes:
        best_idx = max(range(len(incomes)), key=lambda i: incomes[i])
        summary["best_episode"] = best_idx + 1
    return summary


def write_training_summary(
    *,
    phase: int,
    stats: dict[str, Any],
    config: dict[str, Any],
    ppo_config: Any,
) -> Path:
    logs_dir = Path(resolve_project_path(
        get_config_value(config, "logs_dir", "paths.logs_dir", default="demo/results/rl_logs")
    ))
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "phase": phase,
        "timestamp": timestamp,
        "summary": summarize_training_stats(stats),
        "stats": stats,
        "ppo_config": ppo_config,
        "config": config,
    }
    payload = json_safe(payload)

    path = logs_dir / f"rl_train_phase{phase}_{timestamp}.json"
    latest_path = logs_dir / f"latest_phase{phase}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    _logger.info("Training summary written to %s", path)
    return path


def build_ppo_config(config: dict[str, Any], phase: int) -> Any:
    """Map rl_config.yaml into PPOConfig."""
    from agent.rl_trainer import PPOConfig

    model_dir = resolve_project_path(get_config_value(
        config, "model_dir", "paths.models_dir", default="demo/agent/models"
    ))
    total_episodes = get_config_value(
        config, "total_episodes", "training.ppo_episodes", default=500
    )
    if phase == 1 and "total_episodes" not in config:
        total_episodes = get_config_value(
            config, "training.bc_episodes", default=50
        )
    elif phase == 3 and "total_episodes" not in config:
        total_episodes = get_config_value(
            config, "training.finetune_episodes", default=200
        )

    return PPOConfig(
        learning_rate=get_config_value(
            config, "learning_rate", "ppo.learning_rate", default=3e-4
        ),
        gamma=get_config_value(config, "gamma", "ppo.gamma", default=0.995),
        gae_lambda=get_config_value(
            config, "gae_lambda", "ppo.gae_lambda", default=0.95
        ),
        clip_epsilon=get_config_value(
            config, "clip_epsilon", "ppo.clip_epsilon", default=0.2
        ),
        entropy_coef=get_config_value(
            config, "entropy_coef", "ppo.entropy_coef", default=0.01
        ),
        value_coef=get_config_value(
            config, "value_coef", "ppo.value_coef", default=0.5
        ),
        max_grad_norm=get_config_value(
            config, "max_grad_norm", "ppo.max_grad_norm", default=0.5
        ),
        batch_size=get_config_value(
            config, "batch_size", "ppo.batch_size", default=64
        ),
        n_epochs=get_config_value(config, "n_epochs", "ppo.n_epochs", default=10),
        total_episodes=total_episodes,
        eval_interval=get_config_value(
            config, "eval_interval", "training.eval_interval", default=50
        ),
        save_interval=get_config_value(
            config, "save_interval", "training.save_interval", default=100
        ),
        model_dir=model_dir,
        curriculum_start_days=get_config_value(
            config,
            "curriculum_start_days",
            "training.curriculum.start_days",
            default=5,
        ),
        curriculum_end_days=get_config_value(
            config,
            "curriculum_end_days",
            "training.curriculum.max_days",
            default=31,
        ),
        curriculum_ramp_episodes=get_config_value(
            config,
            "curriculum_ramp_episodes",
            "training.curriculum.increment_every",
            default=200,
        ),
    )


def build_maskable_ppo_config(config: dict[str, Any], phase: int, base_model_path: str | None = None) -> Any:
    """Map rl_config.yaml into MaskablePPOConfig."""
    from agent.sb3_maskable_trainer import MaskablePPOConfig

    model_dir = resolve_project_path(get_config_value(
        config, "model_dir", "paths.models_dir", default="demo/agent/models"
    ))
    logs_dir = resolve_project_path(get_config_value(
        config, "logs_dir", "paths.logs_dir", default="demo/results/rl_logs"
    ))
    n_steps = int(get_config_value(config, "n_steps", "ppo.n_steps", default=2048))
    total_timesteps = get_config_value(
        config,
        "total_timesteps",
        "maskable_ppo.total_timesteps",
        default=None,
    )
    if total_timesteps is None:
        episodes = int(get_config_value(
            config, "total_episodes", "training.ppo_episodes", default=1
        ))
        if phase == 1 and "total_episodes" not in config:
            episodes = int(get_config_value(config, "training.bc_episodes", default=1))
        elif phase == 3 and "total_episodes" not in config:
            episodes = int(get_config_value(config, "training.finetune_episodes", default=1))
        total_timesteps = max(1, episodes) * n_steps

    return MaskablePPOConfig(
        learning_rate=get_config_value(
            config, "learning_rate", "ppo.learning_rate", default=3e-4
        ),
        gamma=get_config_value(config, "gamma", "ppo.gamma", default=0.995),
        gae_lambda=get_config_value(
            config, "gae_lambda", "ppo.gae_lambda", default=0.95
        ),
        clip_epsilon=get_config_value(
            config, "clip_epsilon", "ppo.clip_epsilon", default=0.2
        ),
        entropy_coef=get_config_value(
            config, "entropy_coef", "ppo.entropy_coef", default=0.01
        ),
        value_coef=get_config_value(
            config, "value_coef", "ppo.value_coef", default=0.5
        ),
        max_grad_norm=get_config_value(
            config, "max_grad_norm", "ppo.max_grad_norm", default=0.5
        ),
        batch_size=int(get_config_value(
            config, "batch_size", "ppo.batch_size", default=256
        )),
        n_epochs=int(get_config_value(config, "n_epochs", "ppo.n_epochs", default=10)),
        n_steps=n_steps,
        total_timesteps=int(total_timesteps),
        eval_interval=int(get_config_value(
            config, "eval_interval", "training.eval_interval", default=50
        )),
        save_interval=int(get_config_value(
            config, "save_interval", "training.save_interval", default=100
        )),
        model_dir=model_dir,
        logs_dir=logs_dir,
        seed=get_config_value(config, "seed", "maskable_ppo.seed", default=None),
        verbose=int(get_config_value(config, "verbose", "maskable_ppo.verbose", default=1)),
        teacher_pretrain_steps=int(get_config_value(
            config, "teacher_pretrain_steps", "maskable_ppo.teacher_pretrain_steps", default=0
        )),
        teacher_pretrain_epochs=int(get_config_value(
            config, "teacher_pretrain_epochs", "maskable_ppo.teacher_pretrain_epochs", default=0
        )),
        teacher_batch_size=int(get_config_value(
            config, "teacher_batch_size", "maskable_ppo.teacher_batch_size", default=64
        )),
        teacher_learning_rate=float(get_config_value(
            config, "teacher_learning_rate", "maskable_ppo.teacher_learning_rate", default=1e-4
        )),
        teacher_min_cargo_score=float(get_config_value(
            config, "teacher_min_cargo_score", "maskable_ppo.teacher_min_cargo_score", default=0.0
        )),
        base_model_path=base_model_path,
    )


def _extract_driver_config(
    profile: dict[str, Any],
    status: dict[str, Any],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    current_lat = float(status.get("current_lat", profile.get("current_lat", 22.54)))
    current_lng = float(status.get("current_lng", profile.get("current_lng", 114.06)))
    config = {
        "driver_id": status.get("driver_id", profile.get("driver_id", "")),
        "cost_per_km": float(profile.get("cost_per_km", 1.5)),
        "current_lat": current_lat,
        "current_lng": current_lng,
        "preferences": status.get("preferences", profile.get("preferences", [])),
        "constraints": constraints,
        "home": {"lat": current_lat, "lng": current_lng},
    }

    for constraint in constraints:
        params = constraint.get("params", {})
        if constraint.get("type") in ("daily_home_deadline", "scheduled_event"):
            home_lat = params.get("home_lat") or params.get("lat")
            home_lng = params.get("home_lng") or params.get("lng")
            if home_lat is not None and home_lng is not None:
                config["home"] = {"lat": float(home_lat), "lng": float(home_lng)}
        elif constraint.get("type") == "mileage_cap":
            max_km = params.get("max_deadhead_km")
            if max_km is not None:
                config["max_deadhead_km"] = float(max_km)
        elif constraint.get("type") == "daily_rest":
            rest_min = params.get("min_rest_minutes") or params.get("required_rest_min")
            if rest_min is not None:
                config["required_rest_min"] = int(rest_min)

    return config


def _deterministic_constraints(decision_service: Any, driver_id: str, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engine = getattr(decision_service, "_preference_engine", None)
    parser = getattr(engine, "_deterministic_parse", None)
    if callable(parser):
        return parser(preferences)
    fallback = getattr(engine, "_fallback_parse", None)
    if callable(fallback):
        return fallback(preferences)
    return []


def _build_training_candidate_ranker(decision_service: Any):
    from agent.rl_env import _TOP_K

    heuristic = getattr(decision_service, "_heuristic_layer", None)
    profit_search = getattr(decision_service, "profit_search_layer", None)
    time_window = getattr(decision_service, "_time_window_optimizer", None)
    coordination = getattr(decision_service, "coordination_layer", None)
    opportunity = getattr(decision_service, "opportunity_predictor", None)
    sync_tasks = getattr(decision_service, "_sync_constraint_tasks", None)

    def rank(
        items: list[dict[str, Any]],
        status: dict[str, Any],
        state: Any,
        constraints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked_items = items
        if time_window is not None:
            ranked_items = time_window.prescreen_cargo_by_feasibility(
                ranked_items, status, constraints
            )
        if coordination is not None:
            ranked_items = coordination.filter_competitive_cargo(
                getattr(state, "driver_id", ""), ranked_items
            )
        if callable(getattr(state, "record_cargo_seen", None)):
            state.record_cargo_seen(
                int(status.get("simulation_progress_minutes", 0)), len(ranked_items)
            )
        if callable(sync_tasks):
            sync_tasks(
                state,
                constraints,
                int(status.get("simulation_horizon_minutes", 43200)),
            )
        if heuristic is None:
            return ranked_items[:_TOP_K]
        candidates = heuristic.score_and_rank(
            ranked_items, status, state, constraints, top_n=max(12, _TOP_K)
        )
        if profit_search is not None:
            candidates = profit_search.rank_candidates(
                candidates, status, state, constraints, opportunity
            )
        return candidates[:_TOP_K]

    return rank


def _clone_loaded_cargo_repository(base_repo: Any) -> Any:
    """Create an independent simulation repository from an already parsed dataset."""
    repo = base_repo.__class__(base_repo._path, base_repo._earth_radius_km)
    repo._pending = list(base_repo._pending)
    repo._pending_cursor = base_repo._pending_cursor
    repo._online = dict(base_repo._online)
    repo._online_expire_heap = list(base_repo._online_expire_heap)
    repo._online_ids = list(base_repo._online_ids)
    repo._online_lat = base_repo._online_lat.copy()
    repo._online_lng = base_repo._online_lng.copy()
    repo._online_dirty = base_repo._online_dirty
    repo._simulation_start_dt = base_repo._simulation_start_dt
    repo._current_time_minutes = base_repo._current_time_minutes
    return repo


def _replace_maskable_outputs_with_base(model_dir: str, base_model_path: str) -> None:
    """Replace a rejected fine-tuned model with the known base model."""
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("policy_maskable_best.zip", "policy_maskable_final.zip"):
        shutil.copy2(base_model_path, target_dir / name)


def build_envs(
    config: dict[str, Any],
    driver_ids: list[str] | None = None,
) -> list[Any]:
    """Build one RL environment per driver."""
    from simkit.cargo_repository import CargoRepository
    from simkit.driver_state_manager import DriverStateManager
    from simkit.ports import SimulationApiPort
    from server.bench.embedded_agent import EmbeddedDecisionEnvironment
    from server.bench.model_gateway_client import ModelGatewayClient
    from server.bench.settings import load_settings
    from agent.model_decision_service import ModelDecisionService
    from agent.rl_env import DriverRLEnv

    settings = load_settings()
    if driver_ids is None:
        manager = DriverStateManager(settings.drivers_path)
        manager.load()
        driver_ids = manager.list_driver_ids()

    model_gateway = ModelGatewayClient(
        api_url=settings.model_api_url,
        api_key=settings.model_api_key,
        default_model_name=settings.model_name,
        timeout_seconds=settings.model_timeout_seconds,
    )

    base_repo = CargoRepository(settings.cargo_dataset_path)
    base_repo.load()

    envs = []
    for driver_id in driver_ids:
        repo = _clone_loaded_cargo_repository(base_repo)
        manager = DriverStateManager(settings.drivers_path)
        manager.load()
        session_actions: dict[str, list[dict[str, Any]]] = {driver_id: []}
        env_api = EmbeddedDecisionEnvironment(
            repo=repo,
            manager=manager,
            model_gateway=model_gateway,
            session_actions_by_driver=session_actions,
            simulation_duration_days=settings.simulation_duration_days,
        )
        api_port: SimulationApiPort = env_api
        decision_service = ModelDecisionService(api_port, enable_rl_layer=False)
        manager.start_simulation(driver_id, progress_minutes=0)
        driver_status = manager.get_driver_status(driver_id)
        driver_profile = getattr(manager, "_drivers", {}).get(driver_id, {})
        constraints = _deterministic_constraints(
            decision_service, driver_id, driver_status.get("preferences", [])
        )
        driver_config = _extract_driver_config(driver_profile, driver_status, constraints)
        env = DriverRLEnv(
            api=api_port,
            driver_id=driver_id,
            driver_config=driver_config,
            decision_service=decision_service,
            candidate_ranker=_build_training_candidate_ranker(decision_service),
            constraints=constraints,
            max_steps=get_config_value(config, "env.max_steps", default=2000),
        )
        envs.append(env)

    return envs


def train_phase_1(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 1 placeholder: runs short PPO warm-up."""
    _logger.info("Phase 1: warm-up training")
    from agent.rl_trainer import PPOTrainer

    cfg = build_ppo_config(config, phase=1)
    cfg.eval_interval = min(cfg.eval_interval, 10)
    cfg.save_interval = min(cfg.save_interval, 25)
    cfg.curriculum_start_days = 31
    cfg.curriculum_end_days = 31
    cfg.curriculum_ramp_episodes = 0
    stats = PPOTrainer(cfg).train(
        env_factory=lambda: build_envs(config),
        eval_env_factory=lambda: build_envs(config),
    )
    write_training_summary(phase=1, stats=stats, config=config, ppo_config=cfg)
    return stats


def train_phase_2(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 2: PPO training."""
    _logger.info("Phase 2: PPO training")
    from agent.rl_trainer import PPOTrainer

    cfg = build_ppo_config(config, phase=2)
    stats = PPOTrainer(cfg).train(
        env_factory=lambda: build_envs(config),
        eval_env_factory=lambda: build_envs(config),
    )
    write_training_summary(phase=2, stats=stats, config=config, ppo_config=cfg)
    return stats


def train_maskable_ppo(config: dict[str, Any], phase: int, base_model_path: str | None = None) -> dict[str, Any]:
    """MaskablePPO training for the requested phase."""
    _logger.info("Phase %d: MaskablePPO training", phase)
    from agent.sb3_maskable_trainer import MaskablePPOTrainer

    if phase == 3:
        model_dir = resolve_project_path(get_config_value(
            config, "model_dir", "paths.models_dir", default="demo/agent/models"
        ))
        # Resolve base model for fine-tuning
        if base_model_path is None:
            base_model_path = get_config_value(
                config, "base_model", "training.finetune.base_model", default=None,
            )
        if base_model_path is not None:
            base_model_path = str(resolve_project_path(base_model_path))
            if not Path(base_model_path).exists():
                _logger.warning("Base model not found: %s, training from scratch", base_model_path)
                base_model_path = None

        regression_threshold = float(get_config_value(
            config, "regression_threshold", "training.finetune.regression_threshold", default=0.95,
        ))

        lr_factor = float(get_config_value(
            config, "learning_rate_factor", "training.finetune.learning_rate_factor", default=0.5,
        ))

        results = {}
        for driver_id in ["D001", "D002", "D003", "D005", "D006", "D007", "D008", "D010"]:
            cfg = build_maskable_ppo_config(config, phase=3, base_model_path=base_model_path)
            cfg.learning_rate *= lr_factor
            cfg.model_dir = f"{model_dir}/{driver_id}/maskable"
            stats = MaskablePPOTrainer(cfg).train(
                env_factory=lambda driver_id=driver_id: build_envs(config, driver_ids=[driver_id]),
                eval_env_factory=lambda driver_id=driver_id: build_envs(config, driver_ids=[driver_id]),
            )

            # Regression check: compare fine-tuned vs base model
            if base_model_path and Path(base_model_path).exists():
                ft_income = stats.get("best_income", 0.0)
                base_cfg = build_maskable_ppo_config(config, phase=2)
                base_cfg.model_dir = str(Path(base_model_path).parent)
                base_envs = build_envs(config, driver_ids=[driver_id])
                base_trainer = MaskablePPOTrainer(base_cfg)
                try:
                    from sb3_contrib import MaskablePPO
                    base_trainer._model = MaskablePPO.load(base_model_path)
                    base_income = base_trainer._evaluate(base_envs[0]) if base_envs else 0.0
                    regression_error = ""
                except Exception as exc:
                    base_income = 0.0
                    regression_error = str(exc)

                stats["base_income"] = base_income
                if regression_error:
                    stats["regression_check_error"] = regression_error

                discard_regressed = (
                    (base_income > 0 and ft_income < base_income * regression_threshold)
                    or (base_income <= 0 and ft_income <= 0)
                )
                if discard_regressed:
                    _logger.warning(
                        "Driver %s regression: ft_income=%.0f base_income=%.0f threshold=%.2f, replacing fine-tuned model with base",
                        driver_id, ft_income, base_income, regression_threshold,
                    )
                    _replace_maskable_outputs_with_base(cfg.model_dir, base_model_path)
                    stats["regression_discarded"] = True
                    stats["discard_replacement_model"] = base_model_path
                    stats["best_model_path"] = str(Path(cfg.model_dir) / "policy_maskable_best.zip")
                    stats["final_model_path"] = str(Path(cfg.model_dir) / "policy_maskable_final.zip")
                else:
                    _logger.info(
                        "Driver %s fine-tune OK: ft_income=%.0f >= base_income=%.0f * %.2f",
                        driver_id, ft_income, base_income, regression_threshold,
                    )
                    stats["regression_discarded"] = False

            write_training_summary(
                phase=3,
                stats=stats,
                config={**config, "driver_id": driver_id, "algo": "maskable_ppo"},
                ppo_config=cfg,
            )
            results[driver_id] = stats
        return results

    cfg = build_maskable_ppo_config(config, phase=phase, base_model_path=base_model_path)
    if phase == 1:
        cfg.eval_interval = min(cfg.eval_interval, 10)
        cfg.save_interval = min(cfg.save_interval, 25)
    stats = MaskablePPOTrainer(cfg).train(
        env_factory=lambda: build_envs(config),
        eval_env_factory=lambda: build_envs(config),
    )
    write_training_summary(
        phase=phase,
        stats=stats,
        config={**config, "algo": "maskable_ppo"},
        ppo_config=cfg,
    )
    return stats


def train_phase_3(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 3: per-driver fine-tuning."""
    _logger.info("Phase 3: per-driver fine-tuning")
    from agent.rl_trainer import PPOTrainer

    model_dir = resolve_project_path(get_config_value(
        config, "model_dir", "paths.models_dir", default="demo/agent/models"
    ))
    results = {}
    for driver_id in ["D001", "D002", "D003", "D005", "D006", "D007", "D008", "D010"]:
        cfg = build_ppo_config(config, phase=3)
        cfg.learning_rate *= 0.5
        cfg.model_dir = f"{model_dir}/{driver_id}"
        cfg.curriculum_start_days = 31
        cfg.curriculum_end_days = 31
        cfg.curriculum_ramp_episodes = 0
        stats = PPOTrainer(cfg).train(
            env_factory=lambda driver_id=driver_id: build_envs(config, driver_ids=[driver_id]),
            eval_env_factory=lambda driver_id=driver_id: build_envs(config, driver_ids=[driver_id]),
        )
        write_training_summary(
            phase=3,
            stats=stats,
            config={**config, "driver_id": driver_id},
            ppo_config=cfg,
        )
        results[driver_id] = stats
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="RL training for driver optimization")
    parser.add_argument("--config", type=str, default=None, help="Path to rl_config.yaml")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Training phase: 1=warm-up, 2=PPO, 3=fine-tune",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "maskable_ppo"],
        default="ppo",
        help="Training algorithm",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes")
    parser.add_argument("--base-model", type=str, default=None, help="Base model path for fine-tuning (.zip)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config["total_episodes"] = args.episodes

    _logger.info("Starting training phase=%d algo=%s", args.phase, args.algo)
    if args.algo == "maskable_ppo":
        stats = train_maskable_ppo(config, phase=args.phase, base_model_path=args.base_model)
    elif args.phase == 1:
        stats = train_phase_1(config)
    elif args.phase == 2:
        stats = train_phase_2(config)
    else:
        stats = train_phase_3(config)

    _logger.info(
        "Training complete: %s",
        json.dumps(
            {k: v for k, v in stats.items() if k != "episode_rewards"},
            default=str,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
