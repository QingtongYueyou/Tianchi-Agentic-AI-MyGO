import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEMO_DIR)

from agent.rl_env import ACTION_CARGO_0, ACTION_WAIT_30, ACTION_WAIT_60, DriverRLEnv
from agent.rl_integration import RLDecisionLayer
from agent.rl_trainer import PPOConfig, PPOTrainer, Trajectory, TrajectoryStep
from agent.train import build_ppo_config, write_training_summary, _clone_loaded_cargo_repository
from simkit.cargo_repository import CargoRepository


def test_nested_rl_config_is_mapped_to_ppo_config():
    cfg = build_ppo_config(
        {
            "ppo": {
                "learning_rate": 1e-4,
                "gamma": 0.9,
                "batch_size": 128,
            },
            "training": {
                "ppo_episodes": 7,
                "eval_interval": 3,
                "save_interval": 4,
                "curriculum": {
                    "start_days": 2,
                    "max_days": 9,
                    "increment_every": 5,
                },
            },
            "paths": {"models_dir": "tmp/models"},
        },
        phase=2,
    )

    assert cfg.learning_rate == 1e-4
    assert cfg.gamma == 0.9
    assert cfg.batch_size == 128
    assert cfg.total_episodes == 7
    assert cfg.eval_interval == 3
    assert cfg.save_interval == 4
    assert Path(cfg.model_dir) == Path(PROJECT_ROOT) / "tmp" / "models"
    assert cfg.curriculum_start_days == 2
    assert cfg.curriculum_end_days == 9
    assert cfg.curriculum_ramp_episodes == 5


def test_loaded_cargo_repository_clone_has_independent_online_pool(tmp_path):
    dataset = tmp_path / "cargo.jsonl"
    dataset.write_text(
        '{"cargo_id":"C1","create_time":"2026-03-01 00:00:00","remove_time":"2026-03-01 01:00:00",'
        '"start":{"lat":22.5,"lng":114.0},"end":{"lat":22.6,"lng":114.1},"price":1000,'
        '"cost_time_minutes":30}\n',
        encoding="utf-8",
    )
    base = CargoRepository(dataset)
    base.load()

    clone = _clone_loaded_cargo_repository(base)
    assert clone.remove_by_id("C1") is not None

    assert base.get_by_id("C1") is not None


def test_mock_env_tracks_continuous_rest_for_reward_shaping():
    env = DriverRLEnv(driver_id="D008")
    env.reset(seed=1)

    env.step(ACTION_WAIT_60)
    env.step(ACTION_WAIT_60)

    assert env.get_current_rest_streak(120) == 120
    assert env.get_max_continuous_rest_today(120) == 120


def test_fallback_wait_updates_rest_state():
    env = DriverRLEnv(driver_id="D008")
    env.reset(seed=1)
    env._current_cargo_list = []

    env.step(ACTION_CARGO_0)

    assert env.get_current_rest_streak(30) == 30
    assert env.get_max_continuous_rest_today(30) == 30


def test_env_action_mask_blocks_missing_cargo_and_reposition_actions():
    env = DriverRLEnv(driver_id="D001")
    env.reset(seed=1)
    env._current_cargo_list = []
    env._spatial_income = {}

    mask = env.get_action_mask()

    assert bool(mask[ACTION_WAIT_30])
    assert bool(mask[ACTION_WAIT_60])
    assert not mask[ACTION_CARGO_0]
    assert not mask[7]
    assert not mask[8]


class _TwoCargoApi:
    def __init__(self):
        self.progress = 0

    def get_driver_status(self, driver_id):
        return {
            "driver_id": driver_id,
            "current_lat": 22.54,
            "current_lng": 114.06,
            "simulation_progress_minutes": self.progress,
            "simulation_horizon_minutes": 60,
            "preferences": [],
        }

    def query_cargo(self, driver_id, latitude, longitude):
        return {
            "items": [
                {
                    "distance_km": 1.0,
                    "cargo": {
                        "cargo_id": "C1",
                        "price": 1000.0,
                        "cost_time_minutes": 60,
                        "start": {"lat": latitude, "lng": longitude},
                        "end": {"lat": latitude, "lng": longitude},
                    },
                },
                {
                    "distance_km": 1.0,
                    "cargo": {
                        "cargo_id": "C2",
                        "price": 2000.0,
                        "cost_time_minutes": 60,
                        "start": {"lat": latitude, "lng": longitude},
                        "end": {"lat": latitude, "lng": longitude},
                    },
                },
            ]
        }

    def wait(self, driver_id, duration_minutes):
        self.progress += duration_minutes
        return {"simulation_progress_minutes": self.progress}

    def take_order(self, driver_id, cargo_id):
        self.progress += 60
        return {
            "accepted": True,
            "revenue": 1000.0,
            "pickup_deadhead_km": 1.0,
            "haul_distance_km": 1.0,
            "simulation_progress_minutes": self.progress,
        }

    def reposition(self, driver_id, lat, lng):
        return {"simulation_progress_minutes": self.progress}


def test_env_uses_candidate_ranker_for_training_top5():
    def ranker(items, status, state, constraints):
        return list(reversed(items))

    env = DriverRLEnv(
        api=_TwoCargoApi(),
        driver_id="D001",
        candidate_ranker=ranker,
    )
    env.reset()

    assert env._current_cargo_list[0]["cargo"]["cargo_id"] == "C2"


class _IneligibleCargoApi:
    def __init__(self):
        self.progress = 0

    def get_driver_status(self, driver_id):
        return {
            "driver_id": driver_id,
            "current_lat": 22.54,
            "current_lng": 114.06,
            "simulation_progress_minutes": self.progress,
            "simulation_horizon_minutes": 60,
            "preferences": [],
        }

    def query_cargo(self, driver_id, latitude, longitude):
        return {
            "items": [{
                "distance_km": 0.0,
                "cargo": {
                    "cargo_id": "C1",
                    "price": 5000.0,
                    "cost_time_minutes": 120,
                    "start": {"lat": latitude, "lng": longitude},
                    "end": {"lat": latitude, "lng": longitude},
                },
            }]
        }

    def take_order(self, driver_id, cargo_id):
        self.progress = 120
        return {
            "accepted": True,
            "income_eligible": False,
            "revenue": 5000.0,
            "pickup_deadhead_km": 0.0,
            "haul_distance_km": 0.0,
            "simulation_progress_minutes": self.progress,
        }

    def wait(self, driver_id, duration_minutes):
        self.progress += duration_minutes
        return {"simulation_progress_minutes": self.progress}

    def reposition(self, driver_id, lat, lng):
        return {"simulation_progress_minutes": self.progress}


def test_ineligible_order_does_not_count_as_training_income():
    env = DriverRLEnv(api=_IneligibleCargoApi(), driver_id="D001")
    env.reset()

    _, reward, terminated, _, info = env.step(ACTION_CARGO_0)

    assert terminated is True
    assert info["total_income"] == 0.0
    assert reward <= 0.0


def test_trainer_uses_total_income_from_env_info():
    trainer = PPOTrainer(PPOConfig())
    step = TrajectoryStep(
        state=[],
        action=0,
        reward=1.0,
        log_prob=0.0,
        value=0.0,
        done=True,
    )

    returns = trainer._compute_returns([step])

    assert returns == [1.0]


class _WaitOnlyEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return np.zeros(49, dtype=np.float32), {}

    def get_action_mask(self):
        mask = np.zeros(9, dtype=np.bool_)
        mask[ACTION_WAIT_30] = True
        return mask

    def step(self, action):
        self.actions.append(action)
        info = {
            "total_income": 0.0,
            "net_income": 0.0,
            "action_mask": self.get_action_mask(),
        }
        return np.zeros(49, dtype=np.float32), 0.0, True, False, info


class _InvalidActionPolicy:
    def __init__(self):
        import torch

        self.device = torch.device("cpu")

    def __call__(self, state):
        import torch

        logits = torch.zeros((1, 9), dtype=torch.float32)
        logits[0, 6] = 100.0
        value = torch.zeros((1, 1), dtype=torch.float32)
        return logits, value


def test_trainer_sampling_respects_action_mask():
    env = _WaitOnlyEnv()
    trainer = PPOTrainer(PPOConfig())
    trainer._policy_net = _InvalidActionPolicy()

    trainer._collect_trajectory(env, max_days=1)

    assert env.actions == [ACTION_WAIT_30]


def test_training_summary_is_written(tmp_path):
    stats = {
        "episode_rewards": [1.0, 2.0],
        "episode_incomes": [100.0, 150.0],
        "eval_incomes": [],
        "best_income": 150.0,
        "model_dir": str(tmp_path / "models"),
    }
    cfg = PPOConfig(total_episodes=2, model_dir=str(tmp_path / "models"))

    path = write_training_summary(
        phase=2,
        stats=stats,
        config={"paths": {"logs_dir": str(tmp_path / "logs")}},
        ppo_config=cfg,
    )

    latest = tmp_path / "logs" / "latest_phase2.json"
    assert path.exists()
    assert latest.exists()

    import json

    data = json.loads(latest.read_text(encoding="utf-8"))
    assert data["summary"]["episode_count"] == 2
    assert data["summary"]["best_episode"] == 2
    assert data["summary"]["final_income"] == 150.0
    assert data["stats"]["episode_incomes"] == [100.0, 150.0]


class _FakePolicy:
    def forward(self, state):
        probs = np.zeros(9, dtype=np.float32)
        probs[2] = 0.05
        probs[3] = 0.90
        probs[4:] = 0.05 / 5
        return probs, 0.25


class _WaitPolicy:
    def forward(self, state):
        probs = np.zeros(9, dtype=np.float32)
        probs[0] = 0.72
        probs[1] = 0.08
        probs[2] = 0.10
        probs[3] = 0.10
        return probs, -0.1


def _candidate(cargo_id, score):
    return {
        "cargo_id": cargo_id,
        "cargo_name": "test",
        "price": 1000.0,
        "net_profit": 100.0,
        "deadhead_km": 10.0,
        "haul_km": 50.0,
        "total_minutes": 120,
        "cost_time_minutes": 100,
        "score": score,
        "true_net": score,
        "start": {"lat": 22.5, "lng": 114.0},
        "end": {"lat": 23.0, "lng": 114.5},
    }


def test_rl_policy_probability_changes_candidate_ranking():
    layer = RLDecisionLayer()
    layer._loaded = True
    layer._policy = _FakePolicy()
    candidates = [_candidate("C1", 100.0), _candidate("C2", 100.0)]

    ranked = layer.rank_candidates(
        candidates,
        status={"current_lat": 22.5, "current_lng": 114.0, "simulation_progress_minutes": 0},
        state_tracker=None,
        constraints=[],
    )

    assert ranked[0]["cargo_id"] == "C2"
    assert ranked[0]["rl_policy_prob"] > 0.89


def test_rl_layer_can_directly_choose_wait_when_policy_dominates():
    layer = RLDecisionLayer()
    layer._loaded = True
    layer._policy = _WaitPolicy()

    decision = layer.select_wait_action(
        [_candidate("C1", 100.0), _candidate("C2", 120.0)],
        status={"current_lat": 22.5, "current_lng": 114.0, "simulation_progress_minutes": 0},
        state_tracker=None,
        constraints=[],
    )

    assert decision["action"] == "wait"
    assert decision["params"]["duration_minutes"] == 30


def test_order_scorer_features_match_network_dimension():
    features = RLDecisionLayer._extract_cargo_features(_candidate("C1", 100.0), None)

    assert features.shape == (8,)


def test_trainer_rebuilds_envs_from_factory_each_episode(tmp_path, monkeypatch):
    created_envs = []

    def env_factory():
        env = object()
        created_envs.append(env)
        return [env]

    def fake_collect(self, env, max_days):
        episode_no = created_envs.index(env) + 1
        return Trajectory(
            steps=[
                TrajectoryStep(
                    state=np.zeros(49, dtype=np.float32),
                    action=0,
                    reward=float(episode_no),
                    log_prob=0.0,
                    value=0.0,
                    done=True,
                )
            ],
            total_reward=float(episode_no),
            final_income=episode_no * 100.0,
        )

    monkeypatch.setattr(PPOTrainer, "_collect_trajectory", fake_collect)
    monkeypatch.setattr(PPOTrainer, "_update_policy", lambda *args, **kwargs: None)

    cfg = PPOConfig(
        total_episodes=3,
        n_epochs=1,
        save_interval=99,
        model_dir=str(tmp_path / "models"),
    )
    stats = PPOTrainer(cfg).train(env_factory=env_factory)

    assert len(created_envs) == 3
    assert len({id(env) for env in created_envs}) == 3
    assert stats["episode_incomes"] == [100.0, 200.0, 300.0]


def test_best_model_uses_fresh_eval_factory(tmp_path, monkeypatch):
    created_eval_envs = []

    def env_factory():
        return [object()]

    def eval_env_factory():
        env = object()
        created_eval_envs.append(env)
        return [env]

    def fake_collect(self, env, max_days):
        return Trajectory(
            steps=[
                TrajectoryStep(
                    state=np.zeros(49, dtype=np.float32),
                    action=0,
                    reward=1.0,
                    log_prob=0.0,
                    value=0.0,
                    done=True,
                    action_mask=np.ones(9, dtype=np.bool_),
                )
            ],
            total_reward=1.0,
            final_income=10.0,
        )

    monkeypatch.setattr(PPOTrainer, "_collect_trajectory", fake_collect)
    monkeypatch.setattr(PPOTrainer, "_update_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(PPOTrainer, "_evaluate_envs", lambda self, envs, max_days: 999.0)

    cfg = PPOConfig(
        total_episodes=2,
        n_epochs=1,
        eval_interval=50,
        save_interval=99,
        model_dir=str(tmp_path / "models"),
    )
    stats = PPOTrainer(cfg).train(
        env_factory=env_factory,
        eval_env_factory=eval_env_factory,
    )

    assert len(created_eval_envs) == 1
    assert stats["eval_incomes"] == [999.0]
    assert stats["best_income"] == 999.0
