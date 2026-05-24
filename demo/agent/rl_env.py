"""
Gymnasium 强化学习训练环境：将物流仿真封装为标准 RL 接口。

功能说明：
- LogisticsDriverEnv: 单司机 Gymnasium 环境，每个 episode = 一个完整月度仿真（31天/43200分钟）
- MultiDriverEnv: 10个司机的并行环境包装，供批量训练使用
- encode_state: 49维状态向量编码（独立函数，兼容外部调用）

状态空间: 49维浮点向量（时空5 + 累计6 + 约束8 + Top5货源30）
动作空间: 9个离散动作（wait_30, wait_60, cargo_0~4, reposition_home, reposition_hotzone）
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from agent.rl_optimizations import compute_reward_shaping
except Exception:
    compute_reward_shaping = None

# ─────────────────────────────────────────────────────────
# 常量（对外暴露，保持向后兼容）
# ─────────────────────────────────────────────────────────

_STATE_DIM = 49
_ACTION_DIM = 9
_TOP_K = 5
_SIM_DURATION_MIN = 43200  # 30天 = 43200分钟（评测标准）

# 动作定义
ACTION_WAIT_30 = 0
ACTION_WAIT_60 = 1
ACTION_CARGO_0 = 2
ACTION_CARGO_1 = 3
ACTION_CARGO_2 = 4
ACTION_CARGO_3 = 5
ACTION_CARGO_4 = 6
ACTION_REPOSITION_HOME = 7
ACTION_REPOSITION_HOTZONE = 8


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """使用 Haversine 公式计算两点间的距离（公里）。"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _compute_haul_km(cargo: dict) -> float:
    """计算货源装货点到卸货点的运输距离（km）。"""
    start = cargo.get("start", {})
    end = cargo.get("end", {})
    s_lat = start.get("lat")
    s_lng = start.get("lng")
    e_lat = end.get("lat")
    e_lng = end.get("lng")
    if s_lat is not None and s_lng is not None and e_lat is not None and e_lng is not None:
        return _haversine_km(float(s_lat), float(s_lng), float(e_lat), float(e_lng))
    # fallback: 用 cost_time_minutes 近似（60km/h 平均速度）
    cost_min = cargo.get("cost_time_minutes", 0)
    return float(cost_min) * 60.0 / 60.0  # km


# ─────────────────────────────────────────────────────────
# 独立 encode_state 函数（兼容外部 import）
# ─────────────────────────────────────────────────────────

def encode_state(
    status: dict[str, Any],
    state_tracker: Any,
    candidates: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
) -> np.ndarray:
    """将原始状态编码为归一化的 49 维 float 向量。

    兼容接口：供 rl_integration.py 等外部模块调用。
    state_tracker 可以是带属性的对象，也可以是 dict。
    """
    vec = np.zeros(_STATE_DIM, dtype=np.float32)
    sim_min = int(status.get("simulation_progress_minutes", 0))

    # ── 时空特征 (5维) [0..4] ──
    lat = float(status.get("current_lat", 22.0))
    lng = float(status.get("current_lng", 110.0))
    vec[0] = (lat - 22.0) / 4.0
    vec[1] = (lng - 110.0) / 8.0
    vec[2] = (sim_min % 1440) / 1440.0  # time_of_day
    vec[3] = (sim_min // 1440) / 30.0   # day_of_month
    day_idx = sim_min // 1440
    # 2026-03-01 是周日, day_idx=0 → day_of_week=6(周日)
    day_of_week = (day_idx + 6) % 7  # 0=周一
    vec[4] = 1.0 if day_of_week >= 5 else 0.0  # is_weekend

    # ── 累计状态 (6维) [5..10] ──
    def _get_attr(obj, key, default=0.0):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    total_income = float(_get_attr(state_tracker, "total_income", 0.0))
    total_mileage = float(_get_attr(state_tracker, "total_mileage_km", 0.0))
    total_deadhead = float(_get_attr(state_tracker, "total_deadhead_km", 0.0))
    total_orders = float(_get_attr(state_tracker, "total_orders", 0))

    vec[5] = total_income / 100000.0
    vec[6] = total_mileage / 10000.0
    vec[7] = total_deadhead / 5000.0
    vec[8] = total_orders / 10.0  # orders_today approx

    # rest_today_minutes
    if state_tracker is not None:
        get_rest = getattr(state_tracker, "get_max_continuous_rest_today", None)
        if callable(get_rest):
            rest_today = get_rest(sim_min)
        else:
            rest_today = float(_get_attr(state_tracker, "rest_today_min", 0.0))
    else:
        rest_today = 0.0
    vec[9] = rest_today / 480.0

    # consecutive_rest
    if state_tracker is not None:
        get_streak = getattr(state_tracker, "get_current_rest_streak", None)
        if callable(get_streak):
            consec_rest = get_streak(sim_min)
        else:
            consec_rest = float(_get_attr(state_tracker, "consecutive_rest", 0.0))
    else:
        consec_rest = 0.0
    vec[10] = consec_rest / 480.0

    # ── 约束状态 (8维) [11..18] ──
    penalty_so_far = float(_get_attr(state_tracker, "total_penalty", 0.0))
    home_dist = 0.0
    visit_dist = 0.0
    urgent_countdown = 1.0
    rest_deficit = 0.0
    deadhead_budget_ratio = 0.0
    mandatory_pending = 0
    scheduled_pending = 0

    # 从 constraints 和 state_tracker 中提取
    open_tasks = []
    if state_tracker is not None:
        open_tasks = getattr(state_tracker, "open_tasks", [])
        if isinstance(state_tracker, dict):
            open_tasks = state_tracker.get("open_tasks", [])

    for task in open_tasks:
        ttype = task.get("type", "")
        if ttype == "mandatory_cargo":
            mandatory_pending += 1
        elif ttype == "scheduled_event":
            scheduled_pending += 1
        elif ttype in ("daily_home_deadline", "home"):
            home_lat = task.get("home_lat") or task.get("lat")
            home_lng = task.get("home_lng") or task.get("lng")
            if home_lat is not None and home_lng is not None:
                home_dist = _haversine_km(lat, lng, float(home_lat), float(home_lng))
        # 紧急任务倒计时
        deadline = task.get("deadline_minute")
        if deadline is not None:
            slack = deadline - sim_min
            if 0 < slack < 1440:
                urgent_countdown = min(urgent_countdown, slack / 1440.0)

    # 休息欠缺
    if rest_today < 240:
        rest_deficit = max(0.0, (240.0 - rest_today)) / 300.0

    # 空驶预算比例
    for c in (constraints or []):
        if c.get("type") == "mileage_cap":
            max_dh = float(c.get("params", {}).get("max_deadhead_km", 5000))
            if max_dh > 0:
                deadhead_budget_ratio = total_deadhead / max_dh

    vec[11] = penalty_so_far / 10000.0
    vec[12] = home_dist / 200.0
    vec[13] = visit_dist / 200.0
    vec[14] = urgent_countdown
    vec[15] = rest_deficit
    vec[16] = deadhead_budget_ratio
    vec[17] = mandatory_pending / 5.0
    vec[18] = scheduled_pending / 3.0

    # ── Top-5 候选货源特征 (6维 x 5 = 30维) [19..48] ──
    cost_per_km = 1.5
    for i in range(_TOP_K):
        offset = 19 + i * 6
        if i < len(candidates):
            c = candidates[i]
            cargo = c.get("cargo", {})
            price = float(cargo.get("price", 0))
            deadhead_km = float(c.get("distance_km", 0))
            haul_km = float(c.get("haul_km", 0) or _compute_haul_km(cargo))
            cost_min = max(1, int(cargo.get("cost_time_minutes", 1)))

            # 空间收益查询
            end = cargo.get("end", {})
            end_lat = end.get("lat")
            end_lng = end.get("lng")
            spatial_val = 0.0
            if end_lat is not None and end_lng is not None and state_tracker is not None:
                get_sv = getattr(state_tracker, "get_spatial_value", None)
                if callable(get_sv):
                    spatial_val = get_sv(float(end_lat), float(end_lng))

            net_profit = price - (deadhead_km + haul_km) * cost_per_km

            vec[offset + 0] = price / 5000.0
            vec[offset + 1] = deadhead_km / 200.0
            vec[offset + 2] = haul_km / 1000.0
            vec[offset + 3] = cost_min / 480.0
            vec[offset + 4] = net_profit / 3000.0
            vec[offset + 5] = spatial_val / 10000.0

    return vec


# ─────────────────────────────────────────────────────────
# Mock API（当 api_port 为 None 时用于离线测试）
# ─────────────────────────────────────────────────────────

class _MockSimulationApi:
    """模拟仿真 API，用随机数生成数据，方便离线训练/测试。"""

    def __init__(self):
        self._progress = {}
        self._positions = {}

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        progress = self._progress.get(driver_id, 0)
        pos = self._positions.get(driver_id, (22.54 + random.uniform(-0.5, 0.5),
                                               114.06 + random.uniform(-0.5, 0.5)))
        return {
            "driver_id": driver_id,
            "current_lat": pos[0],
            "current_lng": pos[1],
            "simulation_progress_minutes": progress,
            "simulation_horizon_minutes": _SIM_DURATION_MIN,
            "preferences": [],
            "completed_order_count": random.randint(0, 50),
        }

    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]:
        n_items = random.randint(1, 7)
        items = []
        for _ in range(n_items):
            s_lat = latitude + random.uniform(-0.3, 0.3)
            s_lng = longitude + random.uniform(-0.3, 0.3)
            e_lat = latitude + random.uniform(-2.0, 2.0)
            e_lng = longitude + random.uniform(-2.0, 2.0)
            price = random.uniform(500, 5000)
            cost_min = random.randint(30, 480)
            dh_km = random.uniform(5, 100)
            items.append({
                "distance_km": dh_km,
                "cargo": {
                    "cargo_id": f"C{random.randint(100000, 999999)}",
                    "cargo_name": random.choice(["日用百货", "电子产品", "建材", "农产品"]),
                    "price": price,
                    "cost_time_minutes": cost_min,
                    "start": {"lat": s_lat, "lng": s_lng},
                    "end": {"lat": e_lat, "lng": e_lng},
                    "load_time": None,
                }
            })
        return {"items": items}

    def take_order(self, driver_id: str, cargo_id: str) -> dict[str, Any]:
        progress = self._progress.get(driver_id, 0)
        cost_min = random.randint(60, 360)
        self._progress[driver_id] = progress + cost_min
        dh_km = random.uniform(5, 80)
        haul_km = random.uniform(30, 300)
        revenue = random.uniform(800, 4000)
        new_lat = self._positions.get(driver_id, (22.54, 114.06))[0] + random.uniform(-1, 1)
        new_lng = self._positions.get(driver_id, (22.54, 114.06))[1] + random.uniform(-1, 1)
        self._positions[driver_id] = (new_lat, new_lng)
        return {
            "accepted": random.random() > 0.1,
            "position_after": {"lat": new_lat, "lng": new_lng},
            "simulation_progress_minutes": self._progress[driver_id],
            "pickup_deadhead_km": dh_km,
            "haul_distance_km": haul_km,
            "revenue": revenue,
        }

    def wait(self, driver_id: str, duration_minutes: int) -> dict[str, Any]:
        progress = self._progress.get(driver_id, 0)
        self._progress[driver_id] = progress + duration_minutes
        return {"simulation_progress_minutes": self._progress[driver_id]}

    def reposition(self, driver_id: str, lat: float, lng: float) -> dict[str, Any]:
        old_pos = self._positions.get(driver_id, (22.54, 114.06))
        dist = _haversine_km(old_pos[0], old_pos[1], lat, lng)
        travel_min = max(10, int(dist / 60.0 * 60))  # 60km/h
        progress = self._progress.get(driver_id, 0)
        self._progress[driver_id] = progress + travel_min
        self._positions[driver_id] = (lat, lng)
        return {
            "simulation_progress_minutes": self._progress[driver_id],
            "position_after": {"lat": lat, "lng": lng},
        }


# ─────────────────────────────────────────────────────────
# LogisticsDriverEnv: 单司机 Gymnasium 环境
# ─────────────────────────────────────────────────────────

class LogisticsDriverEnv(gym.Env):
    """
    单司机环境。每个 episode = 一个完整月度模拟（31天）。
    外部通过 SimulationApiPort 与仿真服务交互。
    当 api_port 为 None 时自动启用 mock 模式。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        api_port=None,
        driver_id: str = "D001",
        driver_config: dict | None = None,
        *,
        api=None,
        decision_service=None,
        candidate_ranker: Callable[[list[dict[str, Any]], dict[str, Any], Any, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        max_steps: int = 2000,
    ):
        """
        Args:
            api_port: SimulationApiPort 实例，若为 None 则使用 mock 模式
            driver_id: 司机ID，如 "D001"
            driver_config: 包含 cost_per_km, home_lat/lng, max_deadhead_km 等
            api: api_port 的别名（兼容 train.py 调用）
            decision_service: 兼容参数（忽略）
            max_steps: 最大步数限制
        """
        super().__init__()
        # api 和 api_port 兼容处理
        if api_port is None and api is not None:
            api_port = api
        if driver_config is None:
            driver_config = {}
        self.api = api_port if api_port is not None else _MockSimulationApi()
        self.driver_id = driver_id
        self.driver_config = driver_config
        self.cost_per_km = driver_config.get("cost_per_km", 1.5)
        self.decision_service = decision_service
        self._candidate_ranker = candidate_ranker
        self.constraints = constraints if constraints is not None else driver_config.get("constraints", [])
        self._max_steps = max_steps
        self._step_count = 0

        # 状态空间：49维浮点向量
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(_STATE_DIM,), dtype=np.float32
        )

        # 动作空间：9个离散动作
        self.action_space = spaces.Discrete(_ACTION_DIM)

        # 内部状态
        self._current_status: dict[str, Any] | None = None
        self._current_cargo_list: list[dict[str, Any]] = []
        self.spatial_income: dict[tuple[int, int], float] = {}
        self._spatial_income: dict[str, float] = {}  # 网格位置 → 历史收益
        self._total_income = 0.0
        self._total_mileage = 0.0
        self._total_deadhead = 0.0
        self._total_penalty = 0.0
        self._total_orders = 0
        self._spatial_income = {}
        self.spatial_income = {}
        self._orders_today = 0
        self._rest_today_min = 0.0
        self._last_rest_duration = 0.0
        self._max_rest_today_min = 0.0
        self.last_action_end_min = 0
        self._consecutive_wait_minutes: dict[str, float] = {}
        self._episode_done = False
        self._last_day = 0
        self.open_tasks: list[dict[str, Any]] = list(driver_config.get("open_tasks", []))
        self.completed_tasks: list[dict[str, Any]] = []
        self.monthly_planner = None
        self.completed_idle_days = 0
        self.cargo_frequency: dict[int, int] = {}

    def reset(self, seed=None, options=None):
        """重置环境，从仿真服务获取初始状态。"""
        super().reset(seed=seed)
        self._total_income = 0.0
        self._total_mileage = 0.0
        self._total_deadhead = 0.0
        self._total_penalty = 0.0
        self._total_orders = 0
        self._spatial_income = {}
        self.spatial_income = {}
        self._orders_today = 0
        self._rest_today_min = 0.0
        self._last_rest_duration = 0.0
        self._max_rest_today_min = 0.0
        self.last_action_end_min = 0
        self._consecutive_wait_minutes = {self.driver_id: 0}
        self._episode_done = False
        self._last_day = 0
        self._step_count = 0
        self.open_tasks = list(self.driver_config.get("open_tasks", []))
        self.completed_tasks = []
        self.completed_idle_days = 0
        self.cargo_frequency = {}

        # 如果是 mock 模式，重置进度
        if isinstance(self.api, _MockSimulationApi):
            self.api._progress[self.driver_id] = 0
            home = self.driver_config.get("home")
            if home:
                self.api._positions[self.driver_id] = (home["lat"], home["lng"])
            else:
                init_lat = self.driver_config.get("current_lat", 22.54)
                init_lng = self.driver_config.get("current_lng", 114.06)
                self.api._positions[self.driver_id] = (init_lat, init_lng)

        status = self.api.get_driver_status(self.driver_id)
        self._current_status = status
        self._current_cargo_list = self._refresh_candidates(status)

        obs = self._encode_state(status, self._current_cargo_list)
        info = {"driver_id": self.driver_id}
        return obs, info

    def _refresh_candidates(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        cargo_result = self.api.query_cargo(
            self.driver_id, status["current_lat"], status["current_lng"]
        )
        raw_items = cargo_result.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []

        ranked_items: list[dict[str, Any]] | None = None
        if self._candidate_ranker is not None:
            try:
                ranked_items = self._candidate_ranker(
                    raw_items, status, self, self.constraints
                )
            except Exception:
                ranked_items = None

        source_items = raw_items if ranked_items is None else ranked_items
        return [self._candidate_to_env_item(item) for item in source_items[:_TOP_K]]

    @staticmethod
    def _candidate_to_env_item(item: dict[str, Any]) -> dict[str, Any]:
        if isinstance(item.get("cargo"), dict):
            env_item = dict(item)
            env_item["cargo"] = dict(item["cargo"])
            if "haul_km" not in env_item:
                env_item["haul_km"] = _compute_haul_km(env_item["cargo"])
            return env_item

        cargo = {
            "cargo_id": item.get("cargo_id", ""),
            "cargo_name": item.get("cargo_name", ""),
            "price": item.get("price", 0.0),
            "cost_time_minutes": item.get(
                "cost_time_minutes", item.get("total_minutes", 1)
            ),
            "start": item.get("start", {}),
            "end": item.get("end", {}),
            "load_time": item.get("load_time"),
        }
        return {
            "distance_km": item.get("distance_km", item.get("deadhead_km", 0.0)),
            "haul_km": item.get("haul_km", _compute_haul_km(cargo)),
            "cargo": cargo,
            "score": item.get("score", item.get("true_net", 0.0)),
            "true_net": item.get("true_net", item.get("score", 0.0)),
            "profit_search_score": item.get("profit_search_score"),
            "has_soft_penalty": item.get("has_soft_penalty", False),
            "hard_penalty": item.get("hard_penalty", 0.0),
        }

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(_ACTION_DIM, dtype=np.bool_)
        mask[ACTION_WAIT_30] = True
        mask[ACTION_WAIT_60] = True
        for idx in range(min(len(self._current_cargo_list), _TOP_K)):
            mask[ACTION_CARGO_0 + idx] = True
        mask[ACTION_REPOSITION_HOME] = bool(self.driver_config.get("home"))
        mask[ACTION_REPOSITION_HOTZONE] = self._get_hotzone() is not None
        return mask

    def _record_wait(self, duration_minutes: int) -> None:
        self._consecutive_wait_minutes[self.driver_id] = (
            self._consecutive_wait_minutes.get(self.driver_id, 0) + duration_minutes
        )
        self._rest_today_min += duration_minutes
        self._last_rest_duration += duration_minutes
        self._max_rest_today_min = max(
            self._max_rest_today_min, self._last_rest_duration
        )

    @staticmethod
    def _effective_revenue(action_result: dict[str, Any]) -> float:
        if not action_result.get("accepted"):
            return 0.0
        if not action_result.get("income_eligible", True):
            return 0.0
        return float(action_result.get("revenue", 0.0) or 0.0)

    def step(self, action: int):
        """执行动作，返回 (obs, reward, terminated, truncated, info)。"""
        self._step_count += 1
        prev_status = self._current_status
        action_result: dict[str, Any] = {}

        # 检测日期切换，重置每日计数器
        curr_progress = prev_status.get("simulation_progress_minutes", 0)
        curr_day = curr_progress // 1440
        if curr_day != self._last_day:
            self._orders_today = 0
            self._rest_today_min = 0.0
            self._max_rest_today_min = 0.0
            self._last_day = curr_day

        # ──── 执行动作 ────
        if action == ACTION_WAIT_30:
            self.api.wait(self.driver_id, 30)
            self._record_wait(30)

        elif action == ACTION_WAIT_60:
            self.api.wait(self.driver_id, 60)
            self._record_wait(60)

        elif 2 <= action <= 6:
            cargo_idx = action - 2
            self._consecutive_wait_minutes[self.driver_id] = 0
            self._last_rest_duration = 0.0
            if cargo_idx < len(self._current_cargo_list):
                cargo_item = self._current_cargo_list[cargo_idx]
                cargo_id = str(cargo_item["cargo"]["cargo_id"])
                action_result = self.api.take_order(self.driver_id, cargo_id)
                if action_result.get("accepted"):
                    revenue = self._effective_revenue(action_result)
                    self._total_income += revenue
                    dh = action_result.get("pickup_deadhead_km", 0)
                    haul = action_result.get("haul_distance_km", 0)
                    self._total_mileage += haul
                    self._total_deadhead += dh
                    self._total_orders += 1
                    self._orders_today += 1
                    # 更新空间收益热图
                    cargo = cargo_item["cargo"]
                    end_key = self._pos_to_grid(
                        cargo["end"]["lat"], cargo["end"]["lng"]
                    )
                    net = revenue - (dh + haul) * self.cost_per_km
                    self._spatial_income[end_key] = \
                        self._spatial_income.get(end_key, 0) + net
                    tuple_key = (
                        round(float(cargo["end"]["lat"]) * 10),
                        round(float(cargo["end"]["lng"]) * 10),
                    )
                    self.spatial_income[tuple_key] = \
                        self.spatial_income.get(tuple_key, 0.0) + net
                    self.last_action_end_min = int(
                        action_result.get("simulation_progress_minutes", curr_progress)
                    )
            else:
                # 无效动作，降级为 wait 30min
                self.api.wait(self.driver_id, 30)
                self._record_wait(30)

        elif action == ACTION_REPOSITION_HOME:
            home = self.driver_config.get("home")
            if home:
                self.api.reposition(self.driver_id, home["lat"], home["lng"])
                self._consecutive_wait_minutes[self.driver_id] = 0
                self._last_rest_duration = 0.0
            else:
                self.api.wait(self.driver_id, 30)
                self._record_wait(30)

        elif action == ACTION_REPOSITION_HOTZONE:
            hotzone = self._get_hotzone()
            if hotzone:
                self.api.reposition(self.driver_id, hotzone[0], hotzone[1])
                self._consecutive_wait_minutes[self.driver_id] = 0
                self._last_rest_duration = 0.0
            else:
                self.api.wait(self.driver_id, 30)
                self._record_wait(30)

        # ──── 获取新状态 ────
        curr_status = self.api.get_driver_status(self.driver_id)
        self._current_status = curr_status

        # 查询新货源
        self._current_cargo_list = self._refresh_candidates(curr_status)

        # 计算 reward
        reward = self._compute_reward(
            action, action_result, prev_status, curr_status, self.driver_id
        )

        # 判断终止
        progress = curr_status["simulation_progress_minutes"]
        horizon = curr_status.get("simulation_horizon_minutes", _SIM_DURATION_MIN)
        terminated = progress >= horizon
        truncated = self._step_count >= self._max_steps
        net_income = self._net_income()
        terminal_reward = 0.0
        if terminated or truncated:
            terminal_reward = net_income / 1000.0
            reward += terminal_reward

        obs = self._encode_state(curr_status, self._current_cargo_list)
        info = {
            "total_income": self._total_income,
            "net_income": net_income,
            "total_mileage": self._total_mileage,
            "total_deadhead": self._total_deadhead,
            "total_penalty": self._total_penalty,
            "terminal_reward": terminal_reward,
            "progress_minutes": progress,
            "orders_today": self._orders_today,
            "total_orders": self._total_orders,
            "action_mask": self.get_action_mask(),
        }

        return obs, reward, terminated, truncated, info

    @property
    def total_income(self) -> float:
        return self._total_income

    @property
    def total_mileage_km(self) -> float:
        return self._total_mileage + self._total_deadhead

    @property
    def total_deadhead_km(self) -> float:
        return self._total_deadhead

    @property
    def total_orders(self) -> int:
        return self._total_orders

    def _net_income(self) -> float:
        distance_cost = (self._total_mileage + self._total_deadhead) * self.cost_per_km
        return self._total_income - distance_cost - self._total_penalty

    def get_orders_today(self, current_min: int) -> int:
        return self._orders_today

    def was_waiting_until(self, current_min: int) -> bool:
        return self._last_rest_duration > 0

    def record_cargo_seen(self, current_min: int, count: int) -> None:
        hour_slot = int((current_min % 1440) / 60)
        self.cargo_frequency[hour_slot] = self.cargo_frequency.get(hour_slot, 0) + int(count)

    def get_avg_cargo_per_hour(self, current_min: int) -> float:
        if not self.cargo_frequency:
            return 0.0
        return float(sum(self.cargo_frequency.values())) / max(1, len(self.cargo_frequency))

    # ─────────────────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────────────────

    def _encode_state(
        self, status: dict[str, Any], cargo_list: list[dict[str, Any]]
    ) -> np.ndarray:
        """将当前状态编码为 49 维归一化向量。"""
        vec = np.zeros(_STATE_DIM, dtype=np.float32)
        sim_min = int(status.get("simulation_progress_minutes", 0))

        # ── 时空特征 (5维) [0..4] ──
        lat = float(status.get("current_lat", 22.0))
        lng = float(status.get("current_lng", 110.0))
        vec[0] = (lat - 22.0) / 4.0
        vec[1] = (lng - 110.0) / 8.0
        vec[2] = (sim_min % 1440) / 1440.0
        vec[3] = (sim_min // 1440) / 30.0
        day_idx = sim_min // 1440
        day_of_week = (day_idx + 6) % 7  # 2026-03-01=周日 → 0=周一映射
        vec[4] = 1.0 if day_of_week >= 5 else 0.0

        # ── 累计状态 (6维) [5..10] ──
        vec[5] = self._total_income / 100000.0
        vec[6] = self._total_mileage / 10000.0
        vec[7] = self._total_deadhead / 5000.0
        vec[8] = self._orders_today / 10.0
        vec[9] = self._rest_today_min / 480.0
        vec[10] = self._last_rest_duration / 480.0

        # ── 约束状态 (8维) [11..18] ──
        vec[11] = self._total_penalty / 10000.0

        # home_distance
        home = self.driver_config.get("home")
        if home:
            home_dist = _haversine_km(lat, lng, home["lat"], home["lng"])
            vec[12] = home_dist / 200.0
        else:
            vec[12] = 0.0

        # visit_target_distance
        visit = self.driver_config.get("visit_target")
        if visit:
            visit_dist = _haversine_km(lat, lng, visit["lat"], visit["lng"])
            vec[13] = visit_dist / 200.0
        else:
            vec[13] = 0.0

        # urgent_task_countdown
        vec[14] = 1.0  # 默认无紧急任务

        # rest_deficit
        required_rest = self.driver_config.get("required_rest_min", 240)
        rest_deficit = max(0.0, required_rest - self._rest_today_min)
        vec[15] = rest_deficit / 300.0

        # deadhead_budget_ratio
        max_monthly_deadhead = self.driver_config.get("max_deadhead_km", 0)
        if max_monthly_deadhead > 0:
            vec[16] = self._total_deadhead / max_monthly_deadhead
        else:
            vec[16] = 0.0

        # mandatory_cargo_pending / scheduled_event_pending
        vec[17] = self.driver_config.get("mandatory_cargo_pending", 0) / 5.0
        vec[18] = self.driver_config.get("scheduled_event_pending", 0) / 3.0

        # ── Top-5 候选货源特征 (6维 x 5 = 30维) [19..48] ──
        for i in range(_TOP_K):
            offset = 19 + i * 6
            if i < len(cargo_list):
                c = cargo_list[i]
                cargo = c.get("cargo", {})
                price = float(cargo.get("price", 0))
                deadhead_km = float(c.get("distance_km", 0))
                haul_km = float(c.get("haul_km", 0) or _compute_haul_km(cargo))
                cost_min = max(1, int(cargo.get("cost_time_minutes", 1)))

                net_profit = price - (deadhead_km + haul_km) * self.cost_per_km

                end = cargo.get("end", {})
                end_lat = end.get("lat")
                end_lng = end.get("lng")
                spatial_val = 0.0
                if end_lat is not None and end_lng is not None:
                    spatial_val = self._get_spatial_value(
                        float(end_lat), float(end_lng)
                    )

                vec[offset + 0] = price / 5000.0
                vec[offset + 1] = deadhead_km / 200.0
                vec[offset + 2] = haul_km / 1000.0
                vec[offset + 3] = cost_min / 480.0
                vec[offset + 4] = net_profit / 3000.0
                vec[offset + 5] = spatial_val / 10000.0
            # else: 保持 0 填充

        return vec

    def _compute_reward(
        self,
        action: int,
        action_result: dict[str, Any],
        prev_status: dict[str, Any],
        curr_status: dict[str, Any],
        driver_id: str,
    ) -> float:
        """计算即时奖励。"""
        reward = 0.0

        # 1. 即时净收益（核心奖励）
        if 2 <= action <= 6 and action_result.get("accepted"):
            revenue = self._effective_revenue(action_result)
            deadhead_km = action_result.get("pickup_deadhead_km", 0)
            haul_km = action_result.get("haul_distance_km", 0)
            cost = (deadhead_km + haul_km) * self.cost_per_km
            reward += (revenue - cost) / 1000.0

        # 2. 新增罚分惩罚
        new_penalty = curr_status.get("new_penalty", 0)
        reward -= new_penalty / 500.0
        self._total_penalty += new_penalty

        # 3. 时间效率奖励（惩罚无效等待）
        time_elapsed = (
            curr_status["simulation_progress_minutes"]
            - prev_status["simulation_progress_minutes"]
        )
        if action in (ACTION_WAIT_30, ACTION_WAIT_60) and time_elapsed > 0:
            reward -= 0.05 * (time_elapsed / 60.0)

        # 4. 针对低活跃司机的激进探索奖励
        if driver_id in ("D002", "D006", "D007"):
            if 2 <= action <= 6 and action_result.get("accepted"):
                reward += 0.5
            if action in (ACTION_WAIT_30, ACTION_WAIT_60):
                consecutive_wait_h = (
                    self._consecutive_wait_minutes.get(driver_id, 0) / 60.0
                )
                if consecutive_wait_h > 2.0:
                    reward -= 0.1 * (consecutive_wait_h - 2.0)

        # 5. 月末紧迫度加权
        day = curr_status["simulation_progress_minutes"] // 1440
        urgency = 1.0 + 0.3 * (day / 30.0)
        reward *= urgency

        # 6. 休息完成奖励（D008重点）
        if action in (ACTION_WAIT_30, ACTION_WAIT_60):
            if self._just_completed_rest_requirement(driver_id, curr_status):
                reward += 0.3

        if compute_reward_shaping is not None:
            action_name = "other"
            duration = 0
            if action == ACTION_WAIT_30:
                action_name = "wait"
                duration = 30
            elif action == ACTION_WAIT_60:
                action_name = "wait"
                duration = 60
            elif 2 <= action <= 6 and action_result.get("accepted"):
                action_name = "take_order"

            reward += compute_reward_shaping(
                driver_id=driver_id,
                state_tracker=self,
                sim_min=int(curr_status["simulation_progress_minutes"]),
                action=action_name,
                duration_minutes=duration,
                income_delta=self._effective_revenue(action_result),
                cargo_deadhead_km=float(
                    action_result.get("pickup_deadhead_km", 0.0) or 0.0
                ),
            ) / 1000.0

        return reward

    def get_max_continuous_rest_today(self, sim_min: int) -> float:
        return self._max_rest_today_min

    def get_current_rest_streak(self, sim_min: int) -> float:
        return self._last_rest_duration

    def _pos_to_grid(self, lat: float, lng: float) -> str:
        """将经纬度映射到 0.5 度网格键，如 '22.5_114.0'。"""
        grid_lat = round(math.floor(float(lat) * 2) / 2, 1)
        grid_lng = round(math.floor(float(lng) * 2) / 2, 1)
        return f"{grid_lat}_{grid_lng}"

    def _get_hotzone(self) -> tuple[float, float] | None:
        """返回历史最高收益网格的中心坐标。"""
        if not self._spatial_income:
            return None
        best_key = max(self._spatial_income, key=self._spatial_income.get)
        parts = best_key.split("_")
        if len(parts) == 2:
            try:
                grid_lat = float(parts[0]) + 0.25  # 网格中心
                grid_lng = float(parts[1]) + 0.25
                return (grid_lat, grid_lng)
            except ValueError:
                return None
        return None

    def _just_completed_rest_requirement(
        self, driver_id: str, status: dict[str, Any]
    ) -> bool:
        """判断是否刚刚完成休息要求（连续休息达标）。"""
        required_rest = self.driver_config.get("required_rest_min", 240)
        # 检查是否刚好跨过要求的阈值
        prev_rest = self._last_rest_duration - (
            30 if self._last_rest_duration >= 30 else 0
        )
        if prev_rest < required_rest <= self._last_rest_duration:
            return True
        return False

    def _get_spatial_value(self, lat: float, lng: float) -> float:
        """查询给定位置的历史空间收益值。"""
        key = self._pos_to_grid(lat, lng)
        return self._spatial_income.get(key, 0.0)

    def get_spatial_value(self, lat: float, lng: float) -> float:
        return self._get_spatial_value(lat, lng)


# ─────────────────────────────────────────────────────────
# 向后兼容别名
# ─────────────────────────────────────────────────────────

DriverRLEnv = LogisticsDriverEnv


# ─────────────────────────────────────────────────────────
# MultiDriverEnv: 10个司机的并行环境
# ─────────────────────────────────────────────────────────

class MultiDriverEnv:
    """
    管理10个司机的并行环境。
    每次 step 可以对单个司机决策，也可以批量决策。
    """

    def __init__(self, api_port, drivers_config: list[dict]):
        """
        Args:
            api_port: SimulationApiPort 实例（所有司机共享），None 则 mock
            drivers_config: 每个元素包含 driver_id, cost_per_km, home 等配置
        """
        self.envs: dict[str, LogisticsDriverEnv] = {
            d["driver_id"]: LogisticsDriverEnv(api_port, d["driver_id"], d)
            for d in drivers_config
        }

    def reset_all(self) -> dict[str, np.ndarray]:
        """重置所有司机环境，返回 {driver_id: obs}。"""
        return {did: env.reset()[0] for did, env in self.envs.items()}

    def step_driver(
        self, driver_id: str, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """对单个司机执行动作。"""
        return self.envs[driver_id].step(action)

    def step_all(
        self, actions: dict[str, int]
    ) -> dict[str, tuple[np.ndarray, float, bool, bool, dict]]:
        """对所有司机批量执行动作。

        Args:
            actions: {driver_id: action_int}

        Returns:
            {driver_id: (obs, reward, terminated, truncated, info)}
        """
        results = {}
        for did, act in actions.items():
            if did in self.envs:
                results[did] = self.envs[did].step(act)
        return results

    def get_obs(self, driver_id: str) -> np.ndarray:
        """获取指定司机的当前观测向量。"""
        env = self.envs[driver_id]
        return env._encode_state(env._current_status, env._current_cargo_list)

    def get_all_obs(self) -> dict[str, np.ndarray]:
        """获取所有司机当前观测。"""
        return {did: self.get_obs(did) for did in self.envs}

    def is_all_done(self) -> bool:
        """检查所有司机是否都完成了 episode。"""
        for env in self.envs.values():
            if env._current_status is None:
                continue
            progress = env._current_status.get("simulation_progress_minutes", 0)
            horizon = env._current_status.get(
                "simulation_horizon_minutes", _SIM_DURATION_MIN
            )
            if progress < horizon:
                return False
        return True

    @property
    def driver_ids(self) -> list[str]:
        """返回所有司机 ID 列表。"""
        return list(self.envs.keys())
