"""针对性优化：低活跃司机激活、空驶预算、休息规划的 reward shaping。"""

from __future__ import annotations

import math
from typing import Any

# ─────────────────────────────────────────────────────────
# 低活跃司机配置
# ─────────────────────────────────────────────────────────

# 需要激进策略的司机 ID
_LOW_ACTIVITY_DRIVERS = {"D002", "D006", "D007"}

# 低活跃司机：超过此小时数不接单开始罚分
_IDLE_PENALTY_THRESHOLD_HOURS = 2.0
_IDLE_PENALTY_PER_HOUR = -50.0

# 低活跃司机：允许微亏接单的最低 true_net
_LOW_ACTIVITY_MIN_ACCEPTABLE_NET = -100.0


# ─────────────────────────────────────────────────────────
# 空驶预算配置
# ─────────────────────────────────────────────────────────

# 空驶预算安全系数
_DEADHEAD_SAFETY_FACTOR = 1.2


class DeadheadBudgetTracker:
    """跟踪月度空驶预算，判断接单是否会导致超限。"""

    def __init__(self) -> None:
        self._monthly_cap: float = 5000.0  # 默认月度空驶上限
        self._used: float = 0.0

    def update(self, constraints: list[dict[str, Any]], current_deadhead: float) -> None:
        """从约束中读取月度空驶上限。"""
        self._used = current_deadhead
        for c in constraints:
            if c.get("type") == "mileage_cap":
                self._monthly_cap = float(c.get("params", {}).get("max_deadhead_km", 5000))

    def can_accept(self, cargo_deadhead_km: float, sim_min: int) -> bool:
        """判断接单后空驶是否会超限。"""
        day_idx = sim_min // 1440
        remaining_days = max(1, 30 - day_idx)
        budget_per_day = max(0, self._monthly_cap - self._used) / remaining_days
        return cargo_deadhead_km <= budget_per_day * _DEADHEAD_SAFETY_FACTOR

    def remaining_budget(self, sim_min: int) -> float:
        """返回剩余每日空驶预算。"""
        day_idx = sim_min // 1440
        remaining_days = max(1, 30 - day_idx)
        return max(0, self._monthly_cap - self._used) / remaining_days


# ─────────────────────────────────────────────────────────
# 休息规划配置
# ─────────────────────────────────────────────────────────

# 完成当日休息要求的奖励
_REST_COMPLETE_REWARD = 200.0

# 未满足休息的预罚（比实际罚分更重，鼓励提前规划）
_REST_DEFICIT_PENALTY = -500.0

# 最低连续休息要求（分钟）
_MIN_CONTINUOUS_REST = 240


def compute_rest_reward(
    state_tracker: Any,
    sim_min: int,
    action: str,
    duration_minutes: int = 0,
) -> float:
    """计算休息相关的 reward shaping。"""
    reward = 0.0

    max_rest_today = getattr(state_tracker, "get_max_continuous_rest_today", lambda _: 0)(sim_min)

    if action == "wait" and duration_minutes > 0:
        # 等待后预计的连续休息
        projected_rest = max_rest_today + duration_minutes
        if max_rest_today < _MIN_CONTINUOUS_REST and projected_rest >= _MIN_CONTINUOUS_REST:
            reward += _REST_COMPLETE_REWARD

    # 一天快结束时检查休息是否满足
    hour_of_day = (sim_min % 1440) / 60
    if hour_of_day >= 20 and max_rest_today < _MIN_CONTINUOUS_REST:
        reward += _REST_DEFICIT_PENALTY * 0.1  # 温和预罚

    return reward


def compute_idle_penalty(
    driver_id: str,
    state_tracker: Any,
    sim_min: int,
    last_order_min: int,
) -> float:
    """计算低活跃司机的空闲惩罚。"""
    if driver_id not in _LOW_ACTIVITY_DRIVERS:
        return 0.0

    idle_hours = (sim_min - last_order_min) / 60.0
    if idle_hours <= _IDLE_PENALTY_THRESHOLD_HOURS:
        return 0.0

    excess = idle_hours - _IDLE_PENALTY_THRESHOLD_HOURS
    return _IDLE_PENALTY_PER_HOUR * excess


def compute_reward_shaping(
    driver_id: str,
    state_tracker: Any,
    sim_min: int,
    action: str,
    duration_minutes: int = 0,
    income_delta: float = 0.0,
    cargo_deadhead_km: float = 0.0,
) -> float:
    """综合 reward shaping：低活跃 + 空驶 + 休息。"""
    reward = 0.0

    # 低活跃司机惩罚
    last_order_min = getattr(state_tracker, "last_action_end_min", 0)
    reward += compute_idle_penalty(driver_id, state_tracker, sim_min, last_order_min)

    # 休息规划
    reward += compute_rest_reward(state_tracker, sim_min, action, duration_minutes)

    # 空驶惩罚（接单时）
    if action == "take_order" and cargo_deadhead_km > 0:
        if cargo_deadhead_km > 100:
            reward -= (cargo_deadhead_km - 100) * 2.0

    return reward
