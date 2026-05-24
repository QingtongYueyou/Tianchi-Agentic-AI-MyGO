"""三层混合决策架构 Agent：规则层 → 启发式层 → LLM 层。"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort

from .strategic_planner import StrategicPlanner
from .decision_reviewer import DecisionReviewer

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_MONTH_TOTAL_MINUTES = 43200  # 30天
_SPEED_KM_H = 60.0
_COST_PER_KM_DEFAULT = 1.5
_TOKEN_BUDGET = 1_000_000  # 月度 token 预算上限
_TOKEN_DEGRADE_RATIO = 0.80
_CHINA_LAT_RANGE = (18.0, 54.0)
_CHINA_LNG_RANGE = (73.0, 136.0)
_MAX_SINGLE_REPOSITION_KM = 300.0

# ── true_net 参数 ──
_PENALTY_MARGIN = 200                    # 吃罚候选 true_net 需超过安全候选的最小差额
_MAX_SOFT_PENALTY_PER_DAY = 300
_MAX_SOFT_PENALTY_PER_ORDER = 300
_MAX_SOFT_PENALTY_PER_MONTH = 1000
_LONG_ORDER_THRESHOLD_MINUTES = 360      # 6 小时
_HEAVY_DEADHEAD_KM = 90
_EXTREME_DEADHEAD_KM = 150
_DEADHEAD_REJECT_TRUE_NET = 200          # 极端空驶只保留亏损保护，正收益优先跑
_LONG_ORDER_REJECT_TRUE_NET = 80         # >10h 长单只拒绝薄利单，避免把可赚订单等没

_logger = logging.getLogger("agent.hybrid_decision")


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """大圆距离（km）"""
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def _distance_to_minutes(distance_km: float, speed: float = _SPEED_KM_H) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil(distance_km / speed * 60))


def _to_int_or_zero(val: Any, default: int = 0) -> int:
    """Safely convert to int; return default for None/non-numeric values."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _sim_minutes_to_wall(sim_min: int) -> datetime:
    return _SIMULATION_EPOCH + timedelta(minutes=int(sim_min))


def _wall_to_sim_minutes(wall_str: str) -> int:
    dt = datetime.strptime(wall_str.strip(), "%Y-%m-%d %H:%M:%S")
    delta = dt - _SIMULATION_EPOCH
    return int(delta.total_seconds() / 60)


def _get_day_index(sim_min: int) -> int:
    """第几天（0-indexed）"""
    return sim_min // 1440


def _get_hour_of_day(sim_min: int) -> float:
    """当天小时数（0.0 ~ 24.0）"""
    min_in_day = sim_min % 1440
    return min_in_day / 60.0


def _get_horizon_minutes(status: dict) -> int:
    """优先使用评测环境下发的仿真期限，兜底使用月度默认值。"""
    try:
        horizon = int(status.get("simulation_horizon_minutes", _MONTH_TOTAL_MINUTES))
        return max(1, horizon)
    except Exception:
        return _MONTH_TOTAL_MINUTES


def _iter_days_touched(start_min: int, end_min: int) -> range:
    """返回 [start, end) 覆盖的自然日索引。"""
    if end_min <= start_min:
        return range(_get_day_index(start_min), _get_day_index(start_min) + 1)
    return range(_get_day_index(start_min), _get_day_index(end_min - 1) + 1)


def _violates_time_restriction(start_min: int, end_min: int, constraints: list[dict]) -> bool:
    for c in constraints:
        if c.get("type") != "time_restriction":
            continue
        params = c.get("params", {})
        if "take_order" not in params.get("forbidden_actions", ["take_order", "reposition"]):
            continue
        start_h = params.get("start_hour")
        end_h = params.get("end_hour")
        if start_h is None or end_h is None:
            continue
        for day in _iter_days_touched(start_min, end_min):
            forbidden_start = day * 1440 + int(float(start_h) * 60)
            if float(start_h) <= float(end_h):
                forbidden_end = day * 1440 + int(float(end_h) * 60)
                windows = [(forbidden_start, forbidden_end)]
            else:
                windows = [
                    (forbidden_start, (day + 1) * 1440),
                    (day * 1440, day * 1440 + int(float(end_h) * 60)),
                ]
            for a, b in windows:
                if start_min < b and end_min > a:
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MonthlyConstraintPlanner - 月度偏好约束前置规划器 (Task #1)
# ─────────────────────────────────────────────────────────────────────────────
class MonthlyConstraintPlanner:
    """月度偏好约束前置规划器：前瞻式安排空闲日，消除被动罚分。"""

    def __init__(self, preferences_text: str, total_days: int = 31):
        """从偏好文本中解析月度约束并生成空闲日计划。"""
        self.total_days = total_days
        self.required_idle_days = self._parse_idle_days_requirement(preferences_text)
        self.planned_idle_days: list[int] = self._generate_idle_plan()
        # --- 战略计划扩展 ---
        self.rest_plan: list[dict] = []           # 来自战略计划的休息安排
        self.home_visit_plan: list[dict] = []     # 回家/到访计划
        self.scheduled_events: list[dict] = []    # 家事/固定事件
        self.mandatory_cargos: list[dict] = []    # 必接货源

    def _parse_idle_days_requirement(self, text: str) -> int:
        """解析偏好文本中需要几个整天不接单/不活动。"""
        # 匹配类似 "至少X天不接单"、"每月休息X天"、"X个整天" 等
        patterns = [
            r'(\d+)\s*[天个]\s*(?:整天|不接单|休息日|不活动)',
            r'(?:至少|最少)\s*(\d+)\s*天',
            r'(\d+)\s*天\s*(?:的?休息|不接单|不工作|空闲)',
            r'休息\s*(\d+)\s*天',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        # 未匹配到，默认0
        return 0

    def _generate_idle_plan(self) -> list[int]:
        """生成月度空闲日计划，优先安排在周末和月中（货源通常较少的日期）。"""
        if self.required_idle_days <= 0:
            return []

        # 策略: 均匀分布在月内，优先周末位置（7, 14, 21, 28）
        # 然后补充其他周中间位置（如 10, 17, 24 等）
        preferred_days = [7, 14, 21, 28, 10, 17, 24, 5, 12, 19, 26, 3, 8, 15, 22, 29]
        # 过滤掉超出月度总天数的日期
        preferred_days = [d for d in preferred_days if d <= self.total_days]

        planned = []
        for d in preferred_days:
            if len(planned) >= self.required_idle_days:
                break
            planned.append(d)

        # 如果首选日不够，补充剩余日期
        if len(planned) < self.required_idle_days:
            for d in range(1, self.total_days + 1):
                if d not in planned:
                    planned.append(d)
                if len(planned) >= self.required_idle_days:
                    break

        planned.sort()
        return planned

    def get_planned_idle_days(self) -> list[int]:
        """返回规划的空闲日列表（1-based, 1-31）。"""
        return self.planned_idle_days

    def is_today_idle_day(self, current_day: int) -> bool:
        """当前天是否为规划的空闲日（current_day: 1-based）。"""
        return current_day in self.planned_idle_days

    def get_progress_report(self, completed_idle_days: int, current_day: int) -> dict:
        """返回偏好达成进度。"""
        remaining_required = max(0, self.required_idle_days - completed_idle_days)
        remaining_available_days = self.total_days - current_day
        is_behind_schedule = (
            remaining_required > 0 and
            remaining_available_days < remaining_required * 2  # 留有一定余量
        )
        return {
            "required_total": self.required_idle_days,
            "completed": completed_idle_days,
            "remaining_required": remaining_required,
            "remaining_available_days": remaining_available_days,
            "is_behind_schedule": is_behind_schedule,
        }

    def should_trigger_rescue_mode(self, completed_idle_days: int, current_day: int) -> bool:
        """是否需要触发约束补救模式。
        如果剩余天数不足以完成剩余约束，触发补救。
        """
        remaining_required = max(0, self.required_idle_days - completed_idle_days)
        if remaining_required <= 0:
            return False
        remaining_available_days = self.total_days - current_day
        # 剩余天数刚好等于或少于剩余需求时触发补救
        return remaining_available_days <= remaining_required

    def integrate_strategic_plan(self, plan: dict) -> None:
        """整合战略计划中的月度级约束"""
        self.rest_plan = plan.get("rest_plan", [])
        self.home_visit_plan = plan.get("home_or_visit_plan", [])
        # 从 must_do_tasks 中提取分类
        for task in plan.get("must_do_tasks", []):
            task_type = task.get("type", "")
            if task_type == "scheduled_event":
                self.scheduled_events.append(task)
            elif task_type in ("mandatory_cargo",):
                self.mandatory_cargos.append(task)
        # 更新 planned_idle_days（如果战略计划给出了更优的休息日安排）
        strategic_rest_days = [
            r["day"] for r in self.rest_plan
            if r.get("type") == "full_day" and isinstance(r.get("day"), int)
        ]
        if strategic_rest_days and len(strategic_rest_days) >= self.required_idle_days:
            self.planned_idle_days = sorted(strategic_rest_days[:self.required_idle_days])

    def get_day_plan(self, day_index: int) -> dict:
        """获取指定日的月度计划"""
        return {
            "is_planned_idle": day_index in self.planned_idle_days,
            "has_event": any(
                e.get("deadline_minute", 0) // 1440 == day_index or e.get("day") == day_index
                for e in self.scheduled_events
            ),
            "has_visit": any(
                v.get("day") == day_index or v.get("day") == 0  # day=0 means every day
                for v in self.home_visit_plan
            ),
            "mandatory_cargo": [
                c for c in self.mandatory_cargos
                # mandatory cargo 没有特定日期限制，全月有效
            ] if day_index > 0 else [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# StateTracker - 状态追踪器
# ─────────────────────────────────────────────────────────────────────────────
class StateTracker:
    """维护累计收益、里程、接单统计、每日休息等状态。"""

    def __init__(self) -> None:
        self.total_income: float = 0.0
        self.total_mileage_km: float = 0.0
        self.total_deadhead_km: float = 0.0
        self.total_orders: int = 0
        self.orders_by_day: dict[int, int] = {}  # day_index -> order_count
        # 休息追踪：记录每个 wait 区间
        self.wait_intervals: list[tuple[int, int]] = []  # (start_min, end_min)
        self.last_action_end_min: int = 0
        # 无接单天数
        self.idle_days: set[int] = set()
        self.active_days: set[int] = set()  # 有 take_order 或 reposition 的天
        # 空间记忆：网格热力图 (grid_key -> cumulative_income)
        self.spatial_income: dict[tuple[int, int], float] = {}
        # Token 消耗
        self.total_tokens_used: int = 0
        # 是否已初始化
        self._initialized: bool = False
        # 月度规划器（Task #1）
        self.monthly_planner: MonthlyConstraintPlanner | None = None
        self.completed_idle_days: int = 0
        # 货源出现频率统计 (hour_slot -> count)（Task #2）
        self.cargo_frequency: dict[int, int] = {}
        # --- 战略计划相关 ---
        self.strategic_plan: dict | None = None
        self.open_tasks: list[dict] = []        # 未完成的 must_do_tasks
        self.completed_tasks: list[dict] = []   # 已完成的任务
        self.risk_register: list[dict] = []     # 活跃风险
        self.daily_intent: dict | None = None   # 当日策略意图
        self.last_llm_review_minute: int = -1   # 上次 reviewer 调用时间
        self.consecutive_review_waits: int = 0  # reviewer 连续输出 wait 的次数
        self.last_review_reason: str = ""       # 上次 reviewer 否决原因

    def initialize_from_history(self, api: SimulationApiPort, driver_id: str) -> None:
        """首次调用时从历史决策恢复状态。"""
        if self._initialized:
            return
        self._initialized = True
        self.refresh_from_history(api, driver_id)

    def refresh_from_history(self, api: SimulationApiPort, driver_id: str) -> None:
        """用评测会话内存中的真实动作记录重建状态，避免本地预估漂移。"""
        planner = self.monthly_planner
        self.total_income = 0.0
        self.total_mileage_km = 0.0
        self.total_deadhead_km = 0.0
        self.total_orders = 0
        self.orders_by_day = {}
        self.wait_intervals = []
        self.last_action_end_min = 0
        self.idle_days = set()
        self.active_days = set()
        self.spatial_income = {}
        self.total_tokens_used = 0
        self.completed_idle_days = 0
        self.monthly_planner = planner
        try:
            hist = api.query_decision_history(driver_id, -1)
            records = hist.get("records", [])
            if not records:
                return
            for rec in records:
                self._process_history_record(rec)
            # 恢复已完成的空闲日统计
            self._restore_idle_days_count()
        except Exception as e:
            _logger.warning("StateTracker 恢复历史失败: %s", e)

    def _restore_idle_days_count(self) -> None:
        """根据 active_days 恢复已完成的空闲日计数。"""
        if self.last_action_end_min <= 0:
            return
        current_day_idx = _get_day_index(self.last_action_end_min)
        idle_count = 0
        for d in range(current_day_idx):  # 不含当天（当天尚未结束，不能确定是否空闲）
            if d not in self.active_days:
                idle_count += 1
        self.completed_idle_days = idle_count

    def _process_history_record(self, rec: dict) -> None:
        action = rec.get("action", {})
        action_name = action.get("action", "")
        result = rec.get("result", {})
        sim_end = 0

        if "simulation_progress_minutes" in result:
            sim_end = int(result["simulation_progress_minutes"])
        elif "simulation_end_time" in rec:
            try:
                sim_end = _wall_to_sim_minutes(rec["simulation_end_time"] + ":00")
            except Exception:
                pass

        if action_name == "take_order" and result.get("accepted"):
            self.total_orders += 1
            step_elapsed = int(rec.get("step_elapsed_minutes", 0) or 0)
            query_cost = int(rec.get("query_scan_cost_minutes", 0) or 0)
            action_start = max(0, sim_end - step_elapsed + query_cost)
            day_idx = _get_day_index(action_start)
            self.orders_by_day[day_idx] = self.orders_by_day.get(day_idx, 0) + 1
            for d in _iter_days_touched(action_start, sim_end):
                self.active_days.add(d)
            # 记录空间信息
            pos_after = rec.get("position_after", {})
            if pos_after:
                self._update_spatial(pos_after.get("lat", 0), pos_after.get("lng", 0), 1.0)
            deadhead = result.get("pickup_deadhead_km", 0)
            self.total_deadhead_km += float(deadhead)
            haul = result.get("haul_distance_km", 0)
            self.total_mileage_km += float(deadhead) + float(haul)
            for key in ("price", "income", "revenue", "gross_income"):
                if key in result:
                    try:
                        self.total_income += float(result[key])
                        break
                    except Exception:
                        continue

        elif action_name == "wait":
            params = action.get("params", {})
            duration = int(params.get("duration_minutes", 0))
            if sim_end > 0 and duration > 0:
                wait_start = sim_end - duration
                self.wait_intervals.append((wait_start, sim_end))

        elif action_name == "reposition":
            step_elapsed = int(rec.get("step_elapsed_minutes", 0) or 0)
            query_cost = int(rec.get("query_scan_cost_minutes", 0) or 0)
            action_start = max(0, sim_end - step_elapsed + query_cost)
            for d in _iter_days_touched(action_start, sim_end):
                self.active_days.add(d)

        if sim_end > self.last_action_end_min:
            self.last_action_end_min = sim_end

        # Token 追踪
        token_usage = rec.get("token_usage", {})
        self.total_tokens_used += int(token_usage.get("total_tokens", 0))

    def _update_spatial(self, lat: float, lng: float, income: float) -> None:
        grid_key = (round(lat * 10), round(lng * 10))
        self.spatial_income[grid_key] = self.spatial_income.get(grid_key, 0) + income

    def record_take_order(self, sim_min: int, price: float, deadhead_km: float,
                          haul_km: float, end_lat: float, end_lng: float) -> None:
        self.total_orders += 1
        self.total_income += price
        self.total_deadhead_km += deadhead_km
        self.total_mileage_km += deadhead_km + haul_km
        day_idx = _get_day_index(sim_min)
        self.orders_by_day[day_idx] = self.orders_by_day.get(day_idx, 0) + 1
        self.active_days.add(day_idx)
        self._update_spatial(end_lat, end_lng, price)
        self.last_action_end_min = sim_min

    def record_wait(self, start_min: int, duration: int) -> None:
        self.wait_intervals.append((start_min, start_min + duration))
        self.last_action_end_min = start_min + duration

    def was_waiting_until(self, current_min: int) -> bool:
        return any(e == current_min for _, e in self.wait_intervals)

    def record_reposition(self, sim_min: int) -> None:
        day_idx = _get_day_index(sim_min)
        self.active_days.add(day_idx)
        self.last_action_end_min = sim_min

    def get_max_continuous_rest_today(self, current_min: int) -> int:
        """获取今天（到当前时刻）的最大连续休息分钟数（合并相邻区间）。"""
        day_idx = _get_day_index(current_min)
        day_start = day_idx * 1440
        day_end = current_min

        # 收集今天所有 wait 区间并裁剪
        today_intervals = []
        for s, e in self.wait_intervals:
            overlap_s = max(s, day_start)
            overlap_e = min(e, day_end)
            if overlap_e > overlap_s:
                today_intervals.append((overlap_s, overlap_e))

        if not today_intervals:
            return 0

        # 按起始时间排序后合并相邻/重叠区间
        today_intervals.sort()
        merged = [today_intervals[0]]
        for s, e in today_intervals[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:  # 重叠或紧邻
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))

        # 返回最长合并区间
        return max(e - s for s, e in merged)

    def get_current_rest_streak(self, current_min: int) -> int:
        """获取紧贴当前时刻的连续休息分钟数。"""
        day_idx = _get_day_index(current_min)
        day_start = day_idx * 1440
        intervals = []
        for s, e in self.wait_intervals:
            overlap_s = max(s, day_start)
            overlap_e = min(e, current_min)
            if overlap_e > overlap_s:
                intervals.append((overlap_s, overlap_e))

        if not intervals:
            return 0

        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))

        last_s, last_e = merged[-1]
        if last_e == current_min:
            return last_e - last_s
        return 0

    def get_orders_today(self, current_min: int) -> int:
        day_idx = _get_day_index(current_min)
        return self.orders_by_day.get(day_idx, 0)

    def get_full_idle_days_count(self, current_min: int) -> int:
        """已经完成的完整空闲天数（无 take_order 且无 reposition 的天）"""
        current_day = _get_day_index(current_min)
        idle_count = 0
        for d in range(current_day):
            if d not in self.active_days:
                idle_count += 1
        return idle_count

    def get_spatial_value(self, lat: float, lng: float) -> float:
        grid_key = (round(lat * 10), round(lng * 10))
        return self.spatial_income.get(grid_key, 0.0)

    def is_token_budget_exceeded(self) -> bool:
        return self.total_tokens_used >= _TOKEN_BUDGET * _TOKEN_DEGRADE_RATIO

    def record_cargo_seen(self, current_min: int, count: int) -> None:
        """记录某时段看到的货源数量，用于稀缺性评估。"""
        hour_slot = int(_get_hour_of_day(current_min))
        self.cargo_frequency[hour_slot] = self.cargo_frequency.get(hour_slot, 0) + count

    def get_avg_cargo_per_hour(self, current_min: int) -> float:
        """获取当前时段的历史平均货源数量。"""
        hour_slot = int(_get_hour_of_day(current_min))
        return float(self.cargo_frequency.get(hour_slot, 3))  # 默认中频

    def set_strategic_plan(self, plan: dict) -> None:
        """设置战略计划并初始化任务列表"""
        self.strategic_plan = plan
        self.open_tasks = list(plan.get("must_do_tasks", []))
        self.risk_register = list(plan.get("risk_windows", []))

    def update_task_progress(self, current_min: int, history: list[dict]) -> None:
        """根据历史记录刷新任务完成状态"""
        if not self.open_tasks:
            return
        newly_completed = []
        for task in self.open_tasks:
            task_type = task.get("type", "")
            if task_type == "mandatory_cargo":
                window_end = _to_int_or_zero(task.get("window_end"))
                if window_end > 0 and current_min > window_end:
                    task["expired"] = True
                    newly_completed.append(task)
                    continue
                # 检查历史中是否接了指定 cargo_id（嵌套结构: rec.action.params.cargo_id）
                target_cargo = str(task.get("target", ""))
                for h in history:
                    action = h.get("action", {})
                    if action.get("action") == "take_order":
                        cargo_id = str(action.get("params", {}).get("cargo_id", ""))
                        if cargo_id == target_cargo:
                            newly_completed.append(task)
                            break
            elif task_type == "scheduled_event":
                # 家事事件：保守完成判断
                # 必须能从历史看出"先到 pickup，再到 home"的顺序
                # 如果 release_min 有值，还要有 home 附近的 wait/position_after 覆盖到 release 后
                release_min = _to_int_or_zero(task.get("release_min"))
                home_deadline_min = _to_int_or_zero(task.get("home_deadline_min"))
                effective_deadline = release_min if release_min > 0 else home_deadline_min
                if effective_deadline > 0 and current_min < effective_deadline:
                    continue

                pickup_lat = task.get("pickup_lat")
                pickup_lng = task.get("pickup_lng")
                home_lat = task.get("home_lat")
                home_lng = task.get("home_lng")

                _PROXIMITY_DEG = 0.02  # ~1km

                def _first_visit_sim_min(t_lat, t_lng, radius=_PROXIMITY_DEG):
                    """返回第一次到达目标附近的 sim_min，未到达返回 -1"""
                    if t_lat is None or t_lng is None:
                        return -1
                    for h in history:
                        pos = h.get("position_after", {})
                        h_lat = pos.get("lat")
                        h_lng = pos.get("lng")
                        if h_lat is not None and h_lng is not None:
                            if (abs(float(h_lat) - float(t_lat)) < radius and
                                    abs(float(h_lng) - float(t_lng)) < radius):
                                result_h = h.get("result", {})
                                if "simulation_progress_minutes" in result_h:
                                    return int(result_h["simulation_progress_minutes"])
                    return -1

                pickup_visit_min = _first_visit_sim_min(pickup_lat, pickup_lng)
                home_visit_min = _first_visit_sim_min(home_lat, home_lng)

                # 必须到过两个点，且 pickup 在 home 之前
                if pickup_visit_min < 0 or home_visit_min < 0:
                    continue
                if home_visit_min <= pickup_visit_min:
                    continue

                # 如果 release_min 有值，需要 home 附近有覆盖 release 后的证据
                if release_min > 0:
                    home_covered_after_release = False
                    for h in history:
                        pos = h.get("position_after", {})
                        h_lat = pos.get("lat")
                        h_lng = pos.get("lng")
                        if h_lat is None or h_lng is None:
                            continue
                        if (home_lat is not None and home_lng is not None and
                                abs(float(h_lat) - float(home_lat)) < _PROXIMITY_DEG and
                                abs(float(h_lng) - float(home_lng)) < _PROXIMITY_DEG):
                            sim_min = 0
                            result_h = h.get("result", {})
                            if "simulation_progress_minutes" in result_h:
                                sim_min = int(result_h["simulation_progress_minutes"])
                            action = h.get("action", {})
                            duration = 0
                            if action.get("action") == "wait":
                                try:
                                    duration = int(action.get("params", {}).get("duration_minutes", 0))
                                except (ValueError, TypeError):
                                    duration = 0
                            start_of_record = sim_min - duration if duration > 0 else sim_min
                            if start_of_record >= release_min:
                                home_covered_after_release = True
                                break
                            if sim_min >= release_min:
                                home_covered_after_release = True
                                break
                    if not home_covered_after_release:
                        continue

                newly_completed.append(task)
            elif task_type == "monthly_visit":
                # 到访：需要在不同自然日到达指定点附近，达到 min_visit_days 才完成
                target_str = task.get("target", "")
                if "lat=" in target_str and "lng=" in target_str:
                    try:
                        parts = target_str.split(",")
                        t_lat = float(parts[0].split("=")[1])
                        t_lng = float(parts[1].split("=")[1])
                        min_days = task.get("min_visit_days", 1)
                        if isinstance(min_days, str):
                            min_days = int(min_days)
                        if min_days < 1:
                            min_days = 1
                        visited_days: set[int] = set()
                        for h in history:
                            pos = h.get("position_after", {})
                            h_lat = pos.get("lat")
                            h_lng = pos.get("lng")
                            if h_lat is not None and h_lng is not None:
                                if (abs(float(h_lat) - t_lat) < 0.05 and
                                        abs(float(h_lng) - t_lng) < 0.05):
                                    # 从历史记录推算 sim 时间
                                    sim_min = 0
                                    result_h = h.get("result", {})
                                    if "simulation_progress_minutes" in result_h:
                                        sim_min = int(result_h["simulation_progress_minutes"])
                                    elif "simulation_end_time" in h:
                                        try:
                                            sim_min = _wall_to_sim_minutes(h["simulation_end_time"] + ":00")
                                        except Exception:
                                            pass
                                    if sim_min > 0:
                                        day = sim_min // 1440
                                        visited_days.add(day)
                        if len(visited_days) >= min_days:
                            newly_completed.append(task)
                    except (ValueError, IndexError):
                        pass
            elif task_type == "day_off_requirement":
                # 复用 idle_days 统计
                min_days = task.get("min_days_off", 0)
                if isinstance(min_days, str):
                    try:
                        min_days = int(min_days)
                    except ValueError:
                        min_days = 0
                if self.completed_idle_days >= min_days:
                    newly_completed.append(task)

            elif task_type == "daily_home_deadline":
                # 回家deadline：这是每天重复的，不标记为完成
                # 但如果当天已回家（当前在家附近），可以暂时标记
                pass  # 每天重复，不从 open_tasks 移除

        for task in newly_completed:
            if task in self.open_tasks:
                self.open_tasks.remove(task)
                self.completed_tasks.append(task)

    def get_daily_intent(self, day_index: int) -> dict | None:
        """获取指定日的策略意图"""
        if self.strategic_plan and "daily_strategy" in self.strategic_plan:
            strategies = self.strategic_plan["daily_strategy"]
            for s in strategies:
                if s.get("day") == day_index:
                    return s
            # 尝试按索引获取
            if 0 < day_index <= len(strategies):
                return strategies[day_index - 1]
        return self.daily_intent

    def has_urgent_tasks(self, current_min: int) -> list[dict]:
        """返回距 deadline < 12小时(720分钟)的未完成紧急任务"""
        urgent = []
        try:
            horizon = max(
                int(t.get("deadline_minute", 0))
                for t in self.open_tasks
                if t.get("deadline_minute") is not None
            )
        except (ValueError, TypeError):
            horizon = _MONTH_TOTAL_MINUTES
        for task in self.open_tasks:
            task_type = task.get("type", "")
            if task_type == "daily_home_deadline":
                # daily_home_deadline 是每日重复约束，deadline_minute 是时间-of-day
                deadline_minute_of_day = int(task.get("deadline_minute", 23 * 60))
                current_minute_of_day = current_min % 1440
                time_left = deadline_minute_of_day - current_minute_of_day
                if 0 < time_left < 180:  # 距今日回家deadline不到3小时
                    urgent.append(task)
                continue
            if task_type == "mandatory_cargo":
                activation = _to_int_or_zero(task.get("activation_min"))
                window_end = _to_int_or_zero(task.get("window_end"))
                if window_end > 0 and current_min > window_end:
                    continue
                if activation > 0:
                    if current_min >= activation or 0 < activation - current_min < 180:
                        urgent.append(task)
                    continue
            if task_type == "scheduled_event":
                pickup_min = _to_int_or_zero(task.get("pickup_min"))
                home_deadline = _to_int_or_zero(task.get("home_deadline_min"))
                release_min = _to_int_or_zero(task.get("release_min")) or home_deadline
                if pickup_min > 0 and (
                    0 < pickup_min - current_min < 720
                    or pickup_min <= current_min < release_min
                ):
                    urgent.append(task)
                continue
            if task_type == "monthly_visit":
                min_days = _to_int_or_zero(task.get("min_visit_days")) or 1
                total_days = max(1, math.ceil(horizon / 1440))
                if total_days < min_days:
                    continue
                remaining_days = total_days - (_get_day_index(current_min) + 1)
                if remaining_days >= min_days:
                    continue
            deadline = task.get("deadline_minute")
            if deadline is not None:
                try:
                    deadline_val = int(deadline)
                    if 0 < deadline_val - current_min < 720:
                        urgent.append(task)
                except (ValueError, TypeError):
                    pass
        return urgent


# ─────────────────────────────────────────────────────────────────────────────
# PreferenceEngine - 偏好引擎
# ─────────────────────────────────────────────────────────────────────────────
class PreferenceEngine:
    """使用 LLM 将偏好文本解析为结构化约束，之后用代码检查。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._constraints_cache: dict[str, list[dict]] = {}  # driver_id -> constraints

    def get_constraints(self, driver_id: str, preferences: list[dict]) -> list[dict]:
        """获取结构化约束列表。首次调用 LLM 解析，后续直接返回缓存。"""
        signature = json.dumps(preferences, ensure_ascii=False, sort_keys=True, default=str)
        cache_key = f"{driver_id}:{signature}"
        if cache_key in self._constraints_cache:
            return self._constraints_cache[cache_key]

        constraints = self._parse_with_llm(preferences)
        constraints = self._merge_constraints(constraints, self._deterministic_parse(preferences))
        self._constraints_cache[cache_key] = constraints
        return constraints

    def _parse_with_llm(self, preferences: list[dict]) -> list[dict]:
        """调用 LLM 解析偏好文本为结构化约束。"""
        pref_texts = []
        for i, p in enumerate(preferences):
            content = p.get("content", "") if isinstance(p, dict) else str(p)
            penalty = p.get("penalty_amount", 0) if isinstance(p, dict) else 0
            cap = p.get("penalty_cap") if isinstance(p, dict) else None
            pref_texts.append({
                "index": i,
                "content": content,
                "penalty_amount": penalty,
                "penalty_cap": cap
            })

        prompt = json.dumps({
            "task": "将以下司机偏好约束解析为结构化JSON数组",
            "preferences": pref_texts,
            "output_format": {
                "constraints": [
                    {
                        "type": "约束类型(daily_rest/time_restriction/spatial_restrict/mileage_cap/day_off_requirement/time_window_location/daily_home_deadline/mandatory_cargo/scheduled_event/monthly_visit_requirement/cargo_category_ban/max_orders_per_day/max_haul_distance/max_deadhead_distance/custom)",
                        "description": "简要描述",
                        "params": "相关参数对象",
                        "penalty_amount": "单次违规罚款",
                        "penalty_cap": "罚款封顶(null则无封顶)",
                        "pref_index": "原始偏好索引"
                    }
                ]
            },
            "type_hints": {
                "daily_rest": {"min_continuous_minutes": "每天最少连续休息分钟数"},
                "time_restriction": {"start_hour": 0, "end_hour": 6, "forbidden_actions": ["take_order", "reposition"]},
                "spatial_restrict": {"type": "bounding_box或forbidden_circle或required_location", "详情字段": "..."},
                "mileage_cap": {"max_deadhead_km": "最大空驶里程"},
                "day_off_requirement": {"min_days_off": "月内最少休息天数"},
                "time_window_location": {"target_lat": 0, "target_lng": 0, "deadline_sim_min": 0, "description": "..."},
                "daily_home_deadline": {"home_lat": 0, "home_lng": 0, "deadline_hour": 23, "forbidden_start_hour": 23, "forbidden_end_hour": 8},
                "mandatory_cargo": {"cargo_id": "货源编号", "pickup_lat": 0, "pickup_lng": 0, "activation_min": "上架仿真分钟"},
                "scheduled_event": {"event_type": "family", "pickup_lat": 0, "pickup_lng": 0, "home_lat": 0, "home_lng": 0, "pickup_min": 0, "home_deadline_min": 0, "release_min": 0},
                "monthly_visit_requirement": {"min_visit_days": 5, "target_lat": 0, "target_lng": 0, "radius_km": 1},
                "cargo_category_ban": {"banned_categories": ["类别名"]},
                "max_orders_per_day": {"max_orders": 3},
                "max_haul_distance": {"max_km": 100},
                "max_deadhead_distance": {"max_km": 50}
            }
        }, ensure_ascii=False)

        try:
            resp = self._api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": "你是约束解析器。只输出JSON对象，不输出其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            })
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            constraints = parsed.get("constraints", [])
            if isinstance(constraints, list):
                return constraints
        except Exception as e:
            _logger.warning("LLM 解析偏好失败: %s", e)

        # 兜底：返回简单约束
        return self._fallback_parse(preferences)

    def _merge_constraints(self, parsed: list[dict], deterministic: list[dict]) -> list[dict]:
        """LLM 结果和确定性解析互补；确定性解析负责关键罚分项兜底。"""
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for c in list(deterministic or []) + list(parsed or []):
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type", "custom"))
            params = c.get("params", {}) if isinstance(c.get("params", {}), dict) else {}
            key_payload = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
            key = (ctype, key_payload)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _deterministic_parse(self, preferences: list[dict]) -> list[dict]:
        """用业务文本里的稳定模式解析高价值约束，减少对模型解析的依赖。"""
        constraints: list[dict] = []
        for i, p in enumerate(preferences):
            content = p.get("content", "") if isinstance(p, dict) else str(p)
            penalty = p.get("penalty_amount", 0) if isinstance(p, dict) else 0
            cap = p.get("penalty_cap") if isinstance(p, dict) else None
            base = {
                "penalty_amount": penalty,
                "penalty_cap": cap,
                "pref_index": i,
                "description": content[:80],
            }
            if isinstance(p, dict):
                base["visible_start_time"] = p.get("start_time")
                base["visible_end_time"] = p.get("end_time")

            mandatory = self._parse_mandatory_cargo(content, base)
            if mandatory:
                constraints.append(mandatory)

            scheduled = self._parse_family_event(content, base)
            if scheduled:
                constraints.append(scheduled)

            home = self._parse_daily_home_deadline(content, base)
            if home:
                constraints.append(home)

            visit = self._parse_monthly_visit_requirement(content, base)
            if visit:
                constraints.append(visit)

            banned = self._parse_banned_categories(content, base)
            if banned:
                constraints.append(banned)

            rest = self._parse_daily_rest(content, base)
            if rest:
                constraints.append(rest)

            days_off = self._parse_day_off(content, base)
            if days_off:
                constraints.append(days_off)

            time_block = self._parse_time_restriction(content, base)
            if time_block:
                constraints.append(time_block)

            deadhead_cap = self._parse_deadhead_cap(content, base)
            if deadhead_cap:
                constraints.append(deadhead_cap)

            haul_cap = self._parse_haul_cap(content, base)
            if haul_cap:
                constraints.append(haul_cap)

            max_orders = self._parse_max_orders_per_day(content, base)
            if max_orders:
                constraints.append(max_orders)

            spatial = self._parse_spatial_restriction(content, base)
            if spatial:
                constraints.append(spatial)

        return constraints

    def _parse_mandatory_cargo(self, content: str, base: dict) -> dict | None:
        if "熟货" not in content and "指定" not in content and "约定" not in content:
            return None
        match = re.search(r"(?:编号|cargo_id[=：:]?)\s*(\d+)", content, flags=re.I)
        if not match:
            return None
        coords = self._extract_coords(content)
        start_match = re.search(r"上架时间[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", content)
        params: dict[str, Any] = {"cargo_id": match.group(1)}
        if coords:
            params["pickup_lat"], params["pickup_lng"] = coords[0]
        if start_match:
            try:
                params["activation_min"] = _wall_to_sim_minutes(start_match.group(1))
            except Exception:
                pass
        return {**base, "type": "mandatory_cargo", "params": params, "severity": "hard"}

    def _parse_family_event(self, content: str, base: dict) -> dict | None:
        if "家事" not in content and "配偶" not in content:
            return None
        coords = self._extract_coords(content)
        if len(coords) < 2:
            return None
        pickup_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2}):(\d{2})", content)
        deadline_match = re.search(r"须在(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2}):(\d{2})前", content)
        release_matches = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2}):(\d{2})", content)
        if not pickup_match:
            return None

        def to_wall(m: re.Match[str] | tuple[str, ...]) -> str:
            groups = m.groups() if hasattr(m, "groups") else m
            y, mo, d, h, mi = [int(x) for x in groups]
            return f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:00"

        pickup_wall = to_wall(pickup_match)
        deadline_wall = to_wall(deadline_match) if deadline_match else pickup_wall
        release_wall = to_wall(release_matches[-1]) if release_matches else deadline_wall
        params = {
            "event_type": "family",
            "pickup_lat": coords[0][0],
            "pickup_lng": coords[0][1],
            "home_lat": coords[1][0],
            "home_lng": coords[1][1],
            "pickup_min": _wall_to_sim_minutes(pickup_wall),
            "home_deadline_min": _wall_to_sim_minutes(deadline_wall),
            "release_min": _wall_to_sim_minutes(release_wall),
            "pickup_wait_minutes": 10,
        }
        return {**base, "type": "scheduled_event", "params": params, "severity": "hard"}

    def _parse_daily_home_deadline(self, content: str, base: dict) -> dict | None:
        if "家" not in content or "点前" not in content:
            return None
        coords = self._extract_coords(content)
        if not coords:
            return None
        deadline = re.search(r"每天\s*(\d{1,2})点前", content)
        night = re.search(r"(\d{1,2})点至次日\s*(\d{1,2})点", content)
        params = {
            "home_lat": coords[0][0],
            "home_lng": coords[0][1],
            "deadline_hour": int(deadline.group(1)) if deadline else 23,
            "radius_km": 1.0,
        }
        if night:
            params["forbidden_start_hour"] = int(night.group(1))
            params["forbidden_end_hour"] = int(night.group(2))
        return {**base, "type": "daily_home_deadline", "params": params, "severity": "hard"}

    def _parse_monthly_visit_requirement(self, content: str, base: dict) -> dict | None:
        if "不同的自然日到过" not in content and "至少" not in content:
            return None
        coords = self._extract_coords(content)
        days = re.search(r"至少\s*(\d+)\s*个?不同的自然日", content)
        if not coords or not days:
            return None
        return {
            **base,
            "type": "monthly_visit_requirement",
            "params": {
                "min_visit_days": int(days.group(1)),
                "target_lat": coords[0][0],
                "target_lng": coords[0][1],
                "radius_km": 1.0,
            },
            "severity": "monthly",
        }

    def _parse_banned_categories(self, content: str, base: dict) -> dict | None:
        if "不接" not in content and "不拉" not in content and "尽量" not in content:
            return None
        cats = re.findall(r"「([^」]+)」", content)
        if not cats:
            return None
        # 区分硬禁和软偏好
        severity = "soft" if ("尽量" in content or "不太" in content or "最好" in content) else "hard"
        return {**base, "type": "cargo_category_ban", "params": {"banned_categories": cats}, "severity": severity}

    def _parse_daily_rest(self, content: str, base: dict) -> dict | None:
        if "休息" not in content and "歇" not in content and "停车" not in content:
            return None
        match = re.search(r"(?:满|至少|不少于)?\s*(\d+(?:\.\d+)?)\s*小时", content)
        if not match:
            return None
        minutes = int(float(match.group(1)) * 60)
        return {**base, "type": "daily_rest", "params": {"min_continuous_minutes": minutes}, "severity": "soft"}

    def _parse_day_off(self, content: str, base: dict) -> dict | None:
        if "自然月" not in content and "每月" not in content and "月内" not in content:
            return None
        if "天" not in content or ("不接单" not in content and "歇" not in content and "放空" not in content):
            return None
        match = re.search(r"至少(?:要有)?\s*(\d+)\s*个?(?:整天|天|日)", content)
        if not match:
            match = re.search(r"(\d+)\s*个?(?:整天|天|日).*?(?:不接单|歇|放空)", content)
        if match:
            days = int(match.group(1))
        else:
            cn_match = re.search(r"([一二两三四五六七八九十]+)\s*个?(?:整天|天|日).*?(?:不接单|歇|放空)", content)
            days = self._chinese_number(cn_match.group(1)) if cn_match else 0
        if days <= 0:
            return None
        return {**base, "type": "day_off_requirement", "params": {"min_days_off": days}, "severity": "monthly"}

    def _parse_time_restriction(self, content: str, base: dict) -> dict | None:
        if "不接单" not in content and "不空" not in content:
            return None
        match = re.search(r"(\d{1,2})[点:：](?:\d{2})?\s*(?:至|-|到)\s*(?:次日|翌日|上午|下午|早上|中午)?\s*(\d{1,2})[点:：]?", content)
        if not match:
            return None
        return {
            **base,
            "type": "time_restriction",
            "params": {
                "start_hour": int(match.group(1)),
                "end_hour": int(match.group(2)),
                "forbidden_actions": ["take_order", "reposition"],
            },
            "severity": "hard",
        }

    def _parse_deadhead_cap(self, content: str, base: dict) -> dict | None:
        if "空驶" not in content and "赴装货点" not in content:
            return None
        match = re.search(r"(?:不超过|≤|小于等于)\s*(\d+(?:\.\d+)?)\s*公里", content)
        if not match:
            return None
        max_km = float(match.group(1))
        ctype = "mileage_cap" if "月" in content and "空驶" in content else "max_deadhead_distance"
        key = "max_deadhead_km"
        return {**base, "type": ctype, "params": {key: max_km}, "severity": "hard"}

    def _parse_haul_cap(self, content: str, base: dict) -> dict | None:
        if "装货点至卸货点" not in content and "装卸距离" not in content:
            return None
        match = re.search(r"(?:不超过|≤|小于等于)\s*(\d+(?:\.\d+)?)\s*公里", content)
        if not match:
            return None
        return {**base, "type": "max_haul_distance", "params": {"max_km": float(match.group(1))}, "severity": "hard"}

    def _parse_spatial_restriction(self, content: str, base: dict) -> dict | None:
        if "深圳" in content and "范围" in content:
            lat_match = re.search(r"北纬\s*(\d+(?:\.\d+)?)\s*至\s*(\d+(?:\.\d+)?)", content)
            lng_match = re.search(r"东经\s*(\d+(?:\.\d+)?)\s*至\s*(\d+(?:\.\d+)?)", content)
            if lat_match and lng_match:
                return {
                    **base,
                    "type": "spatial_restrict",
                    "params": {
                        "type": "bounding_box",
                        "bounding_box": {
                            "lat_min": float(lat_match.group(1)),
                            "lat_max": float(lat_match.group(2)),
                            "lng_min": float(lng_match.group(1)),
                            "lng_max": float(lng_match.group(2)),
                        },
                    },
                    "severity": "hard",
                }
        if "不得进入" in content and "半径" in content:
            coords = self._extract_coords(content)
            radius = re.search(r"半径\s*(\d+(?:\.\d+)?)\s*公里", content)
            if coords and radius:
                return {
                    **base,
                    "type": "spatial_restrict",
                    "params": {
                        "type": "forbidden_circle",
                        "center_lat": coords[0][0],
                        "center_lng": coords[0][1],
                        "radius_km": float(radius.group(1)),
                    },
                    "severity": "hard",
                }
        return None

    def _parse_max_orders_per_day(self, content: str, base: dict) -> dict | None:
        if "同一天" not in content and "每天" not in content:
            return None
        if "接单" not in content or ("不得超过" not in content and "不超过" not in content and "≤" not in content):
            return None
        match = re.search(r"(?:不得超过|不超过|≤)\s*(\d+)\s*单", content)
        if not match:
            return None
        return {**base, "type": "max_orders_per_day", "params": {"max_orders": int(match.group(1))}, "severity": "hard"}

    @staticmethod
    def _extract_coords(text: str) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for a, b in re.findall(r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[，,]\s*(-?\d+(?:\.\d+)?)\s*[）)]", text):
            try:
                coords.append((float(a), float(b)))
            except Exception:
                continue
        return coords

    @staticmethod
    def _chinese_number(text: str) -> int:
        table = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9}
        if text == "十":
            return 10
        if text.startswith("十"):
            return 10 + table.get(text[1:], 0)
        if "十" in text:
            left, right = text.split("十", 1)
            return table.get(left, 0) * 10 + table.get(right, 0)
        return table.get(text, 0)

    def _fallback_parse(self, preferences: list[dict]) -> list[dict]:
        """无 LLM 时的简单规则匹配兜底。"""
        constraints = []
        for i, p in enumerate(preferences):
            content = p.get("content", "") if isinstance(p, dict) else str(p)
            penalty = p.get("penalty_amount", 0) if isinstance(p, dict) else 0
            cap = p.get("penalty_cap") if isinstance(p, dict) else None
            c = {
                "type": "custom",
                "description": content[:50],
                "params": {"raw_content": content},
                "penalty_amount": penalty,
                "penalty_cap": cap,
                "pref_index": i
            }
            # 尝试识别常见模式
            if "休息" in content and ("小时" in content or "分钟" in content):
                hours = self._extract_number(content, "小时")
                c["type"] = "daily_rest"
                c["params"] = {"min_continuous_minutes": int(hours * 60) if hours else 480}
            elif "不接单" in content and ("点" in content or "时" in content):
                c["type"] = "time_restriction"
                c["params"] = {"raw_content": content, "forbidden_actions": ["take_order", "reposition"]}
            elif "公里" in content and "空驶" in content:
                km = self._extract_number(content, "公里")
                c["type"] = "mileage_cap" if "月" in content else "max_deadhead_distance"
                c["params"] = {"max_deadhead_km": km or 100}
            elif ("整天" in content or "休息日" in content or "不接单" in content) and "天" in content:
                days = self._extract_number(content, "天") or self._extract_number(content, "个")
                c["type"] = "day_off_requirement"
                c["params"] = {"min_days_off": int(days) if days else 2}
            constraints.append(c)
        return constraints

    @staticmethod
    def _extract_number(text: str, after_word: str) -> float | None:
        try:
            idx = text.find(after_word)
            if idx <= 0:
                return None
            # 从 after_word 前面找数字
            num_str = ""
            for ch in reversed(text[:idx]):
                if ch.isdigit() or ch == '.':
                    num_str = ch + num_str
                elif num_str:
                    break
            return float(num_str) if num_str else None
        except Exception:
            return None

    def check_action_compliance(self, action_type: str, action_params: dict,
                                status: dict, state: StateTracker,
                                constraints: list[dict]) -> list[dict]:
        """检查动作是否违反约束。返回违规列表。"""
        violations = []
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = _get_hour_of_day(current_min)

        for c in constraints:
            ctype = c.get("type", "custom")
            params = c.get("params", {})
            penalty = c.get("penalty_amount", 0)

            if ctype == "time_restriction":
                if action_type in ("take_order", "reposition"):
                    start_h = params.get("start_hour")
                    end_h = params.get("end_hour")
                    if start_h is not None and end_h is not None:
                        if self._in_time_range(hour, float(start_h), float(end_h)):
                            violations.append({"constraint": c, "penalty": penalty})

            elif ctype == "spatial_restrict":
                if action_type == "reposition":
                    target_lat = action_params.get("latitude", 0)
                    target_lng = action_params.get("longitude", 0)
                    if params.get("type") == "bounding_box":
                        bb = params.get("bounding_box", {})
                        if not self._in_bounding_box(target_lat, target_lng, bb):
                            violations.append({"constraint": c, "penalty": penalty})
                    elif params.get("type") == "forbidden_circle":
                        cx = params.get("center_lat", 0)
                        cy = params.get("center_lng", 0)
                        r = params.get("radius_km", 20)
                        if haversine(target_lat, target_lng, cx, cy) < r:
                            violations.append({"constraint": c, "penalty": penalty})

            elif ctype == "max_orders_per_day":
                if action_type == "take_order":
                    max_o = int(params.get("max_orders", 99))
                    if state.get_orders_today(current_min) >= max_o:
                        violations.append({"constraint": c, "penalty": penalty})

            elif ctype == "max_deadhead_distance":
                if action_type == "take_order":
                    max_km = float(params.get("max_deadhead_km", 9999))
                    deadhead = action_params.get("deadhead_km", 0)
                    if deadhead > max_km:
                        violations.append({"constraint": c, "penalty": penalty})

            elif ctype == "max_haul_distance":
                if action_type == "take_order":
                    max_km = float(params.get("max_km", 9999))
                    haul = action_params.get("haul_km", 0)
                    if haul > max_km:
                        violations.append({"constraint": c, "penalty": penalty})

            elif ctype == "cargo_category_ban":
                if action_type == "take_order":
                    banned = params.get("banned_categories", [])
                    cargo_name = action_params.get("cargo_name", "")
                    for b in banned:
                        if b in cargo_name:
                            violations.append({"constraint": c, "penalty": penalty})
                            break

        return violations

    @staticmethod
    def _in_time_range(hour: float, start_h: float, end_h: float) -> bool:
        if start_h <= end_h:
            return start_h <= hour < end_h
        else:
            return hour >= start_h or hour < end_h

    @staticmethod
    def _in_bounding_box(lat: float, lng: float, bb: dict) -> bool:
        lat_min = bb.get("lat_min", -90)
        lat_max = bb.get("lat_max", 90)
        lng_min = bb.get("lng_min", -180)
        lng_max = bb.get("lng_max", 180)
        return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max


# ─────────────────────────────────────────────────────────────────────────────
# TimeWindowOptimizer - 时间窗可达性优化器 (Task #3)
# ─────────────────────────────────────────────────────────────────────────────
class TimeWindowOptimizer:
    """时间窗可达性优化器：预筛选不可达货源，标注时间风险。"""

    def prescreen_cargo_by_feasibility(self, candidates: list, status: dict,
                                       constraints: list, lead_time_minutes: int = 30) -> list:
        """筛选时间窗内可达的货源。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        feasible = []
        for item in candidates:
            cargo = item.get("cargo", {})
            load_time = cargo.get("load_time")
            distance_km = float(item.get("distance_km", 0))

            if load_time and isinstance(load_time, list) and len(load_time) == 2:
                # 计算到达时间（按60km/h估算）
                travel_minutes = _distance_to_minutes(distance_km)
                arrival_min = current_min + travel_minutes

                # 解析时间窗
                window_end = self._parse_time_window_end(load_time[1], current_min)

                if window_end and arrival_min + lead_time_minutes <= window_end:
                    feasible.append(item)
                elif not window_end:  # 无法解析时间窗的货物保留
                    feasible.append(item)
            else:
                feasible.append(item)  # 无时间窗信息的货物保留

        return feasible if feasible else candidates[:3]  # 保底返回Top3

    def _parse_time_window_end(self, window_end_value: Any, current_min: int) -> int | None:
        """解析时间窗结束值，返回仿真分钟数。"""
        if window_end_value is None:
            return None

        # 格式1: 完整时间字符串 "YYYY-MM-DD HH:MM:SS"
        if isinstance(window_end_value, str):
            window_end_str = window_end_value.strip()
            try:
                return _wall_to_sim_minutes(window_end_str)
            except Exception:
                pass
            # 格式2: "HH:MM" 格式
            try:
                parts = window_end_str.split(":")
                if len(parts) == 2:
                    h, m = int(parts[0]), int(parts[1])
                    # 当天对应的仿真分钟
                    day_start = (_get_day_index(current_min)) * 1440
                    return day_start + h * 60 + m
            except Exception:
                pass

        # 格式3: 直接是分钟数
        if isinstance(window_end_value, (int, float)):
            return int(window_end_value)

        return None

    def get_time_buffer_recommendation(self, status: dict, constraints: list) -> int:
        """获取建议缓冲时间（分钟）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = (current_min % 1440) / 60

        # 检查是否接近禁行时段
        for c in constraints:
            if isinstance(c, dict) and c.get("type") == "time_restriction":
                return 120

        if hour >= 21 or hour < 5:
            return 60
        return 30

    def annotate_time_risk(self, item: dict, status: dict) -> str:
        """为货物标注时间风险等级: LOW / MEDIUM / HIGH。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        cargo = item.get("cargo", {})
        load_time = cargo.get("load_time")
        distance_km = float(item.get("distance_km", 0))

        if not load_time or not isinstance(load_time, list) or len(load_time) < 2:
            return "LOW"  # 无时间窗限制

        window_end = self._parse_time_window_end(load_time[1], current_min)
        if window_end is None:
            return "LOW"

        travel_minutes = _distance_to_minutes(distance_km)
        arrival_min = current_min + travel_minutes
        buffer = window_end - arrival_min

        if buffer > 120:
            return "LOW"
        elif buffer > 30:
            return "MEDIUM"
        else:
            return "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# HistoryPatternAnalyzer - 历史决策模式分析器 (Task #4)
# ─────────────────────────────────────────────────────────────────────────────
class HistoryPatternAnalyzer:
    """历史决策模式分析器：按时段+货类聚合历史利润，供 LLM 参考。"""

    def __init__(self) -> None:
        # {(hour_bucket, cargo_type): [profit_values]}
        self.patterns: dict[tuple[int, str], list[float]] = {}
        # 历史决策记录列表
        self.decision_history: list[dict] = []

    def record_decision(self, hour: float, cargo_type: str, profit: float, action: str) -> None:
        """记录一次决策结果。"""
        bucket = int(hour // 4)  # 按4小时分段: 0-4, 4-8, 8-12, 12-16, 16-20, 20-24
        key = (bucket, cargo_type)
        if key not in self.patterns:
            self.patterns[key] = []
        self.patterns[key].append(profit)
        self.decision_history.append({
            "hour": hour, "type": cargo_type, "profit": profit, "action": action
        })

    def get_best_historical_pattern(self, current_hour: float, available_types: list) -> dict:
        """获取当前时段最佳历史模式。"""
        bucket = int(current_hour // 4)
        best = {"type": None, "avg_profit": 0, "count": 0}
        for cargo_type in available_types:
            key = (bucket, cargo_type)
            if key in self.patterns and self.patterns[key]:
                avg = sum(self.patterns[key]) / len(self.patterns[key])
                if avg > best["avg_profit"]:
                    best = {"type": cargo_type, "avg_profit": round(avg, 2),
                            "count": len(self.patterns[key])}
        return best

    def get_similar_scenarios(self, current_hour: float, num_candidates: int,
                              limit: int = 2) -> list:
        """找到历史上类似场景的决策。"""
        similar = []
        for record in self.decision_history[-50:]:  # 最近50条
            if abs(record["hour"] - current_hour) < 4:
                similar.append(record)
        return similar[-limit:]


# ─────────────────────────────────────────────────────────────────────────────
# OpportunityPredictor - 未来货源机会预测器 (Task #4)
# ─────────────────────────────────────────────────────────────────────────────
class OpportunityPredictor:
    """未来货源机会预测器：基于历史观察预测未来时段货源质量。"""

    def __init__(self) -> None:
        # {hour_bucket: {"total_count": int, "total_price": float, "observations": int}}
        self.hourly_cargo_stats: dict[int, dict] = {}

    def update_stats(self, hour: float, cargo_count: int, avg_price: float) -> None:
        """更新统计。"""
        bucket = int(hour // 2)  # 按2小时分段
        if bucket not in self.hourly_cargo_stats:
            self.hourly_cargo_stats[bucket] = {
                "total_count": 0, "total_price": 0.0, "observations": 0
            }
        stats = self.hourly_cargo_stats[bucket]
        stats["total_count"] += cargo_count
        stats["total_price"] += avg_price * cargo_count
        stats["observations"] += 1

    def predict_next_hours(self, current_hour: float, hours_ahead: int = 2) -> dict:
        """预测未来N小时的货源质量。"""
        predictions: dict = {}
        for h_offset in range(1, hours_ahead + 1):
            future_hour = (current_hour + h_offset) % 24
            bucket = int(future_hour // 2)
            if bucket in self.hourly_cargo_stats:
                stats = self.hourly_cargo_stats[bucket]
                obs = max(1, stats["observations"])
                predictions[f"+{h_offset}h"] = {
                    "expected_count": round(stats["total_count"] / obs, 1),
                    "expected_avg_price": round(
                        stats["total_price"] / max(1, stats["total_count"]), 1)
                }
            else:
                predictions[f"+{h_offset}h"] = {
                    "expected_count": 0, "expected_avg_price": 0
                }
        return predictions

    def should_wait_for_better(self, current_best_price: float, current_hour: float) -> bool:
        """判断是否应该等待更好的货源。"""
        if current_best_price <= 0:
            return False
        prediction = self.predict_next_hours(current_hour, 2)
        next_hour_data = prediction.get("+1h", {})
        if next_hour_data.get("expected_avg_price", 0) > current_best_price * 1.3:
            return True  # 未来1小时预期价格高30%以上，建议等待
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ProfitSearchLayer - 受约束短视野利润搜索层
# ─────────────────────────────────────────────────────────────────────────────
class ProfitSearchLayer:
    """在启发式 TopK 内重排候选，偏向长期利润而非单步收益。"""

    def rank_candidates(
        self,
        candidates: list[dict],
        status: dict,
        state: StateTracker,
        constraints: list[dict],
        opportunity_predictor: OpportunityPredictor | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)
        ranked: list[dict] = []

        for candidate in candidates:
            item = dict(candidate)
            base = float(item.get("true_net", item.get("score", 0.0)) or 0.0)
            downstream = self._downstream_value(
                item, current_min, horizon, state, opportunity_predictor)
            turnover = self._turnover_bonus(item)
            overnight_risk = self._overnight_long_order_risk_cost(item, current_min)
            search_score = base + downstream + turnover - overnight_risk
            item["downstream_value"] = round(downstream, 2)
            item["turnover_bonus"] = round(turnover, 2)
            item["overnight_risk_cost"] = round(overnight_risk, 2)
            item["profit_search_score"] = round(search_score, 4)
            ranked.append(item)

        ranked.sort(key=self._candidate_value, reverse=True)
        return ranked

    def select_best(self, candidates: list[dict]) -> dict | None:
        """按 profit_search_score 选择最佳候选，同时保留吃罚/长单/大空驶保护。"""
        if not candidates:
            return None

        safe = sorted(
            (c for c in candidates if not c.get("has_soft_penalty", False)),
            key=self._candidate_value,
            reverse=True,
        )
        penalty = sorted(
            (c for c in candidates if c.get("has_soft_penalty", False)),
            key=self._candidate_value,
            reverse=True,
        )

        best_safe = self._first_passing_guard(safe)
        best_penalty = self._first_passing_guard(penalty)

        chosen = None
        if best_penalty and best_safe:
            pn = self._candidate_value(best_penalty)
            sn = self._candidate_value(best_safe)
            chosen = best_penalty if pn > sn + _PENALTY_MARGIN else best_safe
        elif best_safe:
            chosen = best_safe
        elif best_penalty:
            chosen = best_penalty

        if chosen is None:
            return None

        return chosen

    @staticmethod
    def _candidate_value(candidate: dict) -> float:
        return float(candidate.get(
            "profit_search_score",
            candidate.get("true_net", candidate.get("score", 0.0)),
        ) or 0.0)

    @classmethod
    def _guard_value(cls, candidate: dict) -> float:
        true_net = float(candidate.get(
            "true_net", candidate.get("score", 0.0)) or 0.0)
        return max(cls._candidate_value(candidate), true_net)

    @classmethod
    def _passes_guard(cls, candidate: dict) -> bool:
        value = cls._guard_value(candidate)
        total_minutes = int(candidate.get("total_minutes", 0) or 0)
        deadhead_km = float(candidate.get("deadhead_km", 0.0) or 0.0)
        if total_minutes > 600 and value < _LONG_ORDER_REJECT_TRUE_NET:
            return False
        if deadhead_km > _EXTREME_DEADHEAD_KM and value < _DEADHEAD_REJECT_TRUE_NET:
            return False
        return True

    @classmethod
    def _first_passing_guard(cls, candidates: list[dict]) -> dict | None:
        for candidate in candidates:
            if cls._passes_guard(candidate):
                return candidate
        return None

    def _downstream_value(
        self,
        candidate: dict,
        current_min: int,
        horizon: int,
        state: StateTracker,
        opportunity_predictor: OpportunityPredictor | None,
    ) -> float:
        end = candidate.get("end", {})
        end_lat = end.get("lat")
        end_lng = end.get("lng")
        if end_lat is None or end_lng is None:
            return 0.0

        finish_min = current_min + int(candidate.get("total_minutes", 0) or 0)
        remaining_hours = max(0.0, (horizon - finish_min) / 60.0)
        if remaining_hours <= 0:
            return 0.0

        time_factor = min(1.0, remaining_hours / 4.0)
        spatial = self._spatial_value(float(end_lat), float(end_lng), state)
        forecast = self._forecast_value(
            _get_hour_of_day(finish_min), opportunity_predictor)
        return (spatial + forecast) * time_factor

    @staticmethod
    def _spatial_value(end_lat: float, end_lng: float, state: StateTracker) -> float:
        grid_key = (round(end_lat * 10), round(end_lng * 10))
        direct = float(state.spatial_income.get(grid_key, 0.0))
        neighbor = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor += float(state.spatial_income.get(
                    (grid_key[0] + dx, grid_key[1] + dy), 0.0))
        return min(240.0, direct * 0.015 + neighbor * 0.004)

    @staticmethod
    def _forecast_value(
        finish_hour: float,
        opportunity_predictor: OpportunityPredictor | None,
    ) -> float:
        if opportunity_predictor is None:
            return 0.0
        try:
            prediction = opportunity_predictor.predict_next_hours(finish_hour, 2)
        except Exception:
            return 0.0

        best = 0.0
        for data in prediction.values():
            try:
                expected_price = float(data.get("expected_avg_price", 0.0))
                expected_count = float(data.get("expected_count", 0.0))
            except (TypeError, ValueError):
                continue
            density = min(1.0, expected_count / 3.0)
            best = max(best, expected_price * density * 0.18)
        return min(180.0, best)

    @staticmethod
    def _turnover_bonus(candidate: dict) -> float:
        profit = max(0.0, float(candidate.get("net_profit", 0.0) or 0.0))
        minutes = max(1, int(candidate.get("total_minutes", 1) or 1))
        hourly_profit = profit * 60.0 / minutes
        return min(180.0, hourly_profit * 0.12)

    @staticmethod
    def _overnight_long_order_risk_cost(candidate: dict, current_min: int) -> float:
        """Price unknown next-day obligations when a late order blocks the morning."""
        total_minutes = int(candidate.get("total_minutes", 0) or 0)
        if total_minutes <= 600:
            return 0.0
        current_hour = _get_hour_of_day(current_min)
        if current_hour < 18.0:
            return 0.0
        current_day = _get_day_index(current_min)
        finish_min = current_min + total_minutes
        if _get_day_index(finish_min) <= current_day:
            return 0.0
        next_day_ten = (current_day + 1) * 1440 + 10 * 60
        if finish_min <= next_day_ten:
            return 0.0
        overrun_after_ten = max(0, finish_min - next_day_ten)
        long_extra = max(0, total_minutes - 600)
        return min(900.0, 180.0 + overrun_after_ten * 0.25 + long_extra * 0.15)


# ─────────────────────────────────────────────────────────────────────────────
# ProactiveRepositionLayer - 主动空驶策略层 (Task #5)
# ─────────────────────────────────────────────────────────────────────────────
class ProactiveRepositionLayer:
    """主动空驶策略层：在低谷或货源质量持续走低时，主动空驶至高价值区域。"""

    def __init__(self) -> None:
        # {(lat_bucket, lng_bucket): {"count": int, "total_price": float, "hours": {}}}
        self.region_heatmap: dict[tuple[float, float], dict] = {}
        # 最近N次决策的货源质量记录
        self.quality_history: list[dict] = []

    def update_heatmap(self, state: "StateTracker") -> None:
        """从 StateTracker 的 spatial_income 更新区域热力图。"""
        if not hasattr(state, 'spatial_income'):
            return
        for grid_key, income in state.spatial_income.items():
            # StateTracker 中 grid_key 是 (round(lat*10), round(lng*10))
            try:
                if isinstance(grid_key, tuple) and len(grid_key) == 2:
                    raw_lat, raw_lng = grid_key
                    # 还原成实际经纬度的 0.1 度精度
                    lat = raw_lat / 10.0
                    lng = raw_lng / 10.0
                else:
                    continue
            except Exception:
                continue
            bucket = (round(lat, 1), round(lng, 1))
            if bucket not in self.region_heatmap:
                self.region_heatmap[bucket] = {
                    "count": 0, "total_price": 0.0, "hours": {}
                }
            self.region_heatmap[bucket]["count"] += 1
            self.region_heatmap[bucket]["total_price"] += float(income)

    def record_cargo_quality(self, hour: float, candidates: list) -> None:
        """记录当前货源质量。"""
        if candidates:
            avg_price = sum(
                float(c.get("cargo", {}).get("price", 0)) for c in candidates
            ) / len(candidates)
        else:
            avg_price = 0.0
        self.quality_history.append({
            "hour": hour, "avg_price": avg_price, "count": len(candidates)
        })
        if len(self.quality_history) > 20:
            self.quality_history = self.quality_history[-20:]

    def should_reposition(self, status: dict, candidates: list, constraints: list) -> bool:
        """判断是否应该主动空驶。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = (current_min % 1440) / 60

        # 情景1: 深夜低谷期货源贫乏
        if (hour >= 22 or hour < 6) and len(candidates) < 2:
            return True

        # 情景2: 货源质量连续下降（最近3次均价递减）
        if len(self.quality_history) >= 3:
            recent = self.quality_history[-3:]
            if all(recent[i]["avg_price"] > recent[i + 1]["avg_price"] for i in range(2)):
                if recent[-1]["avg_price"] < 100:  # 当前均价低于100元
                    return True

        # 情景3: 当前无货源且不在深夜
        if len(candidates) == 0 and 6 <= hour < 22:
            return True

        return False

    def get_reposition_suggestion(self, status: dict, state: "StateTracker") -> dict | None:
        """获取空驶建议。"""
        if not self.region_heatmap:
            return None

        current_lat = float(status.get("current_lat", 0))
        current_lng = float(status.get("current_lng", 0))
        current_bucket = (round(current_lat, 1), round(current_lng, 1))

        # 找到最高价值且不是当前位置的区域
        best_region: dict | None = None
        best_value = 0.0
        for bucket, data in self.region_heatmap.items():
            if bucket == current_bucket:
                continue
            avg_price = data["total_price"] / max(1, data["count"])
            # 距离惩罚（越远价值越低）
            dist = ((bucket[0] - current_lat) ** 2 + (bucket[1] - current_lng) ** 2) ** 0.5
            distance_km = dist * 111  # 粗略转换
            travel_cost = distance_km * 1.5  # 每km 1.5元成本
            net_value = avg_price - travel_cost

            if net_value > best_value and distance_km < 200:  # 限制200km以内
                best_value = net_value
                best_region = {
                    "target_lat": bucket[0],
                    "target_lng": bucket[1],
                    "expected_value": round(avg_price, 1),
                    "distance_km": round(distance_km, 1),
                    "travel_cost": round(travel_cost, 1),
                    "net_value": round(net_value, 1),
                }

        return best_region


# ─────────────────────────────────────────────────────────────────────────────
# RuleLayer - 规则层（快速路径）
# ─────────────────────────────────────────────────────────────────────────────
class RuleLayer:
    """不调用 LLM 的快速路径决策。"""

    def evaluate(self, status: dict, state: StateTracker,
                 constraints: list[dict], items: list[dict] | None) -> dict | None:
        """返回决策 dict 或 None（需继续后续层）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = _get_hour_of_day(current_min)
        horizon = _get_horizon_minutes(status)
        remaining = horizon - current_min

        # 0. 最高优先级：临时约定、家事、每日回家等强约束。
        urgent_result = self._check_urgent_constraints(status, current_min, constraints, items, state)
        if urgent_result is not None:
            return urgent_result

        # 1. 月度规划空闲日检查（Task #1）
        idle_result = self._check_planned_idle_day(current_min, state, horizon)
        if idle_result is not None:
            return idle_result

        # 2. 月末收官：剩余不足 60 分钟无法完成任何订单
        if remaining <= 0:
            return self._wait(1)
        if remaining <= 60:
            wait_min = max(1, remaining)
            return self._wait(wait_min)

        # 3. 禁行时段检查
        block_result = self._check_time_block(hour, current_min, constraints, horizon)
        if block_result is not None:
            return block_result

        # 4. 强制休息检查
        rest_result = self._check_forced_rest(current_min, state, constraints, horizon)
        if rest_result is not None:
            return rest_result

        # 5. 窗口约束触发（必须前往指定位置）
        location_result = self._check_time_window_location(
            status, current_min, constraints)
        if location_result is not None:
            return location_result

        # 6. 深夜无货（22:00-06:00 且确认无有效货源）
        if items is not None and (hour >= 22.0 or hour < 6.0) and not items:
            # 等到早上6点
            if hour >= 22.0:
                wait_until = ((_get_day_index(current_min) + 1) * 1440) + 360  # 次日6:00
            else:
                wait_until = (_get_day_index(current_min) * 1440) + 360  # 当日6:00
            wait_min = max(1, min(wait_until - current_min, remaining))
            return self._wait(wait_min)

        return None

    def _check_urgent_constraints(self, status: dict, current_min: int,
                                  constraints: list[dict], items: list[dict] | None,
                                  state: "StateTracker | None" = None) -> dict | None:
        # 构建 open task 类型集合，只对未完成任务触发规则
        _open_types: set[str] = set()
        _open_mandatory_ids: set[str] = set()
        if state is not None:
            for t in state.open_tasks:
                tt = t.get("type", "")
                _open_types.add(tt)
                if tt == "mandatory_cargo":
                    tid = str(t.get("target") or t.get("cargo_id") or "")
                    if tid:
                        _open_mandatory_ids.add(tid)

        for c in constraints:
            ctype = c.get("type")
            if ctype == "scheduled_event":
                if "scheduled_event" not in _open_types and state is not None:
                    continue
                decision = self._check_scheduled_event(status, current_min, c)
                if decision:
                    return decision
            if ctype == "daily_home_deadline":
                decision = self._check_daily_home_deadline(status, current_min, c)
                if decision:
                    return decision
            if ctype == "monthly_visit_requirement":
                if "monthly_visit" not in _open_types and state is not None:
                    continue
                decision = self._check_monthly_visit_requirement(status, current_min, c)
                if decision:
                    return decision
            if ctype == "mandatory_cargo":
                cargo_id = str(c.get("params", {}).get("cargo_id", ""))
                if state is not None and cargo_id not in _open_mandatory_ids:
                    continue
                decision = self._check_mandatory_cargo(status, current_min, c, items)
                if decision:
                    return decision
        return None

    def _check_mandatory_cargo(self, status: dict, current_min: int,
                               constraint: dict, items: list[dict] | None) -> dict | None:
        params = constraint.get("params", {})
        cargo_id = str(params.get("cargo_id", "")).strip()
        if not cargo_id:
            return None
        try:
            activation = int(params.get("activation_min", current_min))
        except (ValueError, TypeError):
            activation = current_min
        pickup_lat = params.get("pickup_lat")
        pickup_lng = params.get("pickup_lng")
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))

        if pickup_lat is not None and pickup_lng is not None and current_min < activation:
            dist = haversine(lat, lng, float(pickup_lat), float(pickup_lng))
            travel_min = _distance_to_minutes(dist)
            if current_min + travel_min + 30 >= activation and dist > 1.0:
                return self._reposition(float(pickup_lat), float(pickup_lng))
            if activation - current_min <= 120:
                return self._wait(min(60, activation - current_min))
            return None

        if current_min >= activation:
            if items is None:
                return {"action": "take_order", "params": {"cargo_id": cargo_id}}
            visible_ids = {str((it.get("cargo") or {}).get("cargo_id", "")) for it in items}
            if cargo_id in visible_ids:
                return {"action": "take_order", "params": {"cargo_id": cargo_id}}
        return None

    def _check_scheduled_event(self, status: dict, current_min: int, constraint: dict) -> dict | None:
        params = constraint.get("params", {})
        pickup_min = int(params.get("pickup_min", 10**9))
        deadline = int(params.get("home_deadline_min", pickup_min))
        release = int(params.get("release_min", deadline))
        pickup_lat = params.get("pickup_lat")
        pickup_lng = params.get("pickup_lng")
        home_lat = params.get("home_lat")
        home_lng = params.get("home_lng")
        if None in (pickup_lat, pickup_lng, home_lat, home_lng):
            return None
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        if current_min < pickup_min:
            dist = haversine(lat, lng, float(pickup_lat), float(pickup_lng))
            travel_min = _distance_to_minutes(dist)
            pickup_wait = int(params.get("pickup_wait_minutes", 10))
            prep_buffer = max(15, min(60, pickup_wait + 5))
            if dist > 1.0 and current_min + travel_min + prep_buffer >= pickup_min:
                return self._reposition(float(pickup_lat), float(pickup_lng))
            if dist <= 1.0 and pickup_min - current_min <= prep_buffer:
                return self._wait(min(60, pickup_min - current_min))
            return None

        if pickup_min <= current_min < deadline:
            pickup_dist = haversine(lat, lng, float(pickup_lat), float(pickup_lng))
            if pickup_dist > 1.0:
                return self._reposition(float(pickup_lat), float(pickup_lng))
            if pickup_dist <= 1.0 and current_min < pickup_min + int(params.get("pickup_wait_minutes", 10)):
                return self._wait(int(params.get("pickup_wait_minutes", 10)))
            home_dist = haversine(lat, lng, float(home_lat), float(home_lng))
            if home_dist > 1.0:
                return self._reposition(float(home_lat), float(home_lng))
            return self._wait(min(60, max(1, deadline - current_min)))

        if deadline <= current_min < release:
            home_dist = haversine(lat, lng, float(home_lat), float(home_lng))
            if home_dist > 1.0:
                return self._reposition(float(home_lat), float(home_lng))
            return self._wait(min(480, max(1, release - current_min)))
        return None

    def _check_daily_home_deadline(self, status: dict, current_min: int, constraint: dict) -> dict | None:
        params = constraint.get("params", {})
        home_lat = params.get("home_lat")
        home_lng = params.get("home_lng")
        if home_lat is None or home_lng is None:
            return None
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        hour = _get_hour_of_day(current_min)
        day_start = _get_day_index(current_min) * 1440
        deadline_hour = float(params.get("deadline_hour", 23))
        deadline_min = day_start + int(deadline_hour * 60)
        dist = haversine(lat, lng, float(home_lat), float(home_lng))
        radius = float(params.get("radius_km", 1.0))
        start_h = params.get("forbidden_start_hour")
        end_h = params.get("forbidden_end_hour")

        if start_h is not None and end_h is not None and PreferenceEngine._in_time_range(hour, float(start_h), float(end_h)):
            if dist > radius:
                return self._reposition(float(home_lat), float(home_lng))
            wait_min = self._minutes_until_hour(hour, float(end_h))
            horizon = _get_horizon_minutes(status)
            return self._wait(min(wait_min, max(1, horizon - current_min)))

        travel_min = _distance_to_minutes(dist)
        if dist > radius and current_min + travel_min + 30 >= deadline_min and current_min < deadline_min:
            return self._reposition(float(home_lat), float(home_lng))
        # 已在家且非禁行时段：不拦截，继续搜索
        return None

    def _check_monthly_visit_requirement(self, status: dict, current_min: int, constraint: dict) -> dict | None:
        params = constraint.get("params", {})
        target_lat = params.get("target_lat")
        target_lng = params.get("target_lng")
        if target_lat is None or target_lng is None:
            return None
        horizon = _get_horizon_minutes(status)
        total_days = max(1, math.ceil(horizon / 1440))
        current_day = _get_day_index(current_min) + 1
        required = int(params.get("min_visit_days", 0))
        if required <= 0:
            return None
        # 均匀安排访问日；靠近月末时兜底去目标点。
        planned_days = {max(1, min(total_days, round((i + 1) * total_days / required))) for i in range(required)}
        if current_day not in planned_days and total_days - current_day >= required:
            return None
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        dist = haversine(lat, lng, float(target_lat), float(target_lng))
        if dist > float(params.get("radius_km", 1.0)):
            return self._reposition(float(target_lat), float(target_lng))
        return None

    @staticmethod
    def _minutes_until_hour(current_hour: float, target_hour: float) -> int:
        if current_hour <= target_hour:
            return max(1, int((target_hour - current_hour) * 60))
        return max(1, int((24 - current_hour + target_hour) * 60))

    def _check_time_block(self, hour: float, current_min: int,
                          constraints: list[dict], horizon: int) -> dict | None:
        """检查是否在禁行时段。"""
        for c in constraints:
            if c.get("type") != "time_restriction":
                continue
            params = c.get("params", {})
            start_h = params.get("start_hour")
            end_h = params.get("end_hour")
            if start_h is None or end_h is None:
                continue
            start_h, end_h = float(start_h), float(end_h)
            if PreferenceEngine._in_time_range(hour, start_h, end_h):
                # 等到禁行结束
                if start_h > end_h:  # 跨午夜
                    if hour >= start_h:
                        wait_min = int((24 - hour + end_h) * 60)
                    else:
                        wait_min = int((end_h - hour) * 60)
                else:
                    wait_min = int((end_h - hour) * 60)
                wait_min = max(1, min(wait_min + 1, horizon - current_min))
                return self._wait(wait_min)
        return None

    def _check_forced_rest(self, current_min: int, state: StateTracker,
                           constraints: list[dict], horizon: int) -> dict | None:
        """检查今天是否满足了最低连续休息要求。

        已经处于连续休息段时会继续补足；否则只在当天剩余时间刚好
        只够重新形成完整休息段时才强制 wait。
        """
        for c in constraints:
            if c.get("type") != "daily_rest":
                continue
            params = c.get("params", {})
            required_min = int(params.get("min_continuous_minutes", 0))
            if required_min <= 0:
                continue
            current_rest = state.get_max_continuous_rest_today(current_min)
            if current_rest >= required_min:
                continue  # 已满足，无需强制
            current_streak = state.get_current_rest_streak(current_min)
            remaining_total = horizon - current_min
            remaining_today = 1440 - (current_min % 1440)
            carried_into_new_day = (
                current_min > 0
                and current_min % 1440 == 0
                and state.was_waiting_until(current_min)
            )
            minute_of_day = current_min % 1440

            # 已经在连续休息段中时，优先一口气补足，避免查询货源打断连续休息。
            if current_streak > 0 or carried_into_new_day:
                need = required_min - current_streak
                if need > 0:
                    wait_min = max(1, min(need, remaining_today, remaining_total))
                    return self._wait(wait_min)

            # 清晨先补完整休息块，避免白天多次短 wait 被查询耗时切碎。
            if minute_of_day < 6 * 60:
                wait_min = max(1, min(required_min, remaining_today, remaining_total))
                return self._wait(wait_min)

            # 仅在剩余时间刚好只够重新形成完整连续休息段时才强制。
            if remaining_today <= required_min + 30:
                wait_min = max(1, min(required_min, remaining_today, remaining_total))
                return self._wait(wait_min)
        return None

    def _check_time_window_location(self, status: dict, current_min: int,
                                    constraints: list[dict]) -> dict | None:
        """检查是否有即将到期的位置窗口约束。"""
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        for c in constraints:
            if c.get("type") != "time_window_location":
                continue
            params = c.get("params", {})
            target_lat = params.get("target_lat")
            target_lng = params.get("target_lng")
            deadline = params.get("deadline_sim_min")
            if target_lat is None or target_lng is None or deadline is None:
                continue
            target_lat, target_lng = float(target_lat), float(target_lng)
            deadline = int(deadline)
            dist = haversine(lat, lng, target_lat, target_lng)
            travel_min = _distance_to_minutes(dist)
            # 如果快到 deadline 且还没到位置
            if dist > 1.0 and current_min + travel_min + 30 >= deadline:
                return self._reposition(target_lat, target_lng)
        return None

    def _check_planned_idle_day(self, current_min: int, state: StateTracker, horizon: int) -> dict | None:
        """检查今日是否需要强制空闲（仅月末补救时拦截，其余交给评分层）。"""
        planner = state.monthly_planner
        if planner is None or planner.required_idle_days <= 0:
            return None

        current_day = _get_day_index(current_min) + 1
        total_days = max(1, math.ceil(horizon / 1440))
        remaining = horizon - current_min

        # 短期仿真观测模式：horizon_days < required + 2 时不主动停工
        if total_days < planner.required_idle_days + 2:
            return None

        # 仅月末强制补救：剩余天数 <= 剩余需求时才拦截
        remaining_required = max(0, planner.required_idle_days - state.completed_idle_days)
        remaining_days = total_days - current_day
        if remaining_required > 0 and remaining_days <= remaining_required:
            # 月末强制补救 - 今天必须空闲
            minutes_in_day = current_min % 1440
            wait_until_next_day = 1440 - minutes_in_day
            wait_min = max(1, min(wait_until_next_day, remaining))
            _logger.info("月末强制补救: day=%d, remaining_required=%d, wait=%d min",
                         current_day, remaining_required, wait_min)
            return self._wait(wait_min)

        # 其余情况交给启发式层通过评分影响决策
        return None

    @staticmethod
    def _wait(minutes: int) -> dict:
        return {"action": "wait", "params": {"duration_minutes": max(1, int(minutes))}}

    @staticmethod
    def _reposition(lat: float, lng: float) -> dict:
        return {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}


# ─────────────────────────────────────────────────────────────────────────────
# HeuristicLayer - 启发式评分层 (Task #2 优化)
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicLayer:
    """对候选货源评分并排序，返回 Top-N。"""

    def __init__(self, cost_per_km: float = _COST_PER_KM_DEFAULT) -> None:
        self.cost_per_km = cost_per_km

    def _get_time_phase_multiplier(self, month_progress_ratio: float) -> float:
        """根据月度进度返回时间效率系数（Task #2）。"""
        if month_progress_ratio < 0.3:
            return 1.0  # 月初保守
        elif month_progress_ratio < 0.7:
            return 1.2  # 月中进取
        else:
            return 0.9  # 月末保险

    def _assess_preference_impact(self, item: dict, state: StateTracker,
                                  constraints: list) -> float:
        """评估接单对偏好达成的影响，返回0-1（0=无影响，1=严重影响）（Task #2）。"""
        if state.monthly_planner is None or state.monthly_planner.required_idle_days <= 0:
            return 0.0

        planner = state.monthly_planner
        # 获取当前进度
        current_min = state.last_action_end_min
        current_day = _get_day_index(current_min) + 1

        progress = planner.get_progress_report(state.completed_idle_days, current_day)
        remaining_required = progress["remaining_required"]
        remaining_available = progress["remaining_available_days"]

        if remaining_required <= 0:
            return 0.0

        # 计算接单后的时间消耗占比
        total_minutes = int(item.get("total_minutes", 0) or item.get("cost_time_minutes", 60))
        days_consumed = total_minutes / 1440.0

        # 如果剩余时间紧张，接单影响更大
        if remaining_available <= 0:
            return 1.0
        urgency_ratio = remaining_required / remaining_available
        impact = min(1.0, urgency_ratio * days_consumed * 2.0)
        return impact

    def _get_scarcity_factor(self, item: dict, state: StateTracker) -> float:
        """评估该货源的稀缺性加分（Task #2）。"""
        current_min = state.last_action_end_min
        avg_cargo = state.get_avg_cargo_per_hour(current_min)

        # 低频(<1条/小时): +50
        # 中频(1-5条): +20
        # 高频(>5条): 0
        if avg_cargo < 1:
            return 50.0
        elif avg_cargo <= 5:
            return 20.0
        else:
            return 0.0

    def _compute_cluster_value(self, end_lat: float, end_lng: float,
                               state: StateTracker) -> float:
        """计算终点位置的聚类价值（替代原spatial_bonus）。"""
        # 基于空间记忆的历史收益密度
        spatial_value = state.get_spatial_value(end_lat, end_lng)
        # 考虑周围网格的收益（扩大搜索范围）
        neighbor_value = 0.0
        grid_key = (round(end_lat * 10), round(end_lng * 10))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
                neighbor_value += state.spatial_income.get(neighbor_key, 0.0)
        # 聚类值 = 当前格收益 + 邻近格平均收益
        cluster_value = spatial_value * 0.01 + neighbor_value * 0.005
        return min(cluster_value, 80.0)  # 封顶80

    @staticmethod
    def _urgency_weight(slack_minutes: int) -> float:
        if slack_minutes < 0:
            return 8.0
        if slack_minutes <= 180:
            return 4.0
        if slack_minutes <= 720:
            return 2.0
        if slack_minutes <= 1440:
            return 1.0
        return 0.25

    def _target_position_cost(
        self,
        end_lat: float,
        end_lng: float,
        target_lat: float,
        target_lng: float,
        finish_min: int,
        deadline_min: int,
        penalty_amount: float = 0.0,
    ) -> float:
        dist = haversine(end_lat, end_lng, target_lat, target_lng)
        travel_min = _distance_to_minutes(dist)
        slack = deadline_min - finish_min - travel_min
        move_cost = dist * self.cost_per_km + travel_min * 0.4
        cost = move_cost * self._urgency_weight(slack)
        if slack < 0:
            cost += penalty_amount
        return cost

    def _violates_daily_home_deadline(
        self,
        end_lat: float,
        end_lng: float,
        finish_min: int,
        current_min: int,
        constraint: dict,
    ) -> bool:
        params = constraint.get("params", {})
        home_lat = params.get("home_lat")
        home_lng = params.get("home_lng")
        if home_lat is None or home_lng is None:
            return False
        try:
            deadline_hour = int(params.get("deadline_hour", 23))
            deadline_min = _get_day_index(current_min) * 1440 + deadline_hour * 60
            if current_min >= deadline_min:
                return False
            travel_home_min = _distance_to_minutes(
                haversine(end_lat, end_lng, float(home_lat), float(home_lng))
            )
            return finish_min + travel_home_min > deadline_min
        except (TypeError, ValueError):
            return False

    def _future_position_cost(
        self,
        end_lat: float,
        end_lng: float,
        finish_min: int,
        current_min: int,
        constraints: list[dict],
        state: StateTracker,
        horizon: int,
        mandatory_pickups: list[dict],
    ) -> float:
        cost = 0.0
        finish_day = _get_day_index(finish_min)
        day_end = (finish_day + 1) * 1440

        for c in constraints:
            ctype = c.get("type")
            p = c.get("params", {})
            penalty_amount = float(c.get("penalty_amount", 0) or 0)

            if ctype == "daily_home_deadline":
                home_lat = p.get("home_lat")
                home_lng = p.get("home_lng")
                if home_lat is None or home_lng is None:
                    continue
                try:
                    deadline_hour = int(p.get("deadline_hour", 23))
                    deadline_min = _get_day_index(current_min) * 1440 + deadline_hour * 60
                    if current_min < deadline_min:
                        cost += self._target_position_cost(
                            end_lat, end_lng, float(home_lat), float(home_lng),
                            finish_min, deadline_min, penalty_amount,
                        )
                except (TypeError, ValueError):
                    continue

            elif ctype == "scheduled_event":
                has_open_event = any(t.get("type") == "scheduled_event" for t in state.open_tasks)
                if not has_open_event:
                    continue
                pickup_lat = p.get("pickup_lat")
                pickup_lng = p.get("pickup_lng")
                home_lat = p.get("home_lat")
                home_lng = p.get("home_lng")
                pickup_min = _to_int_or_zero(p.get("pickup_min"))
                home_deadline = _to_int_or_zero(p.get("home_deadline_min"))
                release_min = _to_int_or_zero(p.get("release_min"))
                if pickup_lat is not None and pickup_lng is not None and pickup_min > finish_min:
                    if pickup_min - finish_min <= 2 * 1440:
                        cost += self._target_position_cost(
                            end_lat, end_lng, float(pickup_lat), float(pickup_lng),
                            finish_min, pickup_min, penalty_amount,
                        )
                if home_lat is not None and home_lng is not None:
                    target_deadline = home_deadline if home_deadline > finish_min else release_min
                    if target_deadline > finish_min and target_deadline - finish_min <= 2 * 1440:
                        cost += self._target_position_cost(
                            end_lat, end_lng, float(home_lat), float(home_lng),
                            finish_min, target_deadline, penalty_amount,
                        )

            elif ctype == "monthly_visit_requirement":
                has_open_visit = any(t.get("type") == "monthly_visit" for t in state.open_tasks)
                if not has_open_visit:
                    continue
                target_lat = p.get("target_lat")
                target_lng = p.get("target_lng")
                if target_lat is None or target_lng is None:
                    continue
                days_left = max(1, math.ceil(max(0, horizon - finish_min) / 1440))
                try:
                    required = int(p.get("min_visit_days", 1))
                except (TypeError, ValueError):
                    required = 1
                deadline = day_end if days_left <= required + 1 else min(horizon, finish_min + 1440)
                cost += self._target_position_cost(
                    end_lat, end_lng, float(target_lat), float(target_lng),
                    finish_min, deadline, penalty_amount,
                ) * (1.5 if days_left <= required + 1 else 0.35)

        for mp in mandatory_pickups:
            activation = int(mp.get("activation_min", 0))
            if activation <= finish_min or activation - finish_min > 2 * 1440:
                continue
            cost += self._target_position_cost(
                end_lat, end_lng, float(mp["pickup_lat"]), float(mp["pickup_lng"]),
                finish_min, int(mp.get("window_end", activation)), 0.0,
            )

        # 上限封顶：防止远距离终点把评分打爆
        return min(cost, 300.0)

    @staticmethod
    def _build_mandatory_pickup_info(
        state: StateTracker, constraints: list[dict],
        items: list[dict] | None = None,
    ) -> list[dict]:
        """从 open_tasks + constraints 提取 mandatory_cargo 的 pickup 信息。

        返回列表，每项: {cargo_id, activation_min, window_end, pickup_lat, pickup_lng}
        仅包含有 activation_min + pickup 坐标的 mandatory 任务。
        """
        # 从 constraints 建 cargo_id -> params 索引
        cargo_params: dict[str, dict] = {}
        for c in constraints:
            if c.get("type") == "mandatory_cargo":
                params = c.get("params", {})
                cid = str(params.get("cargo_id", ""))
                if cid:
                    cargo_params[cid] = params

        # 从 items 建 cargo_id -> load_time 索引（用于取装货窗结束时间）
        cargo_window_end: dict[str, int] = {}
        if items:
            for it in items:
                cargo = it.get("cargo", {})
                cid = str(cargo.get("cargo_id", ""))
                load_time = cargo.get("load_time")
                if cid and load_time and isinstance(load_time, list) and len(load_time) >= 2:
                    try:
                        cargo_window_end[cid] = _wall_to_sim_minutes(str(load_time[1]).strip())
                    except Exception:
                        pass

        result: list[dict] = []
        open_tasks = list(getattr(state, "open_tasks", []) or [])
        completed_tasks = list(getattr(state, "completed_tasks", []) or [])

        completed_ids = {
            str(t.get("target") or t.get("cargo_id") or "").strip()
            for t in completed_tasks
            if t.get("type") == "mandatory_cargo"
        }
        completed_ids.discard("")

        open_ids = {
            str(t.get("target") or t.get("cargo_id") or "").strip()
            for t in open_tasks
            if t.get("type") == "mandatory_cargo"
        }
        open_ids.discard("")

        def _make_entry(cargo_id: str, params: dict, task: dict | None = None) -> dict | None:
            task = task or {}
            activation_min = params.get("activation_min") or task.get("activation_min")
            pickup_lat = params.get("pickup_lat") or task.get("pickup_lat")
            pickup_lng = params.get("pickup_lng") or task.get("pickup_lng")
            if activation_min is None or pickup_lat is None or pickup_lng is None:
                return None
            try:
                entry = {
                    "cargo_id": cargo_id,
                    "activation_min": int(activation_min),
                    "pickup_lat": float(pickup_lat),
                    "pickup_lng": float(pickup_lng),
                }
            except (ValueError, TypeError):
                return None

            window_end = (
                params.get("window_end")
                or task.get("window_end")
                or cargo_window_end.get(cargo_id)
            )
            if window_end is not None:
                try:
                    entry["window_end"] = int(window_end)
                except (ValueError, TypeError):
                    pass
            return entry

        for task in open_tasks:
            if task.get("type") != "mandatory_cargo":
                continue
            cargo_id = str(task.get("target") or task.get("cargo_id") or "").strip()
            if not cargo_id or cargo_id in completed_ids:
                continue
            entry = _make_entry(cargo_id, cargo_params.get(cargo_id, {}), task)
            if entry is not None:
                result.append(entry)

        included_ids = {str(e.get("cargo_id", "")) for e in result}
        for cargo_id, params in cargo_params.items():
            if cargo_id in completed_ids or cargo_id in included_ids:
                continue
            if open_ids and cargo_id not in open_ids:
                continue
            entry = _make_entry(cargo_id, params)
            if entry is not None:
                result.append(entry)

        return result

    # ── true_net 分类与估算方法 ──

    @staticmethod
    def classify_constraint_severity(constraint: dict) -> str:
        """返回约束的实际严重级别 'hard' 或 'soft'。

        Hard: mandatory_cargo, scheduled_event, time_restriction,
              spatial_restrict(forbidden_circle),
              daily_home_deadline 仅当 severity=hard 且 penalty>=500。
        Soft: daily_rest, day_off_requirement, monthly_visit_requirement,
              daily_home_deadline(默认), soft cargo_category_ban。
        """
        ctype = constraint.get("type", "")
        severity = constraint.get("severity", "hard")

        if ctype in ("mandatory_cargo", "scheduled_event", "time_restriction"):
            return "hard"
        if ctype == "spatial_restrict":
            params = constraint.get("params", {})
            if params.get("type") == "forbidden_circle":
                return "hard"
            return "hard" if severity == "hard" else "soft"
        if ctype == "daily_home_deadline":
            penalty = float(constraint.get("penalty_amount", 0) or 0)
            if severity == "hard" and penalty >= 500:
                return "hard"
            return "soft"
        if ctype in ("daily_rest", "day_off_requirement", "monthly_visit_requirement"):
            return "soft"
        if ctype == "cargo_category_ban":
            return "soft" if severity == "soft" else "hard"
        return severity if severity in ("hard", "soft") else "hard"

    def _estimate_soft_penalty_cost(
        self,
        constraint: dict,
        cargo_name: str,
        distance_km: float,
        haul_km: float,
        end_lat: float,
        end_lng: float,
        finish_min: int,
        current_min: int,
        state: StateTracker,
    ) -> float:
        """估算违反软约束的经济成本（元），从 true_net 中扣除。"""
        ctype = constraint.get("type", "")
        p_amount = float(constraint.get("penalty_amount", 0) or 0)
        params = constraint.get("params", {})

        if ctype == "daily_rest":
            required_min = int(params.get("min_continuous_minutes", 0))
            if required_min <= 0:
                return 0.0
            current_rest = state.get_max_continuous_rest_today(current_min)
            if current_rest >= required_min:
                return 0.0
            remaining_after = 1440 - (finish_min % 1440)
            need = required_min - current_rest
            if remaining_after < need:
                return max(p_amount, 300.0)
            return 0.0

        if ctype == "daily_home_deadline":
            if p_amount > 0:
                return p_amount * 0.5
            return 150.0

        if ctype == "cargo_category_ban":
            return p_amount * 0.5 if p_amount > 0 else 50.0

        if ctype == "max_deadhead_distance":
            max_km = float(params.get("max_deadhead_km", 9999))
            if distance_km > max_km:
                return p_amount if p_amount > 0 else (distance_km - max_km) * 10.0
            return 0.0

        if ctype == "mileage_cap":
            max_km = float(params.get("max_deadhead_km", 9999))
            if state.total_deadhead_km + distance_km > max_km:
                over = state.total_deadhead_km + distance_km - max_km
                return max(p_amount, over * 10.0)
            return 0.0

        if ctype == "max_haul_distance":
            max_km = float(params.get("max_km", 9999))
            if haul_km > max_km:
                return p_amount if p_amount > 0 else (haul_km - max_km) * 5.0
            return 0.0

        if ctype == "day_off_requirement":
            return p_amount * 0.3 if p_amount > 0 else 100.0

        if ctype == "monthly_visit_requirement":
            return p_amount * 0.2 if p_amount > 0 else 80.0

        return p_amount * 0.5 if p_amount > 0 else 0.0

    def _estimate_hard_penalty(
        self,
        constraint: dict,
        cargo_name: str,
        distance_km: float,
        haul_km: float,
        end_lat: float,
        end_lng: float,
        state: StateTracker,
    ) -> float:
        """估算硬约束的罚分成本（用于排序，不用于过滤）。"""
        ctype = constraint.get("type", "")
        p_amount = float(constraint.get("penalty_amount", 0) or 0)
        params = constraint.get("params", {})

        if ctype == "spatial_restrict" and params.get("type") == "bounding_box":
            bb = params.get("bounding_box", {})
            if not PreferenceEngine._in_bounding_box(end_lat, end_lng, bb):
                return p_amount * 2.0 if p_amount else 2000.0
        if ctype == "max_deadhead_distance":
            max_km = float(params.get("max_deadhead_km", 9999))
            if distance_km > max_km:
                return p_amount * 2.0 if p_amount else (distance_km - max_km) * 20.0
        if ctype == "max_haul_distance":
            max_km = float(params.get("max_km", 9999))
            if haul_km > max_km:
                return p_amount * 2.0 if p_amount else (haul_km - max_km) * 15.0
        if ctype == "daily_home_deadline":
            # 具体可行性在 score_and_rank 中按 finish_min 精确过滤。
            return 0.0
        if ctype == "cargo_category_ban":
            return p_amount * 2.0 if p_amount else 500.0
        return 0.0

    @staticmethod
    def _compute_long_order_penalty(total_minutes: int) -> float:
        """长途订单惩罚。"""
        if total_minutes > 600:
            return 300.0
        if total_minutes > 480:
            return 180.0
        if total_minutes > _LONG_ORDER_THRESHOLD_MINUTES:
            return 80.0
        return 0.0

    @staticmethod
    def _compute_deadhead_risk_cost(deadhead_km: float) -> float:
        """大空驶风险成本。"""
        if deadhead_km > _EXTREME_DEADHEAD_KM:
            return 500.0
        if deadhead_km > _HEAVY_DEADHEAD_KM:
            return 200.0
        if deadhead_km > 60:
            return 80.0
        return 0.0

    @staticmethod
    def _compute_short_safe_order_bonus(
        total_minutes: int, net_profit: float, breaks_critical: bool,
    ) -> float:
        """短单安全奖励。"""
        if total_minutes <= 180 and net_profit >= 80 and not breaks_critical:
            return 60.0
        return 0.0

    @staticmethod
    def estimate_wait_value(
        current_min: int,
        state: StateTracker,
        constraints: list[dict],
        horizon: int,
        opportunity_predictor: "OpportunityPredictor | None" = None,
    ) -> float:
        """估算等待动作的价值（元）。

        wait_value = 未来货源预期收益 - 时间成本 + 休息补足价值 + 任务准备价值
        """
        hour = _get_hour_of_day(current_min)

        future_income = 0.0
        if opportunity_predictor:
            try:
                prediction = opportunity_predictor.predict_next_hours(hour, 2)
                future_income = prediction.get("expected_avg_price", 0) * 0.3
            except Exception:
                pass

        time_cost = 25.0  # 30min ≈ 25 元机会成本

        rest_value = 0.0
        for c in constraints:
            if c.get("type") != "daily_rest":
                continue
            required = int(c.get("params", {}).get("min_continuous_minutes", 0))
            if required <= 0:
                continue
            current_rest = state.get_max_continuous_rest_today(current_min)
            if current_rest < required:
                rest_value = 200.0
                break

        task_prep_value = 0.0
        if hasattr(state, "open_tasks"):
            for task in state.open_tasks:
                if task.get("type") == "mandatory_cargo":
                    activation = task.get("activation_min", 0)
                    if 0 < activation - current_min < 360:
                        task_prep_value = 100.0
                        break

        night_penalty = 30.0 if (hour >= 22 or hour < 6) else 0.0

        return future_income - time_cost + rest_value + task_prep_value - night_penalty

    def score_and_rank(self, items: list[dict], status: dict, state: StateTracker,
                       constraints: list[dict], top_n: int = 5) -> list[dict]:
        """评分并返回 Top N 候选（Task #2 优化评分公式）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)
        remaining = horizon - current_min

        # 计算月度进度比（Task #2）
        month_progress_ratio = current_min / max(1, horizon)
        time_phase_multiplier = self._get_time_phase_multiplier(month_progress_ratio)
        hard_banned, soft_banned = self._get_banned_categories(constraints)
        monthly_deadhead_cap = self._get_monthly_deadhead_cap(constraints)

        # 月度空闲日评分惩罚（Task #2）：今天是 planner 建议的空闲日时，软惩罚所有候选
        idle_day_penalty = 0.0
        if state.monthly_planner and state.monthly_planner.is_today_idle_day(_get_day_index(current_min) + 1):
            progress = state.monthly_planner.get_progress_report(
                state.completed_idle_days, _get_day_index(current_min) + 1)
            if progress["remaining_required"] > 0:
                idle_day_penalty = 30.0  # 软惩罚，高净利润货仍可突破

        scored = []

        # 收集 mandatory_cargo 目标 cargo_id（这些不应被 hard_banned 过滤）
        mandatory_cargo_ids: set[str] = set()
        if hasattr(state, 'open_tasks'):
            for _t in state.open_tasks:
                if _t.get("type") == "mandatory_cargo":
                    tid = str(_t.get("target", ""))
                    if tid:
                        mandatory_cargo_ids.add(tid)

        # 收集 mandatory_cargo 的 pickup 信息（用于过滤赶不到的候选）
        mandatory_pickups = self._build_mandatory_pickup_info(state, constraints, items)
        mandatory_cargo_ids.update(
            str(mp.get("cargo_id", "")) for mp in mandatory_pickups
        )
        mandatory_cargo_ids.discard("")

        for item in items:
            cargo = item.get("cargo", {})
            distance_km = float(item.get("distance_km", 0))
            price = float(cargo.get("price", 0))  # 已是元
            cargo_id = cargo.get("cargo_id", "")
            cost_time = int(cargo.get("cost_time_minutes", 0))
            start = cargo.get("start", {})
            end = cargo.get("end", {})
            start_lat = float(start.get("lat", 0))
            start_lng = float(start.get("lng", 0))
            end_lat = float(end.get("lat", 0))
            end_lng = float(end.get("lng", 0))
            load_time = cargo.get("load_time")
            cargo_name = cargo.get("cargo_name", "")

            # 过滤: 仅硬禁品类先剔除；软偏好品类留下通过评分惩罚
            # mandatory_cargo 目标跳过此过滤（熟货优先）
            if any(b in cargo_name for b in hard_banned) and cargo_id not in mandatory_cargo_ids:
                continue

            # 过滤: 装货时间窗已过期
            deadhead_min = _distance_to_minutes(distance_km) if distance_km > 1e-6 else 0
            wait_for_load_min = 0
            if load_time and isinstance(load_time, list) and len(load_time) == 2:
                try:
                    window_end_min = _wall_to_sim_minutes(str(load_time[1]).strip())
                    arrival_min = current_min + deadhead_min
                    if arrival_min > window_end_min:
                        continue
                    window_start_min = _wall_to_sim_minutes(str(load_time[0]).strip())
                    wait_for_load_min = max(0, window_start_min - arrival_min)
                except Exception:
                    pass

            # 过滤: 月内无法完成
            total_time = deadhead_min + wait_for_load_min + cost_time
            if total_time > remaining:
                continue
            if _violates_time_restriction(current_min, current_min + total_time, constraints):
                continue

            # 过滤: 卸货点违反空间约束（bounding_box / forbidden_circle）
            # mandatory_cargo 目标跳过此过滤
            if cargo_id not in mandatory_cargo_ids:
                _spatial_blocked = False
                for c in constraints:
                    if c.get("type") != "spatial_restrict":
                        continue
                    p = c.get("params", {})
                    if p.get("type") == "bounding_box":
                        bb = p.get("bounding_box", {})
                        if not PreferenceEngine._in_bounding_box(end_lat, end_lng, bb):
                            _spatial_blocked = True
                            break
                    elif p.get("type") == "forbidden_circle":
                        cx = float(p.get("center_lat", 0))
                        cy = float(p.get("center_lng", 0))
                        r = float(p.get("radius_km", 20))
                        if haversine(end_lat, end_lng, cx, cy) < r:
                            _spatial_blocked = True
                            break
                if _spatial_blocked:
                    continue

            # daily_rest 不再硬过滤，由 soft_penalty_cost 经济惩罚处理
            # RuleLayer._check_forced_rest 在剩余时间仅够补休时仍会强制 wait
            _finish = current_min + total_time

            # 赴装货点空驶上限是单笔硬约束，超限订单直接剔除。
            if cargo_id not in mandatory_cargo_ids:
                _deadhead_blocked = False
                for c in constraints:
                    if c.get("type") != "max_deadhead_distance":
                        continue
                    max_km = float(c.get("params", {}).get("max_deadhead_km", 9999))
                    if distance_km > max_km:
                        _deadhead_blocked = True
                        break
                if _deadhead_blocked:
                    continue

            # 过滤: 候选完成后赶不到 mandatory_cargo pickup 的订单
            # 仅在 mandatory 窗口临近（<12h）时才硬过滤，否则交给评分惩罚
            if mandatory_pickups and cargo_id not in mandatory_cargo_ids:
                finish_min = current_min + total_time
                _skip_for_mandatory = False
                for mp in mandatory_pickups:
                    deadline = mp.get("window_end", mp["activation_min"])
                    if deadline < current_min:
                        continue
                    # 远期（>12h）：不过滤，由 future_position_cost 惩罚
                    if deadline - current_min > 720:
                        continue
                    travel_to_pickup = _distance_to_minutes(
                        haversine(end_lat, end_lng, mp["pickup_lat"], mp["pickup_lng"])
                    )
                    if finish_min + travel_to_pickup > deadline:
                        _skip_for_mandatory = True
                        break
                if _skip_for_mandatory:
                    continue

            # 过滤: 硬 daily_home_deadline 场景下，完单后必须能在当天 deadline 前回家。
            if any(
                c.get("type") == "daily_home_deadline"
                and self.classify_constraint_severity(c) == "hard"
                and self._violates_daily_home_deadline(
                    end_lat, end_lng, _finish, current_min, c)
                for c in constraints
            ):
                continue

            # 计算干线距离
            haul_km = haversine(start_lat, start_lng, end_lat, end_lng)

            # 净利润
            total_cost = (distance_km + haul_km) * self.cost_per_km
            net_profit = price - total_cost

            # 时间效率
            total_minutes = deadhead_min + cost_time
            time_efficiency = net_profit / max(total_minutes, 1)

            # ── true_net: 区分硬/软约束，经济化处理 ──
            hard_penalty = 0.0
            soft_penalty_cost = 0.0
            has_soft_penalty = False

            for c in constraints:
                c_severity = self.classify_constraint_severity(c)
                if c_severity == "hard":
                    hard_penalty += self._estimate_hard_penalty(
                        c, cargo_name, distance_km, haul_km,
                        end_lat, end_lng, state)
                else:
                    cost_p = self._estimate_soft_penalty_cost(
                        c, cargo_name, distance_km, haul_km,
                        end_lat, end_lng, _finish, current_min, state)
                    if cost_p > 0:
                        soft_penalty_cost += cost_p
                        has_soft_penalty = True

            soft_penalty_cost = min(soft_penalty_cost, _MAX_SOFT_PENALTY_PER_ORDER)

            # 聚类价值
            cluster_value = self._compute_cluster_value(end_lat, end_lng, state)

            # 稀缺性因子
            scarcity_factor = self._get_scarcity_factor(item, state)

            # 长途惩罚 / 大空驶风险 / 短单奖励
            long_order_penalty = self._compute_long_order_penalty(total_time)
            deadhead_risk_cost = self._compute_deadhead_risk_cost(distance_km)
            short_safe_bonus = self._compute_short_safe_order_bonus(
                total_time, net_profit, False)

            finish_min = current_min + total_time
            future_position_cost = self._future_position_cost(
                end_lat, end_lng, finish_min, current_min,
                constraints, state, horizon, mandatory_pickups,
            )

            # ── true_net 评分公式 ──
            true_net_score = (
                net_profit
                - future_position_cost
                - soft_penalty_cost
                - long_order_penalty
                - deadhead_risk_cost
                + short_safe_bonus
                + cluster_value
                + scarcity_factor
                - idle_day_penalty
            )

            scored.append({
                "cargo_id": cargo_id,
                "cargo_name": cargo_name,
                "price": round(price, 2),
                "net_profit": round(net_profit, 2),
                "deadhead_km": round(distance_km, 2),
                "haul_km": round(haul_km, 2),
                "total_minutes": total_time,
                "time_efficiency": round(time_efficiency, 4),
                "penalty_score": round(soft_penalty_cost, 2),
                "has_soft_penalty": has_soft_penalty,
                "hard_penalty": round(hard_penalty, 2),
                "future_position_cost": round(future_position_cost, 2),
                "long_order_penalty": round(long_order_penalty, 2),
                "deadhead_risk_cost": round(deadhead_risk_cost, 2),
                "score": round(true_net_score, 4),
                "true_net": round(true_net_score, 4),
                "start": {"lat": start_lat, "lng": start_lng},
                "end": {"lat": end_lat, "lng": end_lng},
                "load_time": load_time,
                "cost_time_minutes": cost_time,
            })

        # 按 score 降序排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _get_banned_categories(constraints: list[dict]) -> tuple[set[str], set[str]]:
        hard_banned: set[str] = set()
        soft_banned: set[str] = set()
        for c in constraints:
            if c.get("type") == "cargo_category_ban":
                severity = c.get("severity", "hard")
                params = c.get("params", {})
                for item in params.get("banned_categories", []):
                    if severity == "hard":
                        hard_banned.add(str(item))
                    else:
                        soft_banned.add(str(item))
        return hard_banned, soft_banned

    @staticmethod
    def _get_monthly_deadhead_cap(constraints: list[dict]) -> float | None:
        caps: list[float] = []
        for c in constraints:
            if c.get("type") == "mileage_cap":
                try:
                    caps.append(float(c.get("params", {}).get("max_deadhead_km")))
                except Exception:
                    continue
        return min(caps) if caps else None

    @staticmethod
    def _violates_time_restriction(start_min: int, end_min: int, constraints: list[dict]) -> bool:
        return _violates_time_restriction(start_min, end_min, constraints)


# ─────────────────────────────────────────────────────────────────────────────
# LLMLayer - LLM 决策层
# ─────────────────────────────────────────────────────────────────────────────
class LLMLayer:
    """基于 LLM 的最终决策层。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api

    def decide(self, candidates: list[dict], status: dict, state: StateTracker,
               constraints: list[dict],
               raw_candidates: list[dict] | None = None,
               pattern_analyzer: "HistoryPatternAnalyzer | None" = None,
               opportunity_predictor: "OpportunityPredictor | None" = None) -> dict:
        """让 LLM 从候选中选择最优决策。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = _get_hour_of_day(current_min)
        remaining = _get_horizon_minutes(status) - current_min
        day_idx = _get_day_index(current_min) + 1

        # 构建精简 prompt
        context = {
            "当前状态": {
                "仿真第几天": day_idx,
                "当前时刻": _sim_minutes_to_wall(current_min).strftime("%m-%d %H:%M"),
                "剩余分钟": remaining,
                "累计接单": state.total_orders,
                "今日接单": state.get_orders_today(current_min),
                "今日最长连续休息分钟": state.get_max_continuous_rest_today(current_min),
                "月累计空驶km": round(state.total_deadhead_km, 1),
            },
            "候选货源": [],
            "可选动作": "take_order(选择cargo_id) / wait(等待分钟) / reposition(移动到坐标)"
        }

        for c in candidates:
            context["候选货源"].append({
                "cargo_id": c["cargo_id"],
                "品类": c.get("cargo_name", ""),
                "true_net": c.get("true_net", c.get("score", 0)),
                "净利润元": c["net_profit"],
                "空驶km": c["deadhead_km"],
                "总耗时分钟": c["total_minutes"],
                "有软罚分": c.get("has_soft_penalty", False),
                "罚分成本": c.get("penalty_score", 0),
                "未来位置成本": c.get("future_position_cost", 0),
                "长途惩罚": c.get("long_order_penalty", 0),
                "空驶风险成本": c.get("deadhead_risk_cost", 0),
            })

        # 添加关键约束提示
        constraint_hints = []
        for c in constraints[:5]:
            constraint_hints.append(c.get("description", c.get("type", "")))
        if constraint_hints:
            context["关键约束"] = constraint_hints

        # Task #4: 注入增强上下文（历史模式、未来预测、偏好进度时间线）
        try:
            enhanced = self._build_enhanced_prompt(
                raw_candidates if raw_candidates is not None else [],
                status, state, constraints,
                pattern_analyzer, opportunity_predictor,
                state.monthly_planner if hasattr(state, "monthly_planner") else None,
            )
            if enhanced:
                context["增强分析"] = enhanced
        except Exception as e:
            _logger.warning("构建增强Prompt失败: %s", e)

        prompt = json.dumps(context, ensure_ascii=False)

        try:
            resp = self._api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": (
                        "你是高级货运调度员。根据true_net(真实净利润=收入-成本-罚分)进行决策。"
                        "优先选择true_net最高的候选。如果所有候选true_net<0，选择等待。"
                        "允许小额软罚分(如有软罚分=true)只要true_net仍为正且明显优于无罚方案。"
                        "只输出JSON: {\"action\":\"take_order|wait|reposition\","
                        "\"cargo_id\":\"仅take_order需要\","
                        "\"duration_minutes\":正整数(仅wait),"
                        "\"latitude\":float,\"longitude\":float(仅reposition),"
                        "\"reason\":\"简短理由\"}"
                    )},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            })
            return self._parse_response(resp, candidates)
        except Exception as e:
            _logger.warning("LLM 决策异常: %s，使用兜底", e)
            return self._fallback(candidates)

    def _build_enhanced_prompt(self, candidates: list, status: dict, state: StateTracker,
                               constraints: list,
                               pattern_analyzer: "HistoryPatternAnalyzer | None",
                               opportunity_predictor: "OpportunityPredictor | None",
                               monthly_planner: "MonthlyConstraintPlanner | None") -> dict:
        """构建增强型决策上下文（Task #4）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = (current_min % 1440) / 60

        # 1. 历史模式分析
        cargo_types: list[str] = []
        for c in candidates:
            t = c.get("cargo", {}).get("cargo_type", "未知")
            if t not in cargo_types:
                cargo_types.append(t)

        best_pattern: dict = {}
        similar_scenarios: list = []
        if pattern_analyzer is not None:
            best_pattern = pattern_analyzer.get_best_historical_pattern(hour, cargo_types)
            similar_scenarios = pattern_analyzer.get_similar_scenarios(
                hour, len(candidates))

        # 2. 未来机会预测
        future_prediction: dict = {}
        wait_recommended = False
        current_best_price = max(
            (float(c.get("cargo", {}).get("price", 0)) for c in candidates),
            default=0.0,
        )
        if opportunity_predictor is not None:
            future_prediction = opportunity_predictor.predict_next_hours(hour, 2)
            wait_recommended = opportunity_predictor.should_wait_for_better(
                current_best_price, hour)

        # 3. 偏好进度时间线
        constraint_timeline: dict = {}
        if monthly_planner is not None:
            completed = state.completed_idle_days if hasattr(state, "completed_idle_days") else 0
            current_day = (current_min // 1440) + 1
            progress = monthly_planner.get_progress_report(completed, current_day)
            constraint_timeline = {
                "空闲日进度": (
                    f"已完成{progress.get('completed', 0)}天, "
                    f"还需{progress.get('remaining_required', 0)}天"
                ),
                "剩余可用天数": progress.get("remaining_available_days", 0),
                "是否落后": progress.get("is_behind_schedule", False),
            }

        return {
            "决策分析": {
                "当前最优货物价格": round(current_best_price, 2),
                "未来预测": future_prediction,
                "建议等待": wait_recommended,
                "等待原因": (
                    "未来1小时预计有更高价货源" if wait_recommended else "当前货源质量尚可"
                ),
            },
            "历史参考": {
                "该时段最佳货类": best_pattern.get("type", "无数据") if best_pattern else "无数据",
                "历史平均利润": best_pattern.get("avg_profit", 0) if best_pattern else 0,
                "类似场景决策": similar_scenarios,
            },
            "偏好进度": constraint_timeline,
        }

    def _parse_response(self, resp: dict, candidates: list[dict]) -> dict:
        """解析 LLM 响应为标准动作格式。"""
        try:
            choices = resp.get("choices", [])
            if not choices:
                return self._fallback(candidates)
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return self._fallback(candidates)
            data = json.loads(content)
            action = str(data.get("action", "")).strip().lower()

            if action == "take_order":
                cargo_id = str(data.get("cargo_id", "")).strip()
                # 验证 cargo_id 在候选中
                valid_ids = {c["cargo_id"] for c in candidates}
                if cargo_id and cargo_id in valid_ids:
                    return {"action": "take_order", "params": {"cargo_id": cargo_id}}
                # 如果 LLM 给的 ID 不在候选中，选评分最高的
                if candidates:
                    return {"action": "take_order", "params": {"cargo_id": candidates[0]["cargo_id"]}}
                return self._default_wait()

            if action == "wait":
                duration = int(data.get("duration_minutes", 60))
                duration = max(1, min(duration, 480))
                return {"action": "wait", "params": {"duration_minutes": duration}}

            if action == "reposition":
                lat = float(data.get("latitude", 0))
                lng = float(data.get("longitude", 0))
                if lat != 0 and lng != 0:
                    return {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}
                return self._default_wait()

        except Exception as e:
            _logger.warning("LLM 响应解析失败: %s", e)

        return self._fallback(candidates)

    def _fallback(self, candidates: list[dict]) -> dict:
        """兜底：有高分候选就接单，否则等待。"""
        if candidates and candidates[0].get("score", 0) > -50 and candidates[0].get("penalty_score", 0) == 0:
            return {"action": "take_order", "params": {"cargo_id": candidates[0]["cargo_id"]}}
        return self._default_wait()

    @staticmethod
    def _default_wait() -> dict:
        return {"action": "wait", "params": {"duration_minutes": 60}}


# ─────────────────────────────────────────────────────────────────────────────
# TokenBudgetOptimizer - Token预算智能分配器 (Task #6)
# ─────────────────────────────────────────────────────────────────────────────
class TokenBudgetOptimizer:
    """Token预算智能分配器"""

    def __init__(self, total_budget: int = 5_000_000, num_drivers: int = 10):
        self.total_budget = total_budget
        self.num_drivers = num_drivers
        self.base_per_driver = total_budget // num_drivers
        self.usage_tracker: dict[str, int] = {}  # {driver_id: tokens_used}
        self.complexity_scores: dict[str, float] = {}  # {driver_id: complexity_score}

    def register_driver(self, driver_id: str, preferences_text: str):
        """注册司机并评估偏好复杂度"""
        self.usage_tracker[driver_id] = 0
        # 复杂度评估：基于偏好文本长度和约束关键词数量
        complexity = self._calculate_complexity(preferences_text)
        self.complexity_scores[driver_id] = complexity

    def _calculate_complexity(self, preferences_text: str) -> float:
        """计算偏好复杂度分数（0-10）"""
        if not preferences_text:
            return 1.0
        score = 1.0
        # 文本长度贡献
        score += min(3.0, len(preferences_text) / 100)
        # 约束关键词数量
        constraint_keywords = ["不接", "必须", "禁止", "至少", "不超过", "每天", "每周",
                               "整天", "回家", "约定", "指定", "限制"]
        keyword_count = sum(1 for kw in constraint_keywords if kw in preferences_text)
        score += min(4.0, keyword_count * 0.8)
        # 特殊事件（家事、约定等）
        special_keywords = ["家事", "配偶", "接送", "约定货物", "熟货"]
        special_count = sum(1 for kw in special_keywords if kw in preferences_text)
        score += special_count * 1.5
        return min(10.0, score)

    def get_budget_allocation(self, driver_id: str, current_day: int,
                              total_days: int = 31, preference_fulfillment: float = 1.0) -> int:
        """获取当前司机的Token预算分配"""
        base = self.base_per_driver
        complexity = self.complexity_scores.get(driver_id, 5.0)

        # 1. 复杂度加权：高复杂度司机多分配
        complexity_multiplier = 0.5 + (complexity / 10.0)  # 0.5 ~ 1.5

        # 2. 月度进度调整
        progress_ratio = current_day / total_days
        if progress_ratio < 0.3:
            time_multiplier = 0.8  # 月初保守
        elif progress_ratio > 0.8:
            time_multiplier = 1.3  # 月末冲刺
        else:
            time_multiplier = 1.0

        # 3. 偏好达成度调整
        if preference_fulfillment < 0.7:
            urgency_multiplier = 1.5  # 偏好落后，增加投入
        elif preference_fulfillment < 0.9:
            urgency_multiplier = 1.2
        else:
            urgency_multiplier = 1.0

        allocation = int(base * complexity_multiplier * time_multiplier * urgency_multiplier)
        return allocation

    def record_usage(self, driver_id: str, tokens_used: int):
        """记录Token使用"""
        if driver_id not in self.usage_tracker:
            self.usage_tracker[driver_id] = 0
        self.usage_tracker[driver_id] += tokens_used

    def get_remaining_budget(self, driver_id: str) -> int:
        """获取剩余预算"""
        used = self.usage_tracker.get(driver_id, 0)
        return max(0, self.base_per_driver - used)

    def suggest_strategy(self, driver_id: str, remaining_steps: int) -> str:
        """建议Token使用策略"""
        remaining = self.get_remaining_budget(driver_id)
        if remaining_steps <= 0:
            return "RULE_HEURISTIC_ONLY"

        tokens_per_step = remaining / remaining_steps

        if tokens_per_step > 5000:
            return "FULL_LLM"  # Token充足，每步都调LLM
        elif tokens_per_step > 2000:
            return "SELECTIVE_LLM"  # 仅复杂决策调LLM
        else:
            return "RULE_HEURISTIC_ONLY"  # Token紧张，仅用规则+启发式

    def should_use_llm(self, driver_id: str, decision_complexity: str, remaining_steps: int,
                       context: str = "decision") -> bool:
        """判断当前决策是否应该调用LLM"""
        # 战略规划必调
        if context == "strategic_plan":
            return True
        # 高风险审核：仅在真正接近预算（>=95%）时才拒绝
        if context == "review_high_risk":
            used = self.usage_tracker.get(driver_id, 0)
            if used < self.base_per_driver * 0.95:
                return True
            return False

        strategy = self.suggest_strategy(driver_id, remaining_steps)

        if strategy == "FULL_LLM":
            return True
        elif strategy == "SELECTIVE_LLM":
            # 只在复杂决策时调用
            return decision_complexity in ("high", "medium")
        else:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# MultiDriverCoordinationLayer - 多司机信息共享与协调层 (Task #7)
# ─────────────────────────────────────────────────────────────────────────────
class MultiDriverCoordinationLayer:
    """多司机信息共享与协调层"""

    def __init__(self):
        self.driver_profiles: dict[str, dict] = {}  # {driver_id: profile_dict}
        self.cargo_claim_history: dict[str, str] = {}  # {cargo_id: driver_id} 已被接的货
        self.active_decisions: dict[str, dict] = {}  # {driver_id: last_decision_info}
        self.enable_competition_filter = False

    def register_driver_profile(self, driver_id: str, preferences_text: str,
                                home_location: tuple[float, float] | None = None):
        """注册司机画像"""
        profile = {
            "specialty": self._infer_specialty(preferences_text),
            "home": home_location,
            "avg_distance_preference": self._infer_distance_preference(preferences_text),
            "preferred_cargo_types": self._infer_cargo_types(preferences_text),
            "active_hours": self._infer_active_hours(preferences_text),
            "total_orders": 0,
            "total_income": 0.0
        }
        self.driver_profiles[driver_id] = profile

    def _infer_specialty(self, preferences_text: str) -> str:
        """推断司机擅长领域"""
        if not preferences_text:
            return "general"
        text = preferences_text.lower()
        if any(kw in text for kw in ["短途", "市内", "同城"]):
            return "short_distance"
        elif any(kw in text for kw in ["长途", "干线", "跨省"]):
            return "long_distance"
        elif any(kw in text for kw in ["冷链", "冷藏", "生鲜"]):
            return "cold_chain"
        elif any(kw in text for kw in ["危险品", "化工"]):
            return "hazardous"
        return "general"

    def _infer_distance_preference(self, preferences_text: str) -> str:
        """推断距离偏好"""
        if not preferences_text:
            return "any"
        if "不超过" in preferences_text and "km" in preferences_text.lower():
            return "short"
        if "长途" in preferences_text:
            return "long"
        return "any"

    def _infer_cargo_types(self, preferences_text: str) -> list:
        """推断偏好货类"""
        types = []
        cargo_keywords = {
            "食品": "food", "电子": "electronics", "机械": "machinery",
            "建材": "construction", "快递": "express", "家具": "furniture"
        }
        for cn, en in cargo_keywords.items():
            if cn in (preferences_text or ""):
                types.append(en)
        return types if types else ["any"]

    def _infer_active_hours(self, preferences_text: str) -> tuple:
        """推断活跃时段"""
        # 默认6:00-22:00
        return (6, 22)

    def record_decision(self, driver_id: str, cargo_id: str, action: str):
        """记录司机决策（用于避免竞争）"""
        if action == "take_order" and cargo_id:
            self.cargo_claim_history[cargo_id] = driver_id
            if driver_id in self.driver_profiles:
                self.driver_profiles[driver_id]["total_orders"] += 1
        self.active_decisions[driver_id] = {
            "cargo_id": cargo_id, "action": action
        }

    def filter_competitive_cargo(self, driver_id: str, candidates: list) -> list:
        """过滤已被其他司机接的货源（避免竞争）"""
        if not self.enable_competition_filter:
            return candidates
        filtered = []
        for item in candidates:
            cargo_id = str(item.get("cargo", {}).get("cargo_id", ""))
            if cargo_id not in self.cargo_claim_history:
                filtered.append(item)
        return filtered if filtered else candidates  # 保底

    def get_differentiated_recommendations(self, driver_id: str, candidates: list) -> list:
        """根据司机画像推荐差异化货源"""
        profile = self.driver_profiles.get(driver_id)
        if not profile:
            return candidates

        specialty = profile.get("specialty", "general")

        # 为候选货源打差异化加分
        scored = []
        for item in candidates:
            cargo = item.get("cargo", {})
            bonus = 0

            # 距离匹配
            distance = float(item.get("distance_km", 0)) + float(cargo.get("haul_km", 0))
            if specialty == "short_distance" and distance < 100:
                bonus += 30
            elif specialty == "long_distance" and distance > 200:
                bonus += 30

            # 货类匹配
            cargo_type = cargo.get("cargo_type", "")
            if cargo_type in profile.get("preferred_cargo_types", []):
                bonus += 20

            scored.append({"item": item, "coordination_bonus": bonus})

        # 按加分排序
        scored.sort(key=lambda x: x["coordination_bonus"], reverse=True)
        return [s["item"] for s in scored]

    def get_competition_heat(self, cargo_id: str) -> float:
        """获取货源竞争热度（0-1）"""
        # 简单实现：已有人接则为1，否则为0
        return 1.0 if cargo_id in self.cargo_claim_history else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ModelDecisionService - 主决策服务
# ─────────────────────────────────────────────────────────────────────────────
class ModelDecisionService:
    """三层混合决策架构主入口。"""

    def __init__(self, api: SimulationApiPort, enable_rl_layer: bool = True) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        # 各层组件
        self._state_tracker: dict[str, StateTracker] = {}
        self._preference_engine = PreferenceEngine(api)
        self._rule_layer = RuleLayer()
        self._heuristic_layer = HeuristicLayer()
        self._llm_layer = LLMLayer(api)
        # 时间窗优化器（Task #3）
        self._time_window_optimizer = TimeWindowOptimizer()
        # Task #4: 历史模式分析器、未来机会预测器
        self.pattern_analyzer = HistoryPatternAnalyzer()
        self.opportunity_predictor = OpportunityPredictor()
        self.profit_search_layer = ProfitSearchLayer()
        # Task #5: 主动空驶层
        self.reposition_layer = ProactiveRepositionLayer()
        # 已经从历史填充过 pattern_analyzer 的 driver 集合
        self._pattern_filled: set[str] = set()
        # Task #6: Token预算智能分配器
        self.token_optimizer = TokenBudgetOptimizer()
        # Task #7: 多司机信息共享与协调层
        self.coordination_layer = MultiDriverCoordinationLayer()
        # RL 增强层（可选加载）
        self._rl_layer = self._try_load_rl_layer() if enable_rl_layer else None
        # 已注册到 token_optimizer / coordination_layer 的 driver 集合
        self._registered_drivers: set[str] = set()
        # Task #4: 战略规划器 + 决策审核员
        self._strategic_planner = StrategicPlanner(self._api)
        self._reviewer = DecisionReviewer(self._api)

    def _get_state(self, driver_id: str) -> StateTracker:
        if driver_id not in self._state_tracker:
            self._state_tracker[driver_id] = StateTracker()
        return self._state_tracker[driver_id]

    @staticmethod
    def _try_load_rl_layer() -> Any:
        """尝试加载 RL 增强层（可选，模型文件不存在时返回 None）。"""
        try:
            from agent.rl_integration import RLDecisionLayer
            from pathlib import Path
            model_dir = Path(__file__).parent / "models"
            policy = model_dir / "policy_best.npz"
            value = model_dir / "value_best.npz"
            scorer = model_dir / "scorer_best.npz"
            if policy.exists():
                layer = RLDecisionLayer(
                    policy_path=policy,
                    value_path=value if value.exists() else None,
                    scorer_path=scorer if scorer.exists() else None,
                )
                if layer.is_loaded:
                    return layer
        except Exception:
            pass
        return None

    @staticmethod
    def _task_key(task: dict) -> tuple:
        task_type = task.get("type", "")
        if task_type == "mandatory_cargo":
            return (task_type, str(task.get("target") or task.get("cargo_id") or ""))
        if task_type == "scheduled_event":
            return (
                task_type,
                _to_int_or_zero(task.get("pickup_min")),
                _to_int_or_zero(task.get("home_deadline_min")),
            )
        if task_type == "monthly_visit":
            return (task_type, str(task.get("target", "")))
        if task_type == "daily_home_deadline":
            return (
                task_type,
                str(task.get("home_lat", "")),
                str(task.get("home_lng", "")),
                _to_int_or_zero(task.get("deadline_hour")),
            )
        return (task_type, str(task.get("target", "")))

    def _sync_constraint_tasks(
        self,
        state: StateTracker,
        constraints: list[dict],
        horizon: int,
    ) -> None:
        """把运行中才可见的关键约束补进 open_tasks。"""
        known = {self._task_key(t) for t in state.open_tasks}
        known.update(self._task_key(t) for t in state.completed_tasks)

        def add_once(task: dict) -> None:
            key = self._task_key(task)
            for existing in state.open_tasks:
                if self._task_key(existing) == key:
                    for field, value in task.items():
                        if value is not None:
                            existing[field] = value
                    return
            if key in known:
                return
            state.open_tasks.append(task)
            known.add(key)

        for c in constraints:
            ctype = c.get("type", "")
            p = c.get("params", {})

            if ctype == "mandatory_cargo":
                cargo_id = str(p.get("cargo_id", "")).strip()
                if not cargo_id:
                    continue
                task = {
                    "type": "mandatory_cargo",
                    "target": cargo_id,
                    "activation_min": _to_int_or_zero(p.get("activation_min")),
                    "pickup_lat": p.get("pickup_lat"),
                    "pickup_lng": p.get("pickup_lng"),
                    "deadline_minute": horizon,
                    "priority": 1,
                }
                if c.get("visible_end_time"):
                    try:
                        task["window_end"] = _wall_to_sim_minutes(str(c["visible_end_time"]))
                    except Exception:
                        pass
                add_once(task)

            elif ctype == "scheduled_event":
                task = {
                    "type": "scheduled_event",
                    "target": p.get("description", "event"),
                    "deadline_minute": _to_int_or_zero(p.get("home_deadline_min")),
                    "priority": 1,
                    "pickup_lat": p.get("pickup_lat"),
                    "pickup_lng": p.get("pickup_lng"),
                    "home_lat": p.get("home_lat"),
                    "home_lng": p.get("home_lng"),
                    "pickup_min": _to_int_or_zero(p.get("pickup_min")),
                    "home_deadline_min": _to_int_or_zero(p.get("home_deadline_min")),
                    "release_min": _to_int_or_zero(p.get("release_min")),
                    "pickup_wait_minutes": _to_int_or_zero(p.get("pickup_wait_minutes"), 10),
                }
                if task["pickup_min"] > 0 and task["home_deadline_min"] > 0:
                    add_once(task)

            elif ctype == "monthly_visit_requirement":
                target_lat = p.get("target_lat")
                target_lng = p.get("target_lng")
                if target_lat is None or target_lng is None:
                    continue
                add_once({
                    "type": "monthly_visit",
                    "target": f"lat={target_lat},lng={target_lng}",
                    "min_visit_days": _to_int_or_zero(p.get("min_visit_days")) or 1,
                    "deadline_minute": horizon,
                    "priority": 2,
                })

            elif ctype == "daily_home_deadline":
                add_once({
                    "type": "daily_home_deadline",
                    "target": "home",
                    "home_lat": p.get("home_lat"),
                    "home_lng": p.get("home_lng"),
                    "deadline_hour": _to_int_or_zero(p.get("deadline_hour"), 23),
                    "deadline_minute": _to_int_or_zero(p.get("deadline_hour"), 23) * 60,
                    "priority": 2,
                })

    def decide(self, driver_id: str) -> dict:
        """主决策方法：初始化 → 状态更新 → 规则层 → 查询货源 → 启发式 → LLM"""
        try:
            return self._decide_impl(driver_id)
        except Exception as e:
            self._logger.error("决策异常 driver=%s: %s", driver_id, e, exc_info=True)
            return {"action": "wait", "params": {"duration_minutes": 60}}

    def _validate_action(self, decision: dict, status: dict, constraints: list[dict]) -> dict | None:
        """统一动作安全校验，拒绝危险动作返回 None（调用方跳过该决策继续下一层）。"""
        action = decision.get("action", "")
        params = decision.get("params", {})
        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)

        if action == "reposition":
            lat = float(params.get("latitude", 0))
            lng = float(params.get("longitude", 0))
            cur_lat = float(status.get("current_lat", 0))
            cur_lng = float(status.get("current_lng", 0))

            # 拒绝 (0,0) 或超出中国范围
            if (lat == 0 and lng == 0) or not (_CHINA_LAT_RANGE[0] <= lat <= _CHINA_LAT_RANGE[1] and _CHINA_LNG_RANGE[0] <= lng <= _CHINA_LNG_RANGE[1]):
                self._logger.warning("安全闸拦截: reposition(%s,%s) 坐标异常，跳过该规则决策", lat, lng)
                return None

            # 拒绝单次空驶 > 300km
            dist = haversine(cur_lat, cur_lng, lat, lng)
            if dist > _MAX_SINGLE_REPOSITION_KM:
                step_ratio = (_MAX_SINGLE_REPOSITION_KM * 0.9) / dist
                lat = cur_lat + (lat - cur_lat) * step_ratio
                lng = cur_lng + (lng - cur_lng) * step_ratio
                params["latitude"] = round(lat, 6)
                params["longitude"] = round(lng, 6)
                dist = haversine(cur_lat, cur_lng, lat, lng)
                self._logger.info(
                    "reposition 距离超限，拆分为 %.1fkm 中间点 (%.5f, %.5f)",
                    dist, lat, lng,
                )

            # 拒绝执行后超 horizon
            travel_min = _distance_to_minutes(dist)
            if current_min + travel_min > horizon:
                self._logger.warning("安全闸拦截: reposition 将超出 horizon，跳过该规则决策")
                return None

            # 拒绝违反显式空间约束
            for c in constraints:
                if c.get("type") == "spatial_restrict":
                    p = c.get("params", {})
                    if p.get("type") == "bounding_box":
                        bb = p.get("bounding_box", {})
                        if not PreferenceEngine._in_bounding_box(lat, lng, bb):
                            self._logger.warning("安全闸拦截: reposition 违反 bounding_box 约束，跳过该规则决策")
                            return None
                    elif p.get("type") == "forbidden_circle":
                        cx = float(p.get("center_lat", 0))
                        cy = float(p.get("center_lng", 0))
                        r = float(p.get("radius_km", 20))
                        if haversine(lat, lng, cx, cy) < r:
                            self._logger.warning("安全闸拦截: reposition 进入禁入圆区，跳过该规则决策")
                            return None

        return decision  # 通过校验

    @staticmethod
    def _select_best_by_true_net(candidates: list[dict]) -> dict | None:
        """按 true_net 选择最佳候选：安全候选 vs 吃罚候选，取 penalty_margin 决策。"""
        if not candidates:
            return None

        safe = sorted(
            (c for c in candidates if not c.get("has_soft_penalty", False)),
            key=lambda c: c.get("true_net", c["score"]),
            reverse=True,
        )
        penalty = sorted(
            (c for c in candidates if c.get("has_soft_penalty", False)),
            key=lambda c: c.get("true_net", c["score"]),
            reverse=True,
        )

        best_safe = next((c for c in safe if ProfitSearchLayer._passes_guard(c)), None)
        best_penalty = next((c for c in penalty if ProfitSearchLayer._passes_guard(c)), None)

        chosen = None
        if best_penalty and best_safe:
            pn = best_penalty.get("true_net", best_penalty["score"])
            sn = best_safe.get("true_net", best_safe["score"])
            chosen = best_penalty if pn > sn + _PENALTY_MARGIN else best_safe
        elif best_safe:
            chosen = best_safe
        elif best_penalty:
            chosen = best_penalty

        return chosen

    @staticmethod
    def _best_true_net(candidates: list[dict]) -> float:
        if not candidates:
            return float("-inf")
        return max(float(c.get("true_net", c.get("score", 0))) for c in candidates)

    def _validated_take_order_decision(
        self,
        candidate: dict,
        items: list[dict],
        status: dict,
        constraints: list[dict],
        state: "StateTracker",
    ) -> dict | None:
        """Build a take_order decision only after the same checks used by LLM output."""
        cargo_id = str(candidate.get("cargo_id", ""))
        if not cargo_id:
            return None
        if not self._validate_take_order_feasibility(cargo_id, items, status):
            return None
        ok, reason = self._validate_take_order_constraints(
            cargo_id, items, status, constraints, state
        )
        if not ok:
            self._logger.info(
                "heuristic fast path rejected cargo=%s reason=%s",
                cargo_id,
                reason,
            )
            return None
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}

    def _validate_take_order_feasibility(self, cargo_id: str, items: list[dict],
                                          status: dict) -> bool:
        """校验 take_order 是否能在 horizon 内完成。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))

        for item in items:
            cargo = item.get("cargo", {})
            if str(cargo.get("cargo_id", "")) != str(cargo_id):
                continue
            distance_km = float(item.get("distance_km", 0))
            cost_time = int(cargo.get("cost_time_minutes", 0))
            load_time = cargo.get("load_time")

            deadhead_min = _distance_to_minutes(distance_km)
            wait_for_load = 0
            if load_time and isinstance(load_time, list) and len(load_time) == 2:
                try:
                    window_start = _wall_to_sim_minutes(str(load_time[0]).strip())
                    arrival = current_min + deadhead_min
                    wait_for_load = max(0, window_start - arrival)
                except Exception:
                    pass

            total_time = deadhead_min + wait_for_load + cost_time
            return current_min + total_time <= horizon

        return False  # 未找到 cargo 信息时拒绝（安全优先）

    @staticmethod
    def _build_scheduled_event_info(
        state: "StateTracker", constraints: list[dict]
    ) -> list[dict]:
        open_events = [
            t for t in (getattr(state, "open_tasks", []) or [])
            if t.get("type") == "scheduled_event"
        ]
        if not open_events:
            return []

        constraint_params = [
            c.get("params", {}) for c in constraints
            if c.get("type") == "scheduled_event"
            and isinstance(c.get("params", {}), dict)
        ]
        fallback = constraint_params[0] if constraint_params else {}
        events: list[dict] = []

        for task in open_events:
            params = dict(fallback)
            for key in (
                "pickup_min",
                "home_deadline_min",
                "release_min",
                "pickup_lat",
                "pickup_lng",
                "home_lat",
                "home_lng",
                "pickup_wait_minutes",
            ):
                if task.get(key) is not None:
                    params[key] = task.get(key)
            if params.get("home_deadline_min") is None and task.get("deadline_minute") is not None:
                params["home_deadline_min"] = task.get("deadline_minute")

            pickup_lat = params.get("pickup_lat")
            pickup_lng = params.get("pickup_lng")
            home_lat = params.get("home_lat")
            home_lng = params.get("home_lng")
            if None in (pickup_lat, pickup_lng, home_lat, home_lng):
                continue

            pickup_min = _to_int_or_zero(params.get("pickup_min"))
            home_deadline = _to_int_or_zero(params.get("home_deadline_min"))
            release_min = _to_int_or_zero(params.get("release_min")) or home_deadline
            if pickup_min <= 0 or home_deadline <= 0:
                continue
            try:
                events.append({
                    "pickup_min": pickup_min,
                    "home_deadline_min": home_deadline,
                    "release_min": release_min,
                    "pickup_lat": float(pickup_lat),
                    "pickup_lng": float(pickup_lng),
                    "home_lat": float(home_lat),
                    "home_lng": float(home_lng),
                    "pickup_wait_minutes": int(params.get("pickup_wait_minutes", 10) or 10),
                })
            except (TypeError, ValueError):
                continue

        return events

    def _validate_take_order_constraints(
        self,
        cargo_id: str,
        items: list[dict],
        status: dict,
        constraints: list[dict],
        state: "StateTracker",
    ) -> tuple[bool, str]:
        """综合 take_order 硬校验。返回 (通过, 原因)。

        覆盖检查：
        1. cargo 在可见 items 中
        2. horizon 内可完成
        3. 卸货点不违反 spatial_restrict
        4. 装卸距离不违反 max_haul_distance
        5. 赴装货点不违反 max_deadhead_distance
        6. 执行区间不穿过 time_restriction
        7. daily_home_deadline：完单后能回家
        8. scheduled_event：不跨过事件时间窗
        9. monthly_visit：完单后能到达到访点
        """
        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)
        cur_lat = float(status.get("current_lat", 0))
        cur_lng = float(status.get("current_lng", 0))

        # 找到 cargo
        found_item = None
        for item in items:
            cargo = item.get("cargo", {})
            if str(cargo.get("cargo_id", "")) == str(cargo_id):
                found_item = item
                break
        if found_item is None:
            return False, "cargo_not_in_visible_items"

        cargo = found_item.get("cargo", {})
        distance_km = float(found_item.get("distance_km", 0))
        cost_time = int(cargo.get("cost_time_minutes", 0))
        load_time = cargo.get("load_time")
        start = cargo.get("start", {})
        end = cargo.get("end", {})
        start_lat = float(start.get("lat", 0))
        start_lng = float(start.get("lng", 0))
        end_lat = float(end.get("lat", 0))
        end_lng = float(end.get("lng", 0))
        haul_km = haversine(start_lat, start_lng, end_lat, end_lng)

        deadhead_min = _distance_to_minutes(distance_km)
        wait_for_load = 0
        if load_time and isinstance(load_time, list) and len(load_time) == 2:
            try:
                window_start = _wall_to_sim_minutes(str(load_time[0]).strip())
                arrival = current_min + deadhead_min
                wait_for_load = max(0, window_start - arrival)
            except Exception:
                pass
        total_time = deadhead_min + wait_for_load + cost_time
        finish_min = current_min + total_time

        # 1. horizon 内可完成
        if finish_min > horizon:
            return False, "exceeds_horizon"

        # 2. spatial_restrict：卸货点不能出 bounding_box / 进禁入圆
        for c in constraints:
            if c.get("type") != "spatial_restrict":
                continue
            p = c.get("params", {})
            if p.get("type") == "bounding_box":
                bb = p.get("bounding_box", {})
                if not PreferenceEngine._in_bounding_box(end_lat, end_lng, bb):
                    return False, "destination_out_of_allowed_area"
            elif p.get("type") == "forbidden_circle":
                cx = float(p.get("center_lat", 0))
                cy = float(p.get("center_lng", 0))
                r = float(p.get("radius_km", 20))
                if haversine(end_lat, end_lng, cx, cy) < r:
                    return False, "destination_in_forbidden_zone"

        # 3. max_haul_distance
        for c in constraints:
            if c.get("type") != "max_haul_distance":
                continue
            max_km = float(c.get("params", {}).get("max_km", 9999))
            if haul_km > max_km:
                return False, "haul_distance_exceeded"

        # 4. max_deadhead_distance
        for c in constraints:
            if c.get("type") != "max_deadhead_distance":
                continue
            max_km = float(c.get("params", {}).get("max_deadhead_km", 9999))
            if distance_km > max_km:
                return False, "deadhead_distance_exceeded"

        # 5. time_restriction：执行区间不能穿过禁止时段
        if _violates_time_restriction(current_min, finish_min, constraints):
            return False, "violates_time_restriction"

        # 6. daily_home_deadline：完单后能否回家（软化：仅高罚分硬拒）
        for c in constraints:
            if c.get("type") != "daily_home_deadline":
                continue
            p = c.get("params", {})
            home_lat = p.get("home_lat")
            home_lng = p.get("home_lng")
            if home_lat is None or home_lng is None:
                continue
            try:
                deadline_hour = int(p.get("deadline_hour", 23))
            except (ValueError, TypeError):
                deadline_hour = 23
            day_index = current_min // 1440
            today_deadline = day_index * 1440 + deadline_hour * 60
            travel_home_min = _distance_to_minutes(
                haversine(end_lat, end_lng, float(home_lat), float(home_lng))
            )
            if finish_min + travel_home_min > today_deadline:
                c_severity = HeuristicLayer.classify_constraint_severity(c)
                if c_severity == "hard":
                    return False, "cannot_return_home_before_deadline"
                # 软约束：允许通过，由评分层经济惩罚

        # 7. scheduled_event：订单结束后必须仍能完成 pickup -> home -> stay chain。
        for event in self._build_scheduled_event_info(state, constraints):
            home_deadline = event["home_deadline_min"]
            release_min = event["release_min"]
            if current_min >= release_min:
                continue
            if current_min >= home_deadline:
                return False, "order_during_scheduled_event_stay"

            travel_to_pickup = _distance_to_minutes(
                haversine(
                    end_lat,
                    end_lng,
                    event["pickup_lat"],
                    event["pickup_lng"],
                )
            )
            arrive_pickup = max(finish_min + travel_to_pickup, event["pickup_min"])
            depart_pickup = arrive_pickup + event["pickup_wait_minutes"]
            travel_home = _distance_to_minutes(
                haversine(
                    event["pickup_lat"],
                    event["pickup_lng"],
                    event["home_lat"],
                    event["home_lng"],
                )
            )
            if depart_pickup + travel_home > home_deadline:
                return False, "order_would_miss_scheduled_event"

        # 8. monthly_visit：完单后能到达到访点（仅检查未完成的 open task）
        _has_open_visit = any(
            t.get("type") == "monthly_visit" for t in state.open_tasks
        )
        if _has_open_visit:
            for c in constraints:
                if c.get("type") != "monthly_visit_requirement":
                    continue
                p = c.get("params", {})
                visit_lat = p.get("target_lat")
                visit_lng = p.get("target_lng")
                if visit_lat is None or visit_lng is None:
                    continue
                try:
                    required_days = int(p.get("min_visit_days", 1))
                except (TypeError, ValueError):
                    required_days = 1
                total_days = max(1, math.ceil(horizon / 1440))
                current_day = current_min // 1440 + 1
                remaining_days_after_today = max(0, total_days - current_day)
                # 月度到访不是每天硬性任务；只有收官压力明显不足时才硬拒。
                if total_days < required_days or remaining_days_after_today >= required_days:
                    continue
                travel_to_visit = _distance_to_minutes(
                    haversine(end_lat, end_lng, float(visit_lat), float(visit_lng))
                )
                day_end = ((current_min // 1440) + 1) * 1440
                if finish_min + travel_to_visit > day_end:
                    return False, "cannot_reach_visit_point_today"

        # 9. daily_rest：软化处理——不再硬拒，由评分层经济惩罚
        # RuleLayer._check_forced_rest 在剩余时间仅够补休时仍会强制 wait
        for c in constraints:
            if c.get("type") != "daily_rest":
                continue
            params = c.get("params", {})
            required_min = int(params.get("min_continuous_minutes", 0))
            if required_min <= 0:
                continue
            current_rest = state.get_max_continuous_rest_today(current_min)
            if current_rest >= required_min:
                continue
            need = required_min - current_rest
            remaining_after_order = 1440 - (finish_min % 1440)
            if remaining_after_order < need:
                # 不再 return False，由 soft_penalty_cost 处理
                pass

        # 10. 链式失败检查：接单会导致 mandatory_cargo(<12h) 无法完成 → 硬拒
        _mandatory_pickups = HeuristicLayer._build_mandatory_pickup_info(
            state, constraints, None)
        if _mandatory_pickups:
            for mp in _mandatory_pickups:
                deadline = mp.get("window_end", mp["activation_min"])
                if deadline < current_min:
                    continue
                if deadline - current_min > 720:
                    continue
                travel_to_pickup = _distance_to_minutes(
                    haversine(end_lat, end_lng, mp["pickup_lat"], mp["pickup_lng"])
                )
                if finish_min + travel_to_pickup > deadline:
                    return False, "order_would_miss_mandatory_cargo"

        return True, ""

    def _decide_impl(self, driver_id: str) -> dict:
        # 1. 获取状态
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        current_min = int(status.get("simulation_progress_minutes", 0))
        horizon = _get_horizon_minutes(status)

        # 2. 初始化/更新 StateTracker
        state = self._get_state(driver_id)
        state.refresh_from_history(self._api, driver_id)

        # 2.5 从历史填充 pattern_analyzer（Task #4，仅一次）
        if driver_id not in self._pattern_filled:
            self._fill_pattern_from_history(driver_id)
            self._pattern_filled.add(driver_id)

        # 3. 获取结构化约束
        preferences = status.get("preferences", [])
        constraints = self._preference_engine.get_constraints(driver_id, preferences)

        # 3.5 初始化月度规划器（Task #1）
        if state.monthly_planner is None and preferences:
            pref_text = " ".join(
                p.get("content", "") if isinstance(p, dict) else str(p)
                for p in preferences
            )
            # 也从已解析约束中提取 day_off_requirement
            for c in constraints:
                if c.get("type") == "day_off_requirement":
                    days_off = int(c.get("params", {}).get("min_days_off", 0))
                    if days_off > 0:
                        pref_text += f" 至少{days_off}天不接单"
                        break
            total_days = max(1, math.ceil(horizon / 1440))
            state.monthly_planner = MonthlyConstraintPlanner(pref_text, total_days)
            _logger.info("初始化月度规划器: required_idle=%d, planned=%s",
                         state.monthly_planner.required_idle_days,
                         state.monthly_planner.planned_idle_days)

        # 3.6 注册司机到 Token优化器 和 协调层（Task #6, #7）
        if driver_id not in self._registered_drivers:
            pref_text_for_register = " ".join(
                p.get("content", "") if isinstance(p, dict) else str(p)
                for p in preferences
            )
            self.token_optimizer.register_driver(driver_id, pref_text_for_register)
            self.coordination_layer.register_driver_profile(driver_id, pref_text_for_register)
            self._registered_drivers.add(driver_id)

        # 3.7（Task #4）首次生成战略计划
        if state.strategic_plan is None:
            _prefs_text = ""
            if preferences:
                _prefs_text = " | ".join(
                    p.get("content", "") if isinstance(p, dict) else str(p)
                    for p in preferences
                    if (p.get("content") if isinstance(p, dict) else p)
                )
            try:
                if horizon <= 2 * 1440:
                    plan = self._strategic_planner._fallback_plan(
                        constraints,
                        horizon_minutes=horizon,
                    )
                else:
                    plan = self._strategic_planner.generate_plan(
                        driver_id, _prefs_text, constraints, status,
                        horizon_minutes=horizon,
                    )
                state.set_strategic_plan(plan)
                if state.monthly_planner is not None:
                    state.monthly_planner.integrate_strategic_plan(plan)
                self._logger.info(
                    "战略计划已生成 driver=%s must_do=%d risk_windows=%d",
                    driver_id,
                    len(state.open_tasks),
                    len(state.risk_register),
                )
            except Exception as e:
                self._logger.warning("生成战略计划失败 driver=%s: %s", driver_id, e)

        self._sync_constraint_tasks(state, constraints, horizon)

        # 3.8（Task #4）刷新任务进度
        try:
            _hist_resp = self._api.query_decision_history(driver_id, -1) or {}
            if isinstance(_hist_resp, dict):
                _history_records = _hist_resp.get("records", []) or []
            else:
                _history_records = list(_hist_resp) if _hist_resp else []
            state.update_task_progress(current_min, _history_records)
        except Exception as e:
            self._logger.warning("刷新任务进度失败 driver=%s: %s", driver_id, e)

        # 3.9（Task #4）紧急战略任务处理
        _urgent_tasks = state.has_urgent_tasks(current_min)
        if _urgent_tasks:
            _urgent_decision = self._handle_urgent_tasks(
                _urgent_tasks, status, state, constraints
            )
            if _urgent_decision:
                _validated = self._validate_action(_urgent_decision, status, constraints)
                if _validated:
                    self._logger.info(
                        "紧急战略任务触发 driver=%s action=%s",
                        driver_id, _validated.get("action")
                    )
                    self._update_state_after_decision(state, _validated, current_min)
                    return _validated

        # 4. 规则层快速路径（先不查货源的规则）
        rule_decision = self._rule_layer.evaluate(status, state, constraints, None)
        if rule_decision is not None:
            rule_decision = self._validate_action(rule_decision, status, constraints)
            if rule_decision is None:
                self._logger.info("规则层决策被安全闸拦截，跳过继续下一层 driver=%s", driver_id)
            else:
                # 无货源数据时拒绝 take_order：无法校验 cargo 存在性、可见性、时效
                if rule_decision.get("action") == "take_order":
                    cargo_id = str(rule_decision.get("params", {}).get("cargo_id", ""))
                    mandatory_ids = {
                        str(t.get("target") or t.get("cargo_id") or "")
                        for t in state.open_tasks
                        if t.get("type") == "mandatory_cargo"
                    }
                    if cargo_id not in mandatory_ids:
                        self._logger.info("规则层(无货源) 产出 take_order 但无法校验，跳过等查货后处理 driver=%s", driver_id)
                        rule_decision = None
                if rule_decision is not None:
                    self._logger.info("规则层决策 driver=%s action=%s", driver_id, rule_decision.get("action"))
                    self._update_state_after_decision(state, rule_decision, current_min)
                    return rule_decision

        # 5. 查询货源
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        items = cargo_resp.get("items", [])

        # 更新 current_min（查询可能消耗时间）
        status_after_query = self._api.get_driver_status(driver_id)
        current_min = int(status_after_query.get("simulation_progress_minutes", current_min))

        # 5.5 记录货源频率统计（Task #2）
        state.record_cargo_seen(current_min, len(items))

        # 5.55 熟货优先检查（Task #5）：若有 mandatory_cargo 任务且目标货源已出现，立即接单
        if state.open_tasks:
            for _task in state.open_tasks:
                if _task.get("type") == "mandatory_cargo":
                    _target_cargo_id = str(_task.get("target", ""))
                    if not _target_cargo_id:
                        continue
                    # 在可见货源中查找指定货源
                    for _item in items:
                        _cargo_info = _item.get("cargo", _item)
                        if str(_cargo_info.get("cargo_id", "")) == _target_cargo_id:
                            # 找到指定熟货！立即接单
                            _mandatory_decision = {
                                "action": "take_order",
                                "params": {"cargo_id": _target_cargo_id}
                            }
                            _validated = self._validate_action(
                                _mandatory_decision, status_after_query, constraints
                            )
                            _feasible = _validated and self._validate_take_order_feasibility(
                                _target_cargo_id, items, status_after_query
                            )
                            if _feasible:
                                ok, reason = self._validate_take_order_constraints(
                                    _target_cargo_id, items, status_after_query, constraints, state)
                                if not ok:
                                    self._logger.info(
                                        "熟货约束校验失败 driver=%s cargo=%s reason=%s",
                                        driver_id, _target_cargo_id, reason)
                                    _feasible = False
                            if _feasible:
                                self._logger.info(
                                    "熟货优先触发 driver=%s cargo_id=%s",
                                    driver_id, _target_cargo_id
                                )
                                self._update_state_after_decision(state, _validated, current_min)
                                return _validated
                            break

        # 5.56 mandatory_cargo 阻塞：即将激活的熟货不允许被其他单抢占
        for _task in state.open_tasks:
            if _task.get("type") != "mandatory_cargo":
                continue
            _target_cargo_id = str(_task.get("target", ""))
            if not _target_cargo_id:
                continue
            _activation = _to_int_or_zero(_task.get("activation_min"))
            if _activation <= 0:
                continue
            _time_to_activation = _activation - current_min
            if _time_to_activation > 360:  # >6h，不阻塞
                continue
            # 从 constraints 取 pickup 坐标
            _mp_lat = _mp_lng = None
            for c in constraints:
                if c.get("type") == "mandatory_cargo":
                    p = c.get("params", {})
                    if str(p.get("cargo_id", "")) == _target_cargo_id:
                        _mp_lat = p.get("pickup_lat")
                        _mp_lng = p.get("pickup_lng")
                        break
            if _mp_lat is None or _mp_lng is None:
                continue
            _dist_to_pickup = haversine(
                float(status_after_query.get("current_lat", 0)),
                float(status_after_query.get("current_lng", 0)),
                float(_mp_lat), float(_mp_lng),
            )
            _travel_to_pickup = _distance_to_minutes(_dist_to_pickup)
            # 如果赶不到 pickup 点，reposition 过去
            if _dist_to_pickup > 1.0 and current_min + _travel_to_pickup + 30 >= _activation:
                _block_decision = {
                    "action": "reposition",
                    "params": {"latitude": float(_mp_lat), "longitude": float(_mp_lng)},
                }
                _block_validated = self._validate_action(
                    _block_decision, status_after_query, constraints)
                if _block_validated:
                    self._logger.info(
                        "mandatory_cargo 阻塞: reposition 到 pickup driver=%s cargo=%s dist=%.1fkm",
                        driver_id, _target_cargo_id, _dist_to_pickup)
                    self._update_state_after_decision(state, _block_validated, current_min)
                    return _block_validated
            # 如果已经在 pickup 附近或刚好能到，等在原地不要接其他单
            if _time_to_activation <= 120:
                _wait_dur = max(1, min(60, _time_to_activation))
                self._logger.info(
                    "mandatory_cargo 阻塞: 等待激活 driver=%s cargo=%s wait=%dmin",
                    driver_id, _target_cargo_id, _wait_dur)
                _wait_decision = {"action": "wait", "params": {"duration_minutes": _wait_dur}}
                self._update_state_after_decision(state, _wait_decision, current_min)
                return _wait_decision

        # 5.6 更新未来机会预测器统计（Task #4）
        try:
            hour_now = _get_hour_of_day(current_min)
            if items:
                avg_price_now = sum(
                    float(it.get("cargo", {}).get("price", 0)) for it in items
                ) / len(items)
            else:
                avg_price_now = 0.0
            self.opportunity_predictor.update_stats(hour_now, len(items), avg_price_now)
        except Exception as e:
            self._logger.warning("更新机会预测统计失败: %s", e)

        # 5.7 主动空驶判定（Task #5）：在 LLM 层之前
        try:
            hour_now = _get_hour_of_day(current_min)
            self.reposition_layer.record_cargo_quality(hour_now, items)
            self.reposition_layer.update_heatmap(state)
            if self.reposition_layer.should_reposition(status_after_query, items, constraints):
                suggestion = self.reposition_layer.get_reposition_suggestion(
                    status_after_query, state)
                if suggestion and suggestion.get("net_value", 0) > 50:
                    decision = {
                        "action": "reposition",
                        "params": {
                            "latitude": float(suggestion["target_lat"]),
                            "longitude": float(suggestion["target_lng"]),
                        },
                    }
                    decision = self._validate_action(decision, status_after_query, constraints)
                    if decision is not None:
                        self._logger.info(
                            "主动空驶 driver=%s -> (%.2f,%.2f) net_value=%.1f",
                            driver_id, suggestion["target_lat"],
                            suggestion["target_lng"], suggestion["net_value"])
                        self._update_state_after_decision(state, decision, current_min)
                        return decision
                    else:
                        self._logger.info("主动空驶被安全闸拦截，跳过 driver=%s", driver_id)
        except Exception as e:
            self._logger.warning("主动空驶判定异常: %s", e)

        # 6. 再次检查规则层（带货源信息）
        rule_decision = self._rule_layer.evaluate(status_after_query, state, constraints, items)
        if rule_decision is not None:
            rule_decision = self._validate_action(rule_decision, status_after_query, constraints)
            if rule_decision is None:
                self._logger.info("规则层决策(带货源)被安全闸拦截，跳过继续下一层 driver=%s", driver_id)
            else:
                # take_order horizon 校验（有 items 时用完整校验）
                if rule_decision.get("action") == "take_order":
                    cargo_id = rule_decision.get("params", {}).get("cargo_id", "")
                    if not self._validate_take_order_feasibility(cargo_id, items, status_after_query):
                        self._logger.info("规则层 take_order 可行性校验失败，跳过 driver=%s", driver_id)
                        rule_decision = None
                    else:
                        ok, reason = self._validate_take_order_constraints(
                            cargo_id, items, status_after_query, constraints, state)
                        if not ok:
                            self._logger.info("规则层 take_order 约束校验失败 driver=%s cargo=%s reason=%s",
                                              driver_id, cargo_id, reason)
                            rule_decision = None
                if rule_decision is not None:
                    self._logger.info("规则层决策(带货源) driver=%s action=%s", driver_id, rule_decision.get("action"))
                    self._update_state_after_decision(state, rule_decision, current_min)
                    return rule_decision

        # 7. 无货源时：评估 wait value 而非默认 wait 30
        if not items:
            # bounding_box 冷启动：长时间无货时主动空驶到约束区域中心
            _bb_center = None
            for c in constraints:
                if c.get("type") == "spatial_restrict" and c.get("params", {}).get("type") == "bounding_box":
                    bb = c["params"].get("bounding_box", {})
                    _lat_min = bb.get("lat_min")
                    _lat_max = bb.get("lat_max")
                    _lng_min = bb.get("lng_min")
                    _lng_max = bb.get("lng_max")
                    if all(v is not None for v in [_lat_min, _lat_max, _lng_min, _lng_max]):
                        _bb_center = ((_lat_min + _lat_max) / 2, (_lng_min + _lng_max) / 2)
                    break
            if _bb_center is not None:
                _bb_dist = haversine(
                    float(status_after_query.get("current_lat", 0)),
                    float(status_after_query.get("current_lng", 0)),
                    _bb_center[0], _bb_center[1])
                # 不在 bounding_box 中心附近时，空驶过去
                if _bb_dist > 20:
                    _bb_decision = {
                        "action": "reposition",
                        "params": {"latitude": _bb_center[0], "longitude": _bb_center[1]},
                    }
                    _bb_validated = self._validate_action(
                        _bb_decision, status_after_query, constraints)
                    if _bb_validated:
                        self._logger.info(
                            "bounding_box 冷启动: reposition 到中心 driver=%s (%.2f,%.2f)",
                            driver_id, _bb_center[0], _bb_center[1])
                        self._update_state_after_decision(state, _bb_validated, current_min)
                        return _bb_validated
            wait_val = HeuristicLayer.estimate_wait_value(
                current_min, state, constraints, horizon, self.opportunity_predictor)
            wait_duration = 30 if wait_val > -100 else 60
            decision = {"action": "wait", "params": {"duration_minutes": wait_duration}}
            self._update_state_after_decision(state, decision, current_min)
            return decision

        # 7.5 时间窗预筛选（Task #3）
        items = self._time_window_optimizer.prescreen_cargo_by_feasibility(
            items, status_after_query, constraints)

        # 7.6 多司机竞争过滤（Task #7）
        items = self.coordination_layer.filter_competitive_cargo(driver_id, items)

        # 8. 启发式评分
        candidates = self._heuristic_layer.score_and_rank(
            items, status_after_query, state, constraints, top_n=12)

        if not candidates:
            decision = {"action": "wait", "params": {"duration_minutes": 30}}
            self._update_state_after_decision(state, decision, current_min)
            return decision

        # 评估决策复杂度（供 reviewer 和 token_optimizer 使用）
        candidates = self.profit_search_layer.rank_candidates(
            candidates, status_after_query, state, constraints,
            self.opportunity_predictor,
        )

        # RL 增强重排（可选）
        if self._rl_layer is not None and self._rl_layer.should_use_rl(driver_id, state, candidates):
            candidates = self._rl_layer.rank_candidates(
                candidates, status_after_query, state, constraints,
            )
        _decision_complexity = "low"
        if len(candidates) > 3:
            _decision_complexity = "medium"
        if candidates and candidates[0].get("penalty_score", 0) > 0:
            _decision_complexity = "high"
        if len(candidates) >= 2 and candidates[0]["score"] - candidates[1]["score"] < 10:
            _decision_complexity = "high"

        remaining_min = horizon - current_min
        _remaining_steps = max(1, remaining_min // 60)

        # 同步 StateTracker 的 token 用量到 token_optimizer
        self.token_optimizer.usage_tracker[driver_id] = state.total_tokens_used

        if self._rl_layer is not None:
            _rl_wait = self._rl_layer.select_wait_action(
                candidates, status_after_query, state, constraints
            )
            if _rl_wait is not None:
                decision = {
                    "action": "wait",
                    "params": _rl_wait.get("params", {"duration_minutes": 30}),
                }
                self._logger.info(
                    "RL direct wait driver=%s wait_prob=%.3f cargo_prob=%.3f",
                    driver_id,
                    float(_rl_wait.get("rl_wait_prob", 0.0)),
                    float(_rl_wait.get("rl_best_cargo_prob", 0.0)),
                )
                decision = self._maybe_review_decision(
                    decision, candidates, state, status_after_query,
                    constraints, current_min, driver_id,
                    _decision_complexity, _remaining_steps, items,
                )
                self._update_state_after_decision(state, decision, current_min, candidates)
                self.coordination_layer.record_decision(
                    driver_id,
                    "",
                    decision.get("action", ""),
                )
                return decision

        # 9. 快速接单路径：基于短视野利润搜索的安全/吃罚候选分离
        # 优先使用 RL 选择（如果可用）
        if self._rl_layer is not None:
            _rl_chosen = self._rl_layer.select_best(candidates)
            if _rl_chosen is not None:
                _chosen = _rl_chosen
            else:
                _chosen = self.profit_search_layer.select_best(candidates)
        else:
            _chosen = self.profit_search_layer.select_best(candidates)
        best_true_net = self._best_true_net(candidates)
        best_profit_score = (
            ProfitSearchLayer._candidate_value(_chosen)
            if _chosen is not None else float("-inf")
        )
        best_net_profit = max((float(c.get("net_profit", 0.0)) for c in candidates), default=0.0)
        if _chosen is None or best_profit_score <= 0 or best_net_profit <= 0:
            wait_duration = min(60, max(1, remaining_min))
            decision = {"action": "wait", "params": {"duration_minutes": wait_duration}}
            self._logger.info(
                "profit search wait driver=%s best_score=%.1f best_true_net=%.1f best_net_profit=%.1f",
                driver_id,
                best_profit_score,
                best_true_net,
                best_net_profit,
            )
            self._update_state_after_decision(state, decision, current_min, candidates)
            self.coordination_layer.record_decision(
                driver_id,
                "",
                decision.get("action", ""),
            )
            return decision

        _cid = _chosen["cargo_id"]
        _chosen_tn = _chosen.get("true_net", _chosen["score"])
        _chosen_ps = ProfitSearchLayer._candidate_value(_chosen)
        _chosen_profit = float(_chosen.get("net_profit", 0.0))
        if _chosen.get("hard_penalty", 0) == 0 and _chosen_ps > 0 and _chosen_profit > 0:
            decision = self._validated_take_order_decision(
                _chosen, items, status_after_query, constraints, state
            )
            if decision is not None and not self._reviewer.should_review(
                state, candidates, decision, current_min
            ):
                self._logger.info(
                    "profit search fast take_order driver=%s cargo=%s score=%.1f true_net=%.1f downstream=%.1f",
                    driver_id,
                    _cid,
                    _chosen_ps,
                    _chosen_tn,
                    float(_chosen.get("downstream_value", 0.0)),
                )
                self._update_state_after_decision(state, decision, current_min, candidates)
                self.coordination_layer.record_decision(
                    driver_id,
                    decision.get("params", {}).get("cargo_id", ""),
                    decision.get("action", ""),
                )
                return decision

        if _chosen is not None:
            _cid = _chosen["cargo_id"]
            _chosen_tn = _chosen.get("true_net", _chosen["score"])
            _chosen_ps = ProfitSearchLayer._candidate_value(_chosen)
            # 仅利润搜索分数 > 0 且无硬罚分时快速接单
            if _chosen.get("hard_penalty", 0) == 0 and _chosen_ps > 0:
                if state.is_token_budget_exceeded():
                    decision = {"action": "take_order", "params": {"cargo_id": _cid}}
                    if not self._validate_take_order_feasibility(_cid, items, status_after_query):
                        decision = {"action": "wait", "params": {"duration_minutes": 30}}
                    else:
                        ok, reason = self._validate_take_order_constraints(
                            _cid, items, status_after_query, constraints, state)
                        if not ok:
                            self._logger.info("快速接单约束校验失败 driver=%s cargo=%s reason=%s",
                                              driver_id, _cid, reason)
                            decision = {"action": "wait", "params": {"duration_minutes": 30}}
                    decision = self._maybe_review_decision(
                        decision, candidates, state, status_after_query,
                        constraints, current_min, driver_id,
                        _decision_complexity, _remaining_steps, items,
                    )
                    self._logger.info("快速接单(token降级) driver=%s cargo=%s true_net=%.1f",
                                      driver_id, _cid, _chosen_tn)
                    self._update_state_after_decision(state, decision, current_min, candidates)
                    return decision

        # 10. Token 预算降级：超阈值时不调 LLM
        if state.is_token_budget_exceeded():
            _chosen = self.profit_search_layer.select_best(candidates)
            if (
                _chosen
                and ProfitSearchLayer._candidate_value(_chosen) > 0
                and float(_chosen.get("net_profit", 0.0)) > 0
                and _chosen.get("hard_penalty", 0) == 0
            ):
                _cid = _chosen["cargo_id"]
                if self._validate_take_order_feasibility(_cid, items, status_after_query):
                    ok, reason = self._validate_take_order_constraints(
                        _cid, items, status_after_query, constraints, state)
                    if ok:
                        decision = {"action": "take_order", "params": {"cargo_id": _cid}}
                    else:
                        self._logger.info("token降级约束校验失败 driver=%s cargo=%s reason=%s",
                                          driver_id, _cid, reason)
                        decision = {"action": "wait", "params": {"duration_minutes": 30}}
                else:
                    decision = {"action": "wait", "params": {"duration_minutes": 30}}
            else:
                decision = {"action": "wait", "params": {"duration_minutes": 60}}
            decision = self._maybe_review_decision(
                decision, candidates, state, status_after_query,
                constraints, current_min, driver_id,
                _decision_complexity, _remaining_steps, items,
            )
            self._update_state_after_decision(state, decision, current_min, candidates)
            return decision

        # 10.5 Token预算智能分配检查（Task #6）
        if not self.token_optimizer.should_use_llm(driver_id, _decision_complexity, _remaining_steps):
            _chosen = self.profit_search_layer.select_best(candidates)
            if (
                _chosen
                and ProfitSearchLayer._candidate_value(_chosen) > 0
                and float(_chosen.get("net_profit", 0.0)) > 0
                and _chosen.get("hard_penalty", 0) == 0
            ):
                _cid = _chosen["cargo_id"]
                if self._validate_take_order_feasibility(_cid, items, status_after_query):
                    ok, reason = self._validate_take_order_constraints(
                        _cid, items, status_after_query, constraints, state)
                    if ok:
                        decision = {"action": "take_order", "params": {"cargo_id": _cid}}
                    else:
                        self._logger.info("token优化约束校验失败 driver=%s cargo=%s reason=%s",
                                          driver_id, _cid, reason)
                        decision = {"action": "wait", "params": {"duration_minutes": 30}}
                else:
                    decision = {"action": "wait", "params": {"duration_minutes": 30}}
            else:
                decision = {"action": "wait", "params": {"duration_minutes": 60}}

            decision = self._maybe_review_decision(
                decision, candidates, state, status_after_query,
                constraints, current_min, driver_id,
                _decision_complexity, _remaining_steps, items,
            )

            self._logger.info("Token智能分配跳过LLM driver=%s complexity=%s strategy=%s",
                              driver_id, _decision_complexity,
                              self.token_optimizer.suggest_strategy(driver_id, _remaining_steps))
            self._update_state_after_decision(state, decision, current_min, candidates)
            self.coordination_layer.record_decision(
                driver_id,
                decision.get("params", {}).get("cargo_id", ""),
                decision.get("action", "")
            )
            return decision

        # 10.6（Task #4）DecisionReviewer 审核启发式提议
        try:
            _heuristic_proposal = {
                "action": "take_order",
                "params": {"cargo_id": candidates[0]["cargo_id"]},
            }
            if self._reviewer.should_review(state, candidates, _heuristic_proposal, current_min):
                # 高风险审核：受 token_optimizer 控制（接近预算时跳过）
                if self.token_optimizer.should_use_llm(
                    driver_id, _decision_complexity, _remaining_steps,
                    context="review_high_risk",
                ):
                    _review_context = self._build_review_context(
                        state, status_after_query, constraints,
                        _heuristic_proposal, candidates,
                    )
                    _review_result = self._reviewer.review(_review_context)
                    state.last_llm_review_minute = current_min

                    if not _review_result.get("approve", True):
                        _reviewer_decision = self._convert_review_to_decision(_review_result)
                        _validated = self._validate_reviewer_decision(
                            _reviewer_decision, candidates, status_after_query, constraints,
                            items, state,
                        )
                        if _validated:
                            # Advisory-only: 仅在 reviewer 有量化更优方案时接受否决
                            if not self._should_accept_reviewer_override(
                                _validated, _heuristic_proposal, candidates,
                                state, current_min,
                            ):
                                self._logger.info(
                                    "Reviewer 否决被忽略(无更优方案) driver=%s reason=%s",
                                    driver_id, _review_result.get("reason", ""),
                                )
                            else:
                                if _validated.get("action") == "wait":
                                    dur = int(_validated.get("params", {}).get("duration_minutes", 30))
                                    if dur > 60:
                                        _validated["params"]["duration_minutes"] = 60
                                    state.consecutive_review_waits += 1
                                else:
                                    state.consecutive_review_waits = 0
                                state.last_review_reason = _review_result.get("reason", "")
                                self._logger.info(
                                    "Reviewer 否决启发式 driver=%s action=%s reason=%s consecutive_waits=%d",
                                    driver_id, _validated.get("action"),
                                    _review_result.get("reason", ""),
                                    state.consecutive_review_waits,
                                )
                                self._update_state_after_decision(
                                    state, _validated, current_min, candidates
                                )
                                self.coordination_layer.record_decision(
                                    driver_id,
                                    _validated.get("params", {}).get("cargo_id", ""),
                                    _validated.get("action", ""),
                                )
                                return _validated
                        else:
                            self._logger.info(
                                "Reviewer 建议不合法，回退到 LLM/启发式 driver=%s",
                                driver_id,
                            )
        except Exception as e:
            self._logger.warning("Reviewer 审核异常 driver=%s: %s", driver_id, e)

        # 11. LLM 层决策
        decision = self._llm_layer.decide(
            candidates, status_after_query, state, constraints,
            raw_candidates=items,
            pattern_analyzer=self.pattern_analyzer,
            opportunity_predictor=self.opportunity_predictor,
        )
        self._logger.info("LLM决策 driver=%s action=%s", driver_id, decision.get("action"))

        # 记录 token
        # (token 由 EmbeddedDecisionEnvironment 自动追踪，这里仅更新本地计数)

        decision = self._validate_action(decision, status_after_query, constraints)
        if decision is None:
            self._logger.info("LLM决策被安全闸拦截，fallback wait driver=%s", driver_id)
            decision = {"action": "wait", "params": {"duration_minutes": 30}}
        # 统一接单 horizon + 约束校验
        if decision.get("action") == "take_order":
            cargo_id = decision.get("params", {}).get("cargo_id", "")
            if not self._validate_take_order_feasibility(cargo_id, items, status_after_query):
                decision = {"action": "wait", "params": {"duration_minutes": 30}}
            else:
                ok, reason = self._validate_take_order_constraints(
                    cargo_id, items, status_after_query, constraints, state)
                if not ok:
                    self._logger.info("LLM take_order 约束校验失败 driver=%s cargo=%s reason=%s",
                                      driver_id, cargo_id, reason)
                    decision = {"action": "wait", "params": {"duration_minutes": 30}}
        self._update_state_after_decision(state, decision, current_min, candidates)

        # Task #7: 记录决策到协调层
        self.coordination_layer.record_decision(
            driver_id,
            decision.get("params", {}).get("cargo_id", ""),
            decision.get("action", "")
        )

        return decision

    def _handle_urgent_tasks(self, urgent_tasks: list[dict], status: dict,
                             state: "StateTracker", constraints: list[dict]) -> dict | None:
        """处理紧急战略任务，生成优先动作"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        cur_lat = float(status.get("current_lat", 0))
        cur_lng = float(status.get("current_lng", 0))
        day_index = current_min // 1440 + 1
        minute_in_day = current_min % 1440  # 当天已过的分钟数

        for task in sorted(urgent_tasks, key=lambda t: t.get("priority", 99)):
            task_type = task.get("type", "")

            if task_type == "mandatory_cargo":
                # D009场景：指定熟货必接
                # 如果快到激活时间，移动到 pickup 点附近等待
                activation = _to_int_or_zero(task.get("activation_min"))
                pickup_lat = task.get("pickup_lat")
                pickup_lng = task.get("pickup_lng")
                if pickup_lat is None:
                    for c in constraints:
                        if c.get("type") == "mandatory_cargo":
                            p = c.get("params", {})
                            if str(p.get("cargo_id", "")) == str(task.get("target", "")):
                                pickup_lat = p.get("pickup_lat")
                                pickup_lng = p.get("pickup_lng")
                                break
                if pickup_lat is not None and pickup_lng is not None:
                    pickup_lat = float(pickup_lat)
                    pickup_lng = float(pickup_lng)
                    dist = haversine(cur_lat, cur_lng, pickup_lat, pickup_lng)
                    if dist > 1.0:
                        return {"action": "reposition",
                                "params": {"latitude": pickup_lat, "longitude": pickup_lng}}
                if activation > 0 and current_min < activation:
                    return {"action": "wait",
                            "params": {"duration_minutes": max(1, min(30, activation - current_min))}}
                # Once the mandatory cargo is active and we are already at pickup,
                # let the cargo-query path run so the visible cargo can be taken.
                return None

            elif task_type == "scheduled_event":
                # D010场景：家事临时约定（接配偶 → 回家 → 静止等待）
                pickup_min = _to_int_or_zero(task.get("pickup_min"))
                home_deadline = _to_int_or_zero(task.get("home_deadline_min"))
                release_min = _to_int_or_zero(task.get("release_min"))
                pickup_wait = _to_int_or_zero(task.get("pickup_wait_minutes"), 10)
                pickup_lat = task.get("pickup_lat")
                pickup_lng = task.get("pickup_lng")
                home_lat = task.get("home_lat")
                home_lng = task.get("home_lng")

                # 也从 constraints 中补充坐标
                if pickup_lat is None:
                    for c in constraints:
                        if c.get("type") == "scheduled_event":
                            p = c.get("params", {})
                            pickup_lat = p.get("pickup_lat")
                            pickup_lng = p.get("pickup_lng")
                            home_lat = home_lat or p.get("home_lat")
                            home_lng = home_lng or p.get("home_lng")
                            pickup_min = pickup_min or _to_int_or_zero(p.get("pickup_min"))
                            home_deadline = home_deadline or _to_int_or_zero(p.get("home_deadline_min"))
                            release_min = release_min or _to_int_or_zero(p.get("release_min"))
                            pickup_wait = pickup_wait or _to_int_or_zero(p.get("pickup_wait_minutes"), 10)
                            if pickup_lat:
                                break

                if pickup_lat is None or pickup_lng is None:
                    continue

                pickup_lat = float(pickup_lat)
                pickup_lng = float(pickup_lng)
                pickup_dist = haversine(cur_lat, cur_lng, pickup_lat, pickup_lng)

                # 阶段1: pickup 之前 → 移动到 pickup 点
                if pickup_min > 0 and current_min < pickup_min:
                    travel_to_pickup = _distance_to_minutes(pickup_dist)
                    prep_buffer = max(15, min(60, pickup_wait + 5))
                    if (
                        pickup_dist > 1.0
                        and current_min + travel_to_pickup + prep_buffer >= pickup_min
                    ):
                        return {"action": "reposition",
                                "params": {"latitude": pickup_lat, "longitude": pickup_lng}}
                    if pickup_dist <= 1.0 and pickup_min - current_min <= prep_buffer:
                        return {"action": "wait",
                                "params": {"duration_minutes": max(1, min(60, pickup_min - current_min))}}
                    continue

                # 阶段2: pickup ~ home_deadline → 先完成 pickup 等待，再移动到 home
                if home_deadline > 0 and current_min < home_deadline and not task.get("pickup_wait_done"):
                    if pickup_dist > 1.0:
                        return {"action": "reposition",
                                "params": {"latitude": pickup_lat, "longitude": pickup_lng}}
                    wait_until = _to_int_or_zero(task.get("pickup_wait_until"))
                    if wait_until <= 0:
                        task["pickup_wait_until"] = current_min + max(1, pickup_wait)
                        return {"action": "wait",
                                "params": {"duration_minutes": max(1, pickup_wait)}}
                    if current_min < wait_until:
                        return {"action": "wait",
                                "params": {"duration_minutes": max(1, wait_until - current_min)}}
                    task["pickup_wait_done"] = True

                if home_lat is not None and home_lng is not None:
                    home_lat = float(home_lat)
                    home_lng = float(home_lng)
                    home_dist = haversine(cur_lat, cur_lng, home_lat, home_lng)
                    if home_deadline > 0 and current_min < home_deadline:
                        if home_dist > 1.0:
                            return {"action": "reposition",
                                    "params": {"latitude": home_lat, "longitude": home_lng}}
                        return {"action": "wait",
                                "params": {"duration_minutes": max(1, min(60, home_deadline - current_min))}}

                    # 阶段3: home_deadline ~ release → 留在 home
                    if release_min > 0 and current_min < release_min:
                        if home_dist > 1.0:
                            return {"action": "reposition",
                                    "params": {"latitude": home_lat, "longitude": home_lng}}
                        return {"action": "wait",
                                "params": {"duration_minutes": max(1, min(480, release_min - current_min))}}

                continue

            elif task_type == "monthly_visit":
                # D010场景：到访指定点
                # 如果deadline快到了，主动reposition到目标
                target_str = task.get("target", "")
                if "lat=" in target_str and "lng=" in target_str:
                    try:
                        parts = target_str.split(",")
                        t_lat = float(parts[0].split("=")[1])
                        t_lng = float(parts[1].split("=")[1])
                        # 检查是否已经在目标附近（5km≈0.05度）
                        if (abs(cur_lat - t_lat) < 0.05 and abs(cur_lng - t_lng) < 0.05):
                            # 已到达，等待一段时间确认到访
                            return {
                                "action": "wait",
                                "params": {"duration_minutes": 30}
                            }
                        else:
                            # 需要移动到目标点
                            return {
                                "action": "reposition",
                                "params": {
                                    "latitude": t_lat,
                                    "longitude": t_lng,
                                }
                            }
                    except (ValueError, IndexError):
                        pass

            elif task_type == "daily_home_deadline":
                # D009场景：每日回家deadline
                # 优先从任务字段直接读取，回退到旧的字符串解析
                deadline_hour = task.get("deadline_hour")
                if deadline_hour is None:
                    deadline_hour = 23
                    target_info = task.get("target", "")
                    if "hour=" in target_info:
                        try:
                            deadline_hour = int(target_info.split("hour=")[1].split(",")[0])
                        except (ValueError, IndexError):
                            pass
                try:
                    deadline_hour = int(deadline_hour)
                except (ValueError, TypeError):
                    deadline_hour = 23

                deadline_minute_today = deadline_hour * 60
                time_left = deadline_minute_today - minute_in_day

                if 0 < time_left < 180:  # 距回家deadline不到3小时
                    # 优先从任务字段直接读取坐标，回退到旧的多级查找
                    home_lat = task.get("home_lat")
                    home_lng = task.get("home_lng")
                    if home_lat is None and state.strategic_plan:
                        for hvp in state.strategic_plan.get("home_or_visit_plan", []):
                            if hvp.get("day") == 0 or hvp.get("day") == day_index:
                                home_lat = hvp.get("target_lat")
                                home_lng = hvp.get("target_lng")
                                break
                    if home_lat is None:
                        for c in constraints:
                            if c.get("type") == "daily_home_deadline":
                                params = c.get("params", {})
                                home_lat = params.get("home_lat") or params.get("latitude")
                                home_lng = params.get("home_lng") or params.get("longitude")
                                if home_lat:
                                    break

                    if home_lat is not None and home_lng is not None:
                        try:
                            home_lat = float(home_lat)
                            home_lng = float(home_lng)
                            radius_km = 1.0
                            for c in constraints:
                                if c.get("type") == "daily_home_deadline":
                                    try:
                                        radius_km = float(
                                            c.get("params", {}).get("radius_km", radius_km)
                                        )
                                    except (ValueError, TypeError):
                                        pass
                                    break
                            home_dist = haversine(cur_lat, cur_lng, home_lat, home_lng)
                            # 如果已经在家附近，等待
                            if home_dist <= radius_km:
                                return {
                                    "action": "wait",
                                    "params": {"duration_minutes": 30}
                                }
                            else:
                                return {
                                    "action": "reposition",
                                    "params": {
                                        "latitude": home_lat,
                                        "longitude": home_lng,
                                    }
                                }
                        except (ValueError, TypeError):
                            pass

        return None

    def _build_review_context(self, state: "StateTracker", status: dict,
                              constraints: list[dict], heuristic_proposal: dict,
                              candidates: list[dict]) -> dict:
        """构建 DecisionReviewer 的审核上下文"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        day_index = current_min // 1440 + 1

        context = {
            "current_status": {
                "latitude": status.get("current_lat"),
                "longitude": status.get("current_lng"),
                "current_minute": current_min,
                "day": day_index,
                "total_income": state.total_income,
                "total_orders": state.total_orders,
                "total_deadhead_km": state.total_deadhead_km,
            },
            "strategic_plan_summary": {
                "must_do_tasks_remaining": len(state.open_tasks),
                "tasks": state.open_tasks[:5],
                "daily_intent": state.get_daily_intent(day_index),
            },
            "task_progress": {
                "open": len(state.open_tasks),
                "completed": len(state.completed_tasks),
                "idle_days_done": state.completed_idle_days,
            },
            "constraints_summary": [
                {"type": c.get("type"), "severity": c.get("severity")}
                for c in constraints[:10]
            ],
            "heuristic_proposal": heuristic_proposal,
            "top_candidates": [
                {
                    "cargo_id": c.get("cargo_id") or c.get("cargo", {}).get("cargo_id"),
                    "true_net": c.get("true_net", c.get("score", 0)),
                    "net_profit": c.get("net_profit") or c.get("profit", 0),
                    "deadhead_km": c.get("deadhead_km") or c.get("distance_km", 0),
                    "has_soft_penalty": c.get("has_soft_penalty", False),
                    "penalty_cost": c.get("penalty_score", 0),
                    "cost_time_minutes": c.get("cost_time_minutes")
                        or c.get("cargo", {}).get("cost_time_minutes", 0),
                }
                for c in candidates[:5]
            ],
        }
        return context

    def _convert_review_to_decision(self, review_result: dict) -> dict:
        """将 reviewer 的审核结果转换为标准决策格式"""
        action = review_result.get("action", "wait")
        params: dict = {}

        if action == "take_order":
            cargo_id = review_result.get("cargo_id")
            if cargo_id:
                params["cargo_id"] = str(cargo_id)
            else:
                action = "wait"
                params["duration_minutes"] = 30
        elif action == "reposition":
            lat = review_result.get("latitude")
            lng = review_result.get("longitude")
            if lat is not None and lng is not None:
                params["latitude"] = float(lat)
                params["longitude"] = float(lng)
            else:
                action = "wait"
                params["duration_minutes"] = 30
        else:
            action = "wait"
            params["duration_minutes"] = int(review_result.get("duration_minutes") or 30)

        return {"action": action, "params": params}

    def _validate_reviewer_decision(self, decision: dict, candidates: list[dict],
                                    status: dict, constraints: list[dict],
                                    items: list[dict] | None = None,
                                    state: "StateTracker | None" = None) -> dict | None:
        """校验 reviewer 产生的动作，与 _validate_action 同等标准"""
        validated = self._validate_action(decision, status, constraints)
        if validated and validated.get("action") == "take_order":
            cargo_id = str(validated.get("params", {}).get("cargo_id", ""))
            valid_ids = set()
            for c in candidates:
                cid = c.get("cargo_id") or c.get("cargo", {}).get("cargo_id", "")
                if cid:
                    valid_ids.add(str(cid))
            if cargo_id not in valid_ids:
                return None  # cargo_id 不在可见候选中，安全拒绝
            # 综合硬校验：空间约束、回家 deadline、家事窗口等
            if items is not None and state is not None:
                ok, reason = self._validate_take_order_constraints(
                    cargo_id, items, status, constraints, state)
                if not ok:
                    self._logger.info("Reviewer take_order 约束校验失败 cargo=%s reason=%s",
                                      cargo_id, reason)
                    return None
        return validated

    def _try_reviewer_override(self, decision: dict, candidates: list[dict],
                               state: StateTracker, status: dict,
                               constraints: list[dict], current_min: int,
                               driver_id: str,
                               items: list[dict] | None = None) -> dict:
        """尝试用 DecisionReviewer 审核决策，返回（可能被否决后的）最终决策。
        用于快速接单/token降级等路径。"""
        try:
            _proposal = {
                "action": decision.get("action"),
                "params": decision.get("params", {}),
            }
            if self._reviewer.should_review(state, candidates, _proposal, current_min):
                # 快速/token降级路径：不额外检查 token 预算，直接审核
                _ctx = self._build_review_context(
                    state, status, constraints, _proposal, candidates,
                )
                _result = self._reviewer.review(_ctx)
                state.last_llm_review_minute = current_min
                if not _result.get("approve", True):
                    _reviewer_dec = self._convert_review_to_decision(_result)
                    _validated = self._validate_reviewer_decision(
                        _reviewer_dec, candidates, status, constraints,
                        items, state,
                    )
                    if _validated:
                        self._logger.info(
                            "Reviewer 否决(快速路径) driver=%s action=%s reason=%s",
                            driver_id, _validated.get("action"),
                            _result.get("reason", ""),
                        )
                        return _validated
        except Exception as e:
            self._logger.warning("Reviewer(快速路径)异常 driver=%s: %s", driver_id, e)
        return decision

    def _should_accept_reviewer_override(
        self,
        reviewer_decision: dict,
        current_decision: dict,
        candidates: list[dict],
        state: "StateTracker",
        current_min: int,
    ) -> bool:
        """Reviewer 仅在有量化更优替代方案时才否决。"""
        reviewer_cargo = reviewer_decision.get("params", {}).get("cargo_id", "")
        current_cargo = current_decision.get("params", {}).get("cargo_id", "")

        if reviewer_cargo:
            # Reviewer 建议了另一个货源——检查其 true_net 是否更高
            rev_cand = next(
                (c for c in candidates if c.get("cargo_id") == reviewer_cargo), None)
            cur_cand = next(
                (c for c in candidates if c.get("cargo_id") == current_cargo), None)
            if rev_cand and cur_cand:
                rev_tn = rev_cand.get("true_net", rev_cand.get("score", -9999))
                cur_tn = cur_cand.get("true_net", cur_cand.get("score", -9999))
                return rev_tn > cur_tn
            return reviewer_cargo != current_cargo  # 未知候选，保守接受

        if reviewer_decision.get("action") == "wait":
            # Reviewer 说等——仅在 true_net < 0 或有紧迫任务时接受
            cur_cand = next(
                (c for c in candidates if c.get("cargo_id") == current_cargo), None)
            cur_tn = cur_cand.get("true_net", cur_cand.get("score", 0)) if cur_cand else 0
            if cur_tn < 0:
                return True
            has_tight_task = hasattr(state, "open_tasks") and any(
                (t.get("deadline_minute", 0) or 0) - current_min < 360
                for t in state.open_tasks
                if t.get("deadline_minute")
            )
            return has_tight_task

        return True  # 其他情况（reposition 等）保守接受

    def _maybe_review_decision(self, decision: dict, candidates: list[dict],
                               state: StateTracker, status: dict,
                               constraints: list[dict], current_min: int,
                               driver_id: str,
                               decision_complexity: str = "low",
                               remaining_steps: int = 100,
                               items: list[dict] | None = None) -> dict:
        """统一的 reviewer 审核入口，所有决策路径都应经过此方法。
        遵守 token_optimizer 的 95% 阈值：接近预算时仍允许审核。
        连续 wait >= 3 次时暂停审核 180 分钟。"""
        try:
            _proposal = {
                "action": decision.get("action"),
                "params": decision.get("params", {}),
            }
            if not self._reviewer.should_review(state, candidates, _proposal, current_min):
                return decision
            # 检查 token 预算：review_high_risk 在 <95% 时允许
            if not self.token_optimizer.should_use_llm(
                driver_id, decision_complexity, remaining_steps,
                context="review_high_risk",
            ):
                return decision
            _ctx = self._build_review_context(
                state, status, constraints, _proposal, candidates,
            )
            _result = self._reviewer.review(_ctx)
            state.last_llm_review_minute = current_min
            if not _result.get("approve", True):
                _reviewer_dec = self._convert_review_to_decision(_result)
                _validated = self._validate_reviewer_decision(
                    _reviewer_dec, candidates, status, constraints,
                    items, state,
                )
                if _validated:
                    # Advisory-only: 仅在 reviewer 有量化更优方案时接受否决
                    if not self._should_accept_reviewer_override(
                        _validated, decision, candidates, state, current_min,
                    ):
                        self._logger.info(
                            "Reviewer 否决被忽略(无更优方案) driver=%s reason=%s",
                            driver_id, _result.get("reason", ""),
                        )
                        state.consecutive_review_waits = 0
                        return decision
                    if _validated.get("action") == "wait":
                        dur = int(_validated.get("params", {}).get("duration_minutes", 30))
                        if dur > 60:
                            _validated["params"]["duration_minutes"] = 60
                        state.consecutive_review_waits += 1
                    else:
                        state.consecutive_review_waits = 0
                    state.last_review_reason = _result.get("reason", "")
                    self._logger.info(
                        "Reviewer 否决 driver=%s action=%s reason=%s consecutive_waits=%d",
                        driver_id, _validated.get("action"),
                        _result.get("reason", ""),
                        state.consecutive_review_waits,
                    )
                    return _validated
            else:
                state.consecutive_review_waits = 0
        except Exception as e:
            self._logger.warning("DecisionReviewer 异常 driver=%s: %s", driver_id, e)
        return decision

    def _update_state_after_decision(self, state: StateTracker, decision: dict,
                                     current_min: int, candidates: list[dict] | None = None) -> None:
        """决策后更新本地状态跟踪。"""
        action = decision.get("action", "")
        params = decision.get("params", {})

        if action == "wait":
            duration = int(params.get("duration_minutes", 1))
            state.record_wait(current_min, duration)
        elif action == "reposition":
            # 估算到达时间
            state.record_reposition(current_min)
        elif action == "take_order" and candidates:
            cargo_id = params.get("cargo_id", "")
            candidate = next((c for c in candidates if c.get("cargo_id") == cargo_id), None)
            if candidate:
                state.record_take_order(
                    current_min,
                    candidate.get("price", 0),
                    candidate.get("deadhead_km", 0),
                    candidate.get("haul_km", 0),
                    candidate.get("end", {}).get("lat", 0),
                    candidate.get("end", {}).get("lng", 0),
                )
                # Task #4: 记录到历史模式分析器
                try:
                    hour = _get_hour_of_day(current_min)
                    cargo_type = candidate.get("cargo_name", "未知") or "未知"
                    profit = float(candidate.get("net_profit", 0))
                    self.pattern_analyzer.record_decision(
                        hour, cargo_type, profit, "take_order")
                except Exception as e:
                    self._logger.warning("记录决策模式失败: %s", e)

    def _fill_pattern_from_history(self, driver_id: str) -> None:
        """从历史决策记录回填 pattern_analyzer（Task #4）。"""
        try:
            hist = self._api.query_decision_history(driver_id, -1)
            records = hist.get("records", [])
            for rec in records:
                action = rec.get("action", {})
                action_name = action.get("action", "")
                if action_name != "take_order":
                    continue
                result = rec.get("result", {})
                if not result.get("accepted"):
                    continue

                # 解析仿真分钟以推算 hour
                sim_end = 0
                if "simulation_progress_minutes" in result:
                    sim_end = int(result["simulation_progress_minutes"])
                elif "simulation_end_time" in rec:
                    try:
                        sim_end = _wall_to_sim_minutes(
                            rec["simulation_end_time"] + ":00")
                    except Exception:
                        pass
                hour = _get_hour_of_day(sim_end)

                # 推算 profit: 优先用 result 中的 revenue/income/price，
                # 否则用 deadhead+haul 估算成本
                profit = 0.0
                for key in ("net_profit", "revenue", "income", "price"):
                    if key in result:
                        try:
                            profit = float(result[key])
                            if key in ("revenue", "income", "price"):
                                deadhead = float(result.get("pickup_deadhead_km", 0))
                                haul = float(result.get("haul_distance_km", 0))
                                profit -= (deadhead + haul) * _COST_PER_KM_DEFAULT
                            break
                        except Exception:
                            continue

                cargo_type = (
                    result.get("cargo_name")
                    or result.get("cargo_type")
                    or action.get("params", {}).get("cargo_id", "未知")
                )
                self.pattern_analyzer.record_decision(
                    hour, str(cargo_type), profit, "take_order")
        except Exception as e:
            self._logger.warning("填充历史模式失败 driver=%s: %s", driver_id, e)
