"""三层混合决策架构 Agent：规则层 → 启发式层 → LLM 层。"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_MONTH_TOTAL_MINUTES = 43200  # 30天
_SPEED_KM_H = 60.0
_COST_PER_KM_DEFAULT = 1.5
_TOKEN_BUDGET = 1_000_000  # 月度 token 预算上限
_TOKEN_DEGRADE_RATIO = 0.80

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

    def initialize_from_history(self, api: SimulationApiPort, driver_id: str) -> None:
        """首次调用时从历史决策恢复状态。"""
        if self._initialized:
            return
        self._initialized = True
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
        for d in range(current_day_idx + 1):
            if d not in self.active_days:
                idle_count += 1
        self.completed_idle_days = idle_count

    def _process_history_record(self, rec: dict) -> None:
        action = rec.get("action", {})
        action_name = action.get("action", "")
        result = rec.get("result", {})
        step_start = rec.get("step", 0)
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
            day_idx = _get_day_index(sim_end)
            self.orders_by_day[day_idx] = self.orders_by_day.get(day_idx, 0) + 1
            self.active_days.add(day_idx)
            # 记录空间信息
            pos_after = rec.get("position_after", {})
            if pos_after:
                self._update_spatial(pos_after.get("lat", 0), pos_after.get("lng", 0), 1.0)
            deadhead = result.get("pickup_deadhead_km", 0)
            self.total_deadhead_km += float(deadhead)
            haul = result.get("haul_distance_km", 0)
            self.total_mileage_km += float(deadhead) + float(haul)

        elif action_name == "wait":
            params = action.get("params", {})
            duration = int(params.get("duration_minutes", 0))
            if sim_end > 0 and duration > 0:
                wait_start = sim_end - duration
                self.wait_intervals.append((wait_start, sim_end))

        elif action_name == "reposition":
            day_idx = _get_day_index(sim_end)
            self.active_days.add(day_idx)

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

    def record_reposition(self, sim_min: int) -> None:
        day_idx = _get_day_index(sim_min)
        self.active_days.add(day_idx)
        self.last_action_end_min = sim_min

    def get_max_continuous_rest_today(self, current_min: int) -> int:
        """获取今天（到当前时刻）的最大连续休息分钟数。"""
        day_idx = _get_day_index(current_min)
        day_start = day_idx * 1440
        day_end = current_min
        max_rest = 0
        for s, e in self.wait_intervals:
            # 计算与今天重叠的部分
            overlap_s = max(s, day_start)
            overlap_e = min(e, day_end)
            if overlap_e > overlap_s:
                max_rest = max(max_rest, overlap_e - overlap_s)
        return max_rest

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
        if driver_id in self._constraints_cache:
            return self._constraints_cache[driver_id]

        constraints = self._parse_with_llm(preferences)
        self._constraints_cache[driver_id] = constraints
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
                        "type": "约束类型(daily_rest/time_restriction/spatial_restrict/mileage_cap/day_off_requirement/time_window_location/cargo_category_ban/max_orders_per_day/max_haul_distance/max_deadhead_distance/custom)",
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
            elif "整天" in content or "休息日" in content or "不接单" in content and "天" in content:
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
                 constraints: list[dict], items: list[dict]) -> dict | None:
        """返回决策 dict 或 None（需继续后续层）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        hour = _get_hour_of_day(current_min)
        remaining = _MONTH_TOTAL_MINUTES - current_min

        # 0. 月度规划空闲日检查（Task #1）
        idle_result = self._check_planned_idle_day(current_min, state)
        if idle_result is not None:
            return idle_result

        # 1. 月末收官：剩余不足 60 分钟无法完成任何订单
        if remaining <= 0:
            return {"action": "wait", "params": {"duration_minutes": 0}}
        if remaining <= 60:
            wait_min = max(1, remaining)
            return self._wait(wait_min)

        # 2. 禁行时段检查
        block_result = self._check_time_block(hour, current_min, constraints)
        if block_result is not None:
            return block_result

        # 3. 强制休息检查
        rest_result = self._check_forced_rest(current_min, state, constraints)
        if rest_result is not None:
            return rest_result

        # 4. 窗口约束触发（必须前往指定位置）
        location_result = self._check_time_window_location(
            status, current_min, constraints)
        if location_result is not None:
            return location_result

        # 5. 深夜无货（22:00-06:00 且无有效货源）
        if (hour >= 22.0 or hour < 6.0) and not items:
            # 等到早上6点
            if hour >= 22.0:
                wait_until = ((_get_day_index(current_min) + 1) * 1440) + 360  # 次日6:00
            else:
                wait_until = (_get_day_index(current_min) * 1440) + 360  # 当日6:00
            wait_min = max(1, min(wait_until - current_min, remaining))
            return self._wait(wait_min)

        return None

    def _check_time_block(self, hour: float, current_min: int,
                          constraints: list[dict]) -> dict | None:
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
                wait_min = max(1, min(wait_min + 1, _MONTH_TOTAL_MINUTES - current_min))
                return self._wait(wait_min)
        return None

    def _check_forced_rest(self, current_min: int, state: StateTracker,
                           constraints: list[dict]) -> dict | None:
        """检查今天是否满足了最低连续休息要求。"""
        for c in constraints:
            if c.get("type") != "daily_rest":
                continue
            params = c.get("params", {})
            required_min = int(params.get("min_continuous_minutes", 0))
            if required_min <= 0:
                continue
            current_rest = state.get_max_continuous_rest_today(current_min)
            hour = _get_hour_of_day(current_min)
            # 如果今天快结束了但还没满足休息要求，强制休息
            remaining_today = 1440 - (current_min % 1440)
            if current_rest < required_min and remaining_today <= required_min + 60:
                need = required_min - current_rest
                remaining_total = _MONTH_TOTAL_MINUTES - current_min
                wait_min = max(1, min(need, remaining_today, remaining_total))
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

    def _check_planned_idle_day(self, current_min: int, state: StateTracker) -> dict | None:
        """检查今日是否为规划的空闲日或是否需要触发补救模式（Task #1）。"""
        planner = state.monthly_planner
        if planner is None or planner.required_idle_days <= 0:
            return None

        # 当前天数（1-based）
        current_day = _get_day_index(current_min) + 1
        remaining = _MONTH_TOTAL_MINUTES - current_min

        # 如果今天是规划的空闲日，直接等待到次日0点
        if planner.is_today_idle_day(current_day):
            # 计算到次日0点的分钟数
            minutes_in_day = current_min % 1440
            wait_until_next_day = 1440 - minutes_in_day
            wait_min = max(1, min(wait_until_next_day, remaining))
            _logger.info("规划空闲日: day=%d, wait=%d min", current_day, wait_min)
            return self._wait(wait_min)

        # 如果触发补救模式（剩余天数不够完成空闲日需求），强制等待
        if planner.should_trigger_rescue_mode(state.completed_idle_days, current_day):
            minutes_in_day = current_min % 1440
            wait_until_next_day = 1440 - minutes_in_day
            wait_min = max(1, min(wait_until_next_day, remaining))
            _logger.info("偏好补救模式: day=%d, completed_idle=%d, wait=%d min",
                         current_day, state.completed_idle_days, wait_min)
            return self._wait(wait_min)

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

    def score_and_rank(self, items: list[dict], status: dict, state: StateTracker,
                       constraints: list[dict], top_n: int = 5) -> list[dict]:
        """评分并返回 Top N 候选（Task #2 优化评分公式）。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        remaining = _MONTH_TOTAL_MINUTES - current_min

        # 计算月度进度比（Task #2）
        month_progress_ratio = current_min / _MONTH_TOTAL_MINUTES
        time_phase_multiplier = self._get_time_phase_multiplier(month_progress_ratio)

        scored = []

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

            # 过滤: 装货时间窗已过期
            if load_time and isinstance(load_time, list) and len(load_time) == 2:
                try:
                    window_end_min = _wall_to_sim_minutes(str(load_time[1]).strip())
                    deadhead_min = _distance_to_minutes(distance_km)
                    arrival_min = current_min + deadhead_min
                    if arrival_min > window_end_min:
                        continue
                except Exception:
                    pass

            # 过滤: 月内无法完成
            deadhead_min = _distance_to_minutes(distance_km)
            total_time = deadhead_min + cost_time
            if total_time > remaining:
                continue

            # 计算干线距离
            haul_km = haversine(start_lat, start_lng, end_lat, end_lng)

            # 净利润
            total_cost = (distance_km + haul_km) * self.cost_per_km
            net_profit = price - total_cost

            # 时间效率
            total_minutes = deadhead_min + cost_time
            time_efficiency = net_profit / max(total_minutes, 1)

            # 偏好合规性评估
            penalty_score = 0.0
            for c in constraints:
                ctype = c.get("type", "")
                params = c.get("params", {})
                p_amount = float(c.get("penalty_amount", 0))

                if ctype == "cargo_category_ban":
                    banned = params.get("banned_categories", [])
                    for b in banned:
                        if b in cargo_name:
                            penalty_score += p_amount
                            break

                elif ctype == "max_deadhead_distance":
                    max_km = float(params.get("max_deadhead_km", 9999))
                    if distance_km > max_km:
                        penalty_score += p_amount

                elif ctype == "max_haul_distance":
                    max_km = float(params.get("max_km", 9999))
                    if haul_km > max_km:
                        penalty_score += p_amount

                elif ctype == "spatial_restrict":
                    if params.get("type") == "bounding_box":
                        bb = params.get("bounding_box", {})
                        if not PreferenceEngine._in_bounding_box(end_lat, end_lng, bb):
                            penalty_score += p_amount
                    elif params.get("type") == "forbidden_circle":
                        cx = float(params.get("center_lat", 0))
                        cy = float(params.get("center_lng", 0))
                        r = float(params.get("radius_km", 20))
                        # 检查起点终点是否经过禁区
                        if haversine(end_lat, end_lng, cx, cy) < r:
                            penalty_score += p_amount
                        if haversine(start_lat, start_lng, cx, cy) < r:
                            penalty_score += p_amount

            # 聚类价值（替代原spatial_bonus）（Task #2）
            cluster_value = self._compute_cluster_value(end_lat, end_lng, state)

            # 偏好影响评估（Task #2）
            item_for_assess = {"total_minutes": total_minutes, "cost_time_minutes": cost_time}
            preference_impact = self._assess_preference_impact(item_for_assess, state, constraints)

            # 稀缺性因子（Task #2）
            scarcity_factor = self._get_scarcity_factor(item, state)

            # 新综合评分公式（Task #2）
            score = (
                time_efficiency * 60 * time_phase_multiplier
                - penalty_score * 2.0 * (1 + preference_impact)
                + cluster_value
                + net_profit * 0.15
                + scarcity_factor
            )

            scored.append({
                "cargo_id": cargo_id,
                "cargo_name": cargo_name,
                "price": round(price, 2),
                "net_profit": round(net_profit, 2),
                "deadhead_km": round(distance_km, 2),
                "haul_km": round(haul_km, 2),
                "total_minutes": total_minutes,
                "time_efficiency": round(time_efficiency, 4),
                "penalty_score": round(penalty_score, 2),
                "score": round(score, 4),
                "start": {"lat": start_lat, "lng": start_lng},
                "end": {"lat": end_lat, "lng": end_lng},
                "load_time": load_time,
                "cost_time_minutes": cost_time,
            })

        # 按 score 降序排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]


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
        remaining = _MONTH_TOTAL_MINUTES - current_min
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
                "净利润元": c["net_profit"],
                "空驶km": c["deadhead_km"],
                "总耗时分钟": c["total_minutes"],
                "时间效率": c["time_efficiency"],
                "违规罚款": c["penalty_score"],
                "评分": c["score"],
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
                        "你是高级货运调度员。结合历史模式、未来货源预测、"
                        "偏好约束时间线进行决策。优先保障偏好不违规，其次最大化收益。"
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
        if candidates and candidates[0].get("score", 0) > 0 and candidates[0].get("penalty_score", 0) == 0:
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

    def should_use_llm(self, driver_id: str, decision_complexity: str, remaining_steps: int) -> bool:
        """判断当前决策是否应该调用LLM"""
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

    def register_driver_profile(self, driver_id: str, preferences_text: str,
                                home_location: tuple = None):
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
        self.active_decisions[driver_id] = {
            "cargo_id": cargo_id, "action": action
        }
        # 更新统计
        if driver_id in self.driver_profiles:
            self.driver_profiles[driver_id]["total_orders"] += 1

    def filter_competitive_cargo(self, driver_id: str, candidates: list) -> list:
        """过滤已被其他司机接的货源（避免竞争）"""
        filtered = []
        for item in candidates:
            cargo_id = item.get("cargo", {}).get("id", "")
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

    def __init__(self, api: SimulationApiPort) -> None:
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
        # Task #5: 主动空驶层
        self.reposition_layer = ProactiveRepositionLayer()
        # 已经从历史填充过 pattern_analyzer 的 driver 集合
        self._pattern_filled: set[str] = set()
        # Task #6: Token预算智能分配器
        self.token_optimizer = TokenBudgetOptimizer()
        # Task #7: 多司机信息共享与协调层
        self.coordination_layer = MultiDriverCoordinationLayer()
        # 已注册到 token_optimizer / coordination_layer 的 driver 集合
        self._registered_drivers: set[str] = set()

    def _get_state(self, driver_id: str) -> StateTracker:
        if driver_id not in self._state_tracker:
            self._state_tracker[driver_id] = StateTracker()
        return self._state_tracker[driver_id]

    def decide(self, driver_id: str) -> dict:
        """主决策方法：初始化 → 状态更新 → 规则层 → 查询货源 → 启发式 → LLM"""
        try:
            return self._decide_impl(driver_id)
        except Exception as e:
            self._logger.error("决策异常 driver=%s: %s", driver_id, e, exc_info=True)
            return {"action": "wait", "params": {"duration_minutes": 60}}

    def _decide_impl(self, driver_id: str) -> dict:
        # 1. 获取状态
        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        current_min = int(status.get("simulation_progress_minutes", 0))

        # 2. 初始化/更新 StateTracker
        state = self._get_state(driver_id)
        state.initialize_from_history(self._api, driver_id)

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
            total_days = _MONTH_TOTAL_MINUTES // 1440
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

        # 4. 规则层快速路径（先不查货源的规则）
        rule_decision = self._rule_layer.evaluate(status, state, constraints, [])
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
                    self._logger.info(
                        "主动空驶 driver=%s -> (%.2f,%.2f) net_value=%.1f",
                        driver_id, suggestion["target_lat"],
                        suggestion["target_lng"], suggestion["net_value"])
                    self._update_state_after_decision(state, decision, current_min)
                    return decision
        except Exception as e:
            self._logger.warning("主动空驶判定异常: %s", e)

        # 6. 再次检查规则层（带货源信息）
        rule_decision = self._rule_layer.evaluate(status_after_query, state, constraints, items)
        if rule_decision is not None:
            self._logger.info("规则层决策(带货源) driver=%s action=%s", driver_id, rule_decision.get("action"))
            self._update_state_after_decision(state, rule_decision, current_min)
            return rule_decision

        # 7. 无货源时默认等待
        if not items:
            decision = {"action": "wait", "params": {"duration_minutes": 30}}
            self._update_state_after_decision(state, decision, current_min)
            return decision

        # 7.5 时间窗预筛选（Task #3）
        items = self._time_window_optimizer.prescreen_cargo_by_feasibility(
            items, status_after_query, constraints)

        # 7.6 多司机竞争过滤（Task #7）
        items = self.coordination_layer.filter_competitive_cargo(driver_id, items)

        # 8. 启发式评分
        candidates = self._heuristic_layer.score_and_rank(
            items, status_after_query, state, constraints, top_n=5)

        if not candidates:
            decision = {"action": "wait", "params": {"duration_minutes": 30}}
            self._update_state_after_decision(state, decision, current_min)
            return decision

        # 9. 快速接单路径：如果最优候选评分远超第二，且无违规，直接接单
        if (len(candidates) >= 1 and
                candidates[0]["score"] > 0 and
                candidates[0]["penalty_score"] == 0 and
                candidates[0]["net_profit"] > 50):
            if len(candidates) == 1 or candidates[0]["score"] > candidates[1]["score"] * 1.5:
                # Token 预算紧张时直接用启发式结果
                if state.is_token_budget_exceeded():
                    decision = {"action": "take_order", "params": {"cargo_id": candidates[0]["cargo_id"]}}
                    self._logger.info("快速接单(token降级) driver=%s cargo=%s",
                                      driver_id, candidates[0]["cargo_id"])
                    self._update_state_after_decision(state, decision, current_min,
                                                      candidates[0])
                    return decision

        # 10. Token 预算降级：超阈值时不调 LLM
        if state.is_token_budget_exceeded():
            if candidates and candidates[0]["score"] > 0 and candidates[0]["penalty_score"] == 0:
                decision = {"action": "take_order", "params": {"cargo_id": candidates[0]["cargo_id"]}}
            else:
                decision = {"action": "wait", "params": {"duration_minutes": 60}}
            self._update_state_after_decision(state, decision, current_min,
                                              candidates[0] if candidates else None)
            return decision

        # 10.5 Token预算智能分配检查（Task #6）
        # 评估决策复杂度
        _decision_complexity = "low"
        if len(candidates) > 3:
            _decision_complexity = "medium"
        if candidates and candidates[0].get("penalty_score", 0) > 0:
            _decision_complexity = "high"
        if len(candidates) >= 2 and candidates[0]["score"] - candidates[1]["score"] < 10:
            _decision_complexity = "high"  # 候选接近，需LLM精细决策

        # 估算剩余决策步数
        remaining_min = _MONTH_TOTAL_MINUTES - current_min
        _remaining_steps = max(1, remaining_min // 60)  # 粗估每60分钟一步

        # 同步 StateTracker 的 token 用量到 token_optimizer
        self.token_optimizer.usage_tracker[driver_id] = state.total_tokens_used

        if not self.token_optimizer.should_use_llm(driver_id, _decision_complexity, _remaining_steps):
            # Token预算建议不使用LLM，直接用启发式结果
            if candidates and candidates[0]["score"] > 0 and candidates[0]["penalty_score"] == 0:
                decision = {"action": "take_order", "params": {"cargo_id": candidates[0]["cargo_id"]}}
            else:
                decision = {"action": "wait", "params": {"duration_minutes": 60}}
            self._logger.info("Token智能分配跳过LLM driver=%s complexity=%s strategy=%s",
                              driver_id, _decision_complexity,
                              self.token_optimizer.suggest_strategy(driver_id, _remaining_steps))
            self._update_state_after_decision(state, decision, current_min,
                                              candidates[0] if candidates else None)
            # Task #7: 记录决策到协调层
            self.coordination_layer.record_decision(
                driver_id,
                decision.get("params", {}).get("cargo_id", ""),
                decision.get("action", "")
            )
            return decision

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

        self._update_state_after_decision(state, decision, current_min, candidates[0] if candidates else None)

        # Task #7: 记录决策到协调层
        self.coordination_layer.record_decision(
            driver_id,
            decision.get("params", {}).get("cargo_id", ""),
            decision.get("action", "")
        )

        return decision

    def _update_state_after_decision(self, state: StateTracker, decision: dict,
                                     current_min: int, top_candidate: dict | None = None) -> None:
        """决策后更新本地状态跟踪。"""
        action = decision.get("action", "")
        params = decision.get("params", {})

        if action == "wait":
            duration = int(params.get("duration_minutes", 1))
            state.record_wait(current_min, duration)
        elif action == "reposition":
            # 估算到达时间
            state.record_reposition(current_min)
        elif action == "take_order" and top_candidate:
            cargo_id = params.get("cargo_id", "")
            # 找到匹配的候选
            candidate = top_candidate
            if candidate and candidate.get("cargo_id") == cargo_id:
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
