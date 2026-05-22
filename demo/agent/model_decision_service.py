"""三层混合决策架构 Agent：规则层 → 启发式层 → LLM 层。"""

from __future__ import annotations

import json
import logging
import math
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
    R = 6371.0
    p1, l1, p2, l2 = map(math.radians, (lat1, lng1, lat2, lng2))
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    h = min(1.0, max(0.0, h))
    return 2 * R * math.asin(math.sqrt(h))


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
# MonthlyConstraintPlanner - 月度偏好约束前置规划器
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
        import re
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
        # 货源出现频率统计 (hour_slot -> count)
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
        for d in range(current_day_idx):
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
# HeuristicLayer - 启发式评分层
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicLayer:
    """对候选货源评分并排序，返回 Top-N。"""

    def __init__(self, cost_per_km: float = _COST_PER_KM_DEFAULT) -> None:
        self.cost_per_km = cost_per_km

    def score_and_rank(self, items: list[dict], status: dict, state: StateTracker,
                       constraints: list[dict], top_n: int = 5) -> list[dict]:
        """评分并返回 Top N 候选。"""
        current_min = int(status.get("simulation_progress_minutes", 0))
        lat = float(status.get("current_lat", 0))
        lng = float(status.get("current_lng", 0))
        remaining = _MONTH_TOTAL_MINUTES - current_min
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

            # 终点位置价值
            spatial_value = state.get_spatial_value(end_lat, end_lng)
            spatial_bonus = min(spatial_value * 0.01, 50.0)

            # 综合评分
            score = (
                time_efficiency * 60  # 每小时收益权重
                - penalty_score * 0.5  # 罚款惩罚
                + spatial_bonus  # 位置加分
                + (net_profit * 0.1)  # 绝对利润补偿
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
               constraints: list[dict]) -> dict:
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

        prompt = json.dumps(context, ensure_ascii=False)

        try:
            resp = self._api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": (
                        "你是货运调度决策器。根据状态和候选选择最优动作。"
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

        # 3. 获取结构化约束
        preferences = status.get("preferences", [])
        constraints = self._preference_engine.get_constraints(driver_id, preferences)

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

        # 11. LLM 层决策
        decision = self._llm_layer.decide(candidates, status_after_query, state, constraints)
        self._logger.info("LLM决策 driver=%s action=%s", driver_id, decision.get("action"))

        # 记录 token
        # (token 由 EmbeddedDecisionEnvironment 自动追踪，这里仅更新本地计数)

        self._update_state_after_decision(state, decision, current_min, candidates[0] if candidates else None)
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
