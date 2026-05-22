from __future__ import annotations
import json
import logging
import math
from typing import Any

from simkit.ports import SimulationApiPort

logger = logging.getLogger(__name__)


def _safe_int(val: Any, default: int = 0) -> int:
    """安全转 int，None/非数字返回 default。"""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


STRATEGIC_PLAN_SYSTEM_PROMPT = """你是一个货运战略规划师。根据司机偏好和约束，生成月度战略计划。

关键原则：
1. 高额罚分任务（指定货源、到访要求、家事事件）优先级最高
2. 必须为回家、休息、到访预留足够时间窗
3. 不确定时宁可保守（等待/靠近关键地点）
4. 只输出JSON，不要解释

输出格式：
{
  "hard_constraints": [{"type": str, "description": str, "penalty": int}],
  "soft_constraints": [{"type": str, "description": str}],
  "must_do_tasks": [{"type": str, "target": str, "deadline_minute": int, "priority": int}],
  "risk_windows": [{"description": str, "start_minute": int, "end_minute": int}],
  "rest_plan": [{"day": int, "type": "full_day|partial", "reason": str}],
  "home_or_visit_plan": [{"day": int, "target_lat": float, "target_lng": float, "deadline_minute": int}],
  "cargo_policy": {"preferred": [str], "avoid": [str], "distance_limit_km": int|null},
  "daily_strategy": [{"day": int, "intent": str, "priority_action": str}]
}"""


class StrategicPlanner:
    """月度战略规划器 - 每司机首次决策时生成长期计划"""

    def __init__(self, api: SimulationApiPort):
        self._api = api
        self._plan_cache: dict[str, dict] = {}  # driver_id -> plan

    def generate_plan(self, driver_id: str, preferences_text: str,
                      constraints: list[dict], initial_status: dict,
                      horizon_minutes: int = 43200) -> dict:
        """生成或返回缓存的战略计划"""
        if driver_id in self._plan_cache:
            return self._plan_cache[driver_id]
        plan = self._call_llm_for_plan(driver_id, preferences_text, constraints,
                                       initial_status, horizon_minutes)
        self._plan_cache[driver_id] = plan
        return plan

    def _call_llm_for_plan(self, driver_id: str, preferences_text: str,
                           constraints: list[dict], initial_status: dict,
                           horizon_minutes: int = 43200) -> dict:
        """调用 LLM 生成战略计划"""
        # 构建用户消息：包含偏好原文、约束列表、初始状态
        user_content = {
            "driver_id": driver_id,
            "preferences_raw": preferences_text,
            "constraints": constraints,
            "initial_position": {
                "latitude": initial_status.get("current_lat"),
                "longitude": initial_status.get("current_lng"),
            },
            "simulation_horizon_days": math.ceil(horizon_minutes / 1440),
            "simulation_total_minutes": horizon_minutes,
            "current_minute": int(initial_status.get("simulation_progress_minutes", 0)),
            "instruction": (
                "请根据以上司机偏好和约束，生成完整的月度战略计划。"
                "特别注意：高额罚分任务（指定货源必接、到访要求、家事事件、每日回家deadline）"
                "必须被识别为 must_do_tasks 并设定合理的 deadline_minute。"
                "rest_plan 需要覆盖月度休息天数要求。"
                "daily_strategy 为每天生成意图（earn/rest/visit/wait_for_cargo/go_home等）。"
            ),
        }

        try:
            resp = self._api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": STRATEGIC_PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            })
            content = resp["choices"][0]["message"]["content"]
            plan = json.loads(content)
            plan = self._normalize_plan(plan)
            # 关键：LLM 输出必须经过白名单清洗，防止幻觉任务
            return self._sanitize_plan(plan, constraints, horizon_minutes)
        except Exception as e:
            logger.warning("StrategicPlanner LLM call failed for %s: %s", driver_id, e)
            return self._fallback_plan(constraints, horizon_minutes)

    def _normalize_plan(self, plan: dict) -> dict:
        """确保计划包含所有必要字段，缺失的用空列表/None填充"""
        defaults = {
            "hard_constraints": [],
            "soft_constraints": [],
            "must_do_tasks": [],
            "risk_windows": [],
            "rest_plan": [],
            "home_or_visit_plan": [],
            "cargo_policy": {"preferred": [], "avoid": [], "distance_limit_km": None},
            "daily_strategy": [],
        }
        for key, default in defaults.items():
            if key not in plan:
                plan[key] = default
        return plan

    def _sanitize_plan(self, plan: dict, constraints: list[dict],
                       horizon_minutes: int) -> dict:
        """白名单清洗：must_do_tasks 只能来自真实 constraints，LLM 不能幻觉任务。

        LLM 可以自由输出 rest_plan / daily_strategy / cargo_policy 等策略字段，
        但 must_do_tasks 和 home_or_visit_plan 的任务条目必须由代码从 constraints 生成。
        """
        # 从 constraints 构建允许的任务白名单
        allowed_tasks: list[dict] = []
        allowed_home_plans: list[dict] = []

        for c in constraints:
            ctype = c.get("type", "")
            p = c.get("params", {})

            if ctype == "mandatory_cargo":
                cargo_id = str(p.get("cargo_id", ""))
                if not cargo_id:
                    continue
                try:
                    activation = int(p.get("activation_min", 0))
                except (ValueError, TypeError):
                    activation = 0
                allowed_tasks.append({
                    "type": "mandatory_cargo",
                    "target": cargo_id,
                    "activation_min": activation,
                    "deadline_minute": horizon_minutes,
                    "priority": 1,
                })

            elif ctype == "scheduled_event":
                pickup_min = _safe_int(p.get("pickup_min"))
                home_deadline = _safe_int(p.get("home_deadline_min"))
                release_min = _safe_int(p.get("release_min"))
                effective_deadline = home_deadline if home_deadline > 0 else (
                    release_min if release_min > 0 else (pickup_min // 1440 + 1) * 1440
                )
                allowed_tasks.append({
                    "type": "scheduled_event",
                    "target": p.get("description", "event"),
                    "deadline_minute": effective_deadline,
                    "priority": 1,
                    "pickup_lat": p.get("pickup_lat"),
                    "pickup_lng": p.get("pickup_lng"),
                    "home_lat": p.get("home_lat"),
                    "home_lng": p.get("home_lng"),
                    "pickup_min": pickup_min,
                    "home_deadline_min": home_deadline,
                    "release_min": release_min,
                })

            elif ctype == "monthly_visit_requirement":
                target_lat = p.get("target_lat")
                target_lng = p.get("target_lng")
                if target_lat is None or target_lng is None:
                    continue
                try:
                    min_days = int(p.get("min_visit_days", 1))
                except (ValueError, TypeError):
                    min_days = 1
                allowed_tasks.append({
                    "type": "monthly_visit",
                    "target": f"lat={target_lat},lng={target_lng}",
                    "min_visit_days": min_days,
                    "deadline_minute": horizon_minutes,
                    "priority": 2,
                })

            elif ctype == "daily_home_deadline":
                try:
                    deadline_hour = int(p.get("deadline_hour", 23))
                except (ValueError, TypeError):
                    deadline_hour = 23
                allowed_home_plans.append({
                    "day": 0,
                    "target_lat": p.get("home_lat", 0),
                    "target_lng": p.get("home_lng", 0),
                    "deadline_minute": deadline_hour * 60,
                })

        # 用白名单替换 LLM 幻觉的 must_do_tasks
        plan["must_do_tasks"] = allowed_tasks
        # 无条件覆盖 home_or_visit_plan，清除 LLM 幻觉的回家/到访计划
        plan["home_or_visit_plan"] = allowed_home_plans

        return plan

    def _fallback_plan(self, constraints: list[dict],
                       horizon_minutes: int = 43200) -> dict:
        """LLM调用失败时的降级计划 - 从约束中提取基本信息"""
        plan = {
            "hard_constraints": [],
            "soft_constraints": [],
            "must_do_tasks": [],
            "risk_windows": [],
            "rest_plan": [],
            "home_or_visit_plan": [],
            "cargo_policy": {"preferred": [], "avoid": [], "distance_limit_km": None},
            "daily_strategy": [],
        }
        # 从约束中提取关键任务
        for c in constraints:
            ctype = c.get("type", "")
            params = c.get("params", {})
            if ctype == "mandatory_cargo":
                plan["must_do_tasks"].append({
                    "type": "mandatory_cargo",
                    "target": str(params.get("cargo_id", "")),
                    "deadline_minute": horizon_minutes,
                    "priority": 1,
                })
            elif ctype == "scheduled_event":
                # 约束解析器产出 pickup_min/home_deadline_min/release_min
                try:
                    pickup_min_int = int(params.get("pickup_min", 0))
                except (ValueError, TypeError):
                    pickup_min_int = 0
                try:
                    home_deadline_int = int(params.get("home_deadline_min", 0))
                except (ValueError, TypeError):
                    home_deadline_int = 0
                try:
                    release_int = int(params.get("release_min", 0))
                except (ValueError, TypeError):
                    release_int = 0
                # 用 home_deadline_min 作为 deadline（事件须在此之前到家）
                effective_deadline = home_deadline_int if home_deadline_int > 0 else (
                    release_int if release_int > 0 else (pickup_min_int // 1440 + 1) * 1440
                )
                task_entry: dict[str, Any] = {
                    "type": "scheduled_event",
                    "target": params.get("description", "event"),
                    "deadline_minute": effective_deadline,
                    "priority": 1,
                    "pickup_lat": params.get("pickup_lat"),
                    "pickup_lng": params.get("pickup_lng"),
                    "home_lat": params.get("home_lat"),
                    "home_lng": params.get("home_lng"),
                    "pickup_min": pickup_min_int,
                    "home_deadline_min": home_deadline_int,
                    "release_min": release_int,
                }
                plan["must_do_tasks"].append(task_entry)
            elif ctype == "monthly_visit_requirement":
                # 约束解析器产出 target_lat/target_lng，非 latitude/longitude
                plan["must_do_tasks"].append({
                    "type": "monthly_visit",
                    "target": f"lat={params.get('target_lat')},lng={params.get('target_lng')}",
                    "min_visit_days": int(params.get("min_visit_days", 1)),
                    "deadline_minute": horizon_minutes,
                    "priority": 2,
                })
            elif ctype == "daily_home_deadline":
                plan["home_or_visit_plan"].append({
                    "day": 0,  # 每天
                    "target_lat": params.get("home_lat", 0),
                    "target_lng": params.get("home_lng", 0),
                    "deadline_minute": params.get("deadline_hour", 23) * 60,
                })
            elif ctype == "day_off_requirement":
                days_needed = params.get("min_days_off", 0)
                for i in range(days_needed):
                    plan["rest_plan"].append({
                        "day": 7 * (i + 1),  # 均匀分布
                        "type": "full_day",
                        "reason": "monthly_rest_requirement",
                    })
        return plan
