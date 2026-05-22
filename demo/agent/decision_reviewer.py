from __future__ import annotations
import json
import logging
from typing import Any, TYPE_CHECKING

from simkit.ports import SimulationApiPort

if TYPE_CHECKING:
    from .model_decision_service import StateTracker

logger = logging.getLogger(__name__)


DECISION_REVIEW_SYSTEM_PROMPT = """你是货运决策审核员。你的任务是审核启发式系统推荐的动作，判断它是否会破坏司机的长期目标。

审核视角：
1. 这单会不会让司机错过指定熟货？
2. 接了之后还能不能在截止时间前回家？
3. 现在是不是应该为了每日休息继续等待？
4. 今天是不是计划的整天休息日？
5. 空驶是否会打破月度空驶上限？
6. 高额罚分任务是不是快到 deadline 了？

决策原则：
- 高额固定罚分 > 短期运费收益
- 不要为了当前高利润牺牲 deadline
- 不确定时选择等待或靠近关键地点

重要——advisory-only 规则：
- 你只提供建议，不直接否决。如果启发式系统的决策有正向净利润(true_net > 0)，
  除非你能指出一个明确更好的替代方案（更高true_net的货源），否则应该批准(approve=true)。
- 不要因为"风险不确定"就否决正向收益的决策。
- 如果你要否决，必须在 cargo_id 中指定一个更好的替代货源，或说明具体的风险量化依据。

只输出JSON：
{
  "approve": bool,
  "action": "take_order|wait|reposition",
  "cargo_id": str|null,
  "duration_minutes": int|null,
  "latitude": float|null,
  "longitude": float|null,
  "reason": str (简短),
  "risk": str (简短)
}"""


class DecisionReviewer:
    """决策审核员 - 审核启发式动作是否破坏长期目标"""

    def __init__(self, api: SimulationApiPort):
        self._api = api

    def should_review(self, state: "StateTracker", candidates: list[dict],
                      heuristic_decision: dict, current_min: int) -> bool:
        """判断是否需要触发 LLM 审核"""

        # 安全阀：连续 reviewer 输出 wait >= 3 次，暂停审核 180 分钟
        if state.consecutive_review_waits >= 3:
            if state.last_llm_review_minute >= 0 and current_min - state.last_llm_review_minute < 180:
                return False
            # 超过 180 分钟，重置计数器允许重新审核
            state.consecutive_review_waits = 0

        # 没有 open_tasks 且没有硬时间/位置风险时，不触发审核
        _has_hard_risk = False
        if state.strategic_plan:
            home_visits = state.strategic_plan.get("home_or_visit_plan", [])
            if home_visits and heuristic_decision.get("action") == "take_order":
                _has_hard_risk = True
        if not state.open_tasks and not _has_hard_risk:
            return False

        # 条件1：存在 open_tasks 中 deadline < 12小时 (720分钟)
        if state.open_tasks:
            for task in state.open_tasks:
                deadline_raw = task.get("deadline_minute")
                try:
                    deadline = int(deadline_raw) if deadline_raw is not None else 0
                except (ValueError, TypeError):
                    deadline = 0
                if deadline > 0 and deadline - current_min < 720:
                    return True

        # 条件2：top1/top2 分差 < 10
        if len(candidates) >= 2:
            scores = [c.get("score", 0) for c in candidates[:2]]
            if len(scores) == 2 and abs(scores[0] - scores[1]) < 10:
                return True

        # 条件3：候选动作结束后远离 home/visit 点
        # (简化：如果有 home_or_visit_plan 且当前有 take_order 提议)
        if state.strategic_plan:
            home_visits = state.strategic_plan.get("home_or_visit_plan", [])
            if home_visits and heuristic_decision.get("action") == "take_order":
                return True  # 有回家/到访计划时，接单需审核

        # 条件4：偏好已有落后风险（open_tasks 数量多且月度过半）
        if state.open_tasks and current_min > 22320:  # 过了月中 (31*1440/2)
            if len(state.open_tasks) > len(state.completed_tasks):
                return True

        # 条件5：单笔耗时 > 6小时 (360分钟)
        if heuristic_decision.get("action") == "take_order":
            # 从candidates中查找对应货源的耗时
            cargo_id = heuristic_decision.get("params", {}).get("cargo_id")
            if cargo_id:
                for c in candidates:
                    cid = c.get("cargo_id") or c.get("cargo", {}).get("cargo_id")
                    if str(cid) == str(cargo_id):
                        cost_time = c.get("cost_time_minutes") or c.get("cargo", {}).get("cost_time_minutes", 0)
                        if cost_time and int(cost_time) > 360:
                            return True
                        break

        # 条件6：每天开始（day boundary，前60分钟内）
        if current_min % 1440 < 60:
            if state.open_tasks:  # 仅在有待完成任务时
                return True

        # 条件7：距上次 reviewer 调用 > 120 分钟 且存在 open_tasks
        if (state.open_tasks and
            state.last_llm_review_minute >= 0 and
            current_min - state.last_llm_review_minute > 120):
            return True

        return False

    def review(self, context: dict) -> dict:
        """审核启发式提议的动作，返回审核结果"""
        try:
            resp = self._api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": DECISION_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
                ],
                "response_format": {"type": "json_object"},
            })
            content = resp["choices"][0]["message"]["content"]
            result = json.loads(content)
            return self._normalize_result(result)
        except Exception as e:
            logger.warning("DecisionReviewer LLM call failed: %s", e)
            # 降级：直接批准启发式决策
            return {"approve": True, "action": "", "cargo_id": None,
                    "duration_minutes": None, "latitude": None,
                    "longitude": None, "reason": "reviewer_fallback", "risk": ""}

    def _normalize_result(self, result: dict) -> dict:
        """确保审核结果包含所有必要字段"""
        defaults = {
            "approve": True,
            "action": "",
            "cargo_id": None,
            "duration_minutes": None,
            "latitude": None,
            "longitude": None,
            "reason": "",
            "risk": "",
        }
        for key, default in defaults.items():
            if key not in result:
                result[key] = default
        # 确保 approve 是 bool
        result["approve"] = bool(result["approve"])
        return result
