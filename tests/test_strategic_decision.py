"""战略决策模块单元测试。

覆盖以下修复点：
1. scheduled_event fallback 保留 release_min/home_deadline_min 并用 home_deadline_min 作 deadline
2. mandatory_cargo 绕过硬禁品类过滤
3. 快速/token降级路径经过 DecisionReviewer
4. monthly_visit 使用不同自然日统计 min_visit_days
5. StrategicPlanner fallback 保留 scheduled_event 的 pickup/home/release 字段
"""

import sys
import os
import types
import importlib

import pytest

# ── path setup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEMO_DIR)

# mock simkit only when the real demo package is unavailable.
try:
    importlib.import_module("simkit.ports")
except ModuleNotFoundError:
    _simkit = types.ModuleType("simkit")
    _simkit.__path__ = []  # make it a package so sub-module imports work
    _simkit.__package__ = "simkit"
    _ports = types.ModuleType("simkit.ports")
    _ports.__package__ = "simkit"
    _ports.SimulationApiPort = type("SimulationApiPort", (), {})
    _simkit.ports = _ports
    sys.modules["simkit"] = _simkit
    sys.modules["simkit.ports"] = _ports

from agent.model_decision_service import (
    ModelDecisionService,
    StateTracker,
    MonthlyConstraintPlanner,
    HeuristicLayer,
    _get_day_index,
    haversine,
    _distance_to_minutes,
)
from agent.strategic_planner import StrategicPlanner


# ═══════════════════════════════════════════════════════════════
# Fix #5 + #1: StrategicPlanner fallback 保留 scheduled_event 字段
# ═══════════════════════════════════════════════════════════════
class TestScheduledEventFallback:
    """_fallback_plan 应保留 scheduled_event 的完整字段并用 home_deadline_min 作 deadline。"""

    def _make_planner(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        return StrategicPlanner(api)

    def test_fallback_preserves_fields(self):
        planner = self._make_planner()
        constraints = [{
            "type": "scheduled_event",
            "params": {
                "pickup_lat": 22.5,
                "pickup_lng": 114.0,
                "home_lat": 23.0,
                "home_lng": 113.5,
                "pickup_min": 1000,
                "home_deadline_min": 2000,
                "release_min": 3000,
            },
        }]
        plan = planner._fallback_plan(constraints, 43200)
        tasks = [t for t in plan["must_do_tasks"] if t["type"] == "scheduled_event"]
        assert len(tasks) == 1
        t = tasks[0]
        assert t["pickup_lat"] == 22.5
        assert t["pickup_lng"] == 114.0
        assert t["home_lat"] == 23.0
        assert t["home_lng"] == 113.5
        assert t["pickup_min"] == 1000
        assert t["home_deadline_min"] == 2000
        assert t["release_min"] == 3000

    def test_fallback_uses_home_deadline_as_deadline(self):
        planner = self._make_planner()
        constraints = [{
            "type": "scheduled_event",
            "params": {
                "pickup_min": 1000,
                "home_deadline_min": 2000,
                "release_min": 3000,
            },
        }]
        plan = planner._fallback_plan(constraints, 43200)
        task = [t for t in plan["must_do_tasks"] if t["type"] == "scheduled_event"][0]
        assert task["deadline_minute"] == 2000  # home_deadline_min

    def test_fallback_release_min_as_second_choice(self):
        """home_deadline_min 为 0 时退化到 release_min。"""
        planner = self._make_planner()
        constraints = [{
            "type": "scheduled_event",
            "params": {
                "pickup_min": 1000,
                "home_deadline_min": 0,
                "release_min": 3000,
            },
        }]
        plan = planner._fallback_plan(constraints, 43200)
        task = [t for t in plan["must_do_tasks"] if t["type"] == "scheduled_event"][0]
        assert task["deadline_minute"] == 3000

    def test_scheduled_event_not_completed_before_deadline(self):
        """update_task_progress 不应在 deadline 之前完成 scheduled_event。"""
        state = StateTracker()
        state.open_tasks = [{
            "type": "scheduled_event",
            "target": "event",
            "deadline_minute": 2000,
            "release_min": 3000,
            "home_deadline_min": 2000,
            "pickup_min": 1000,
        }]
        # current_min=1500 < release_min=3000，不应完成
        state.update_task_progress(1500, [])
        assert len(state.open_tasks) == 1, "scheduled_event 不应在 deadline 前被标记完成"


# ═══════════════════════════════════════════════════════════════
# Fix #2: mandatory_cargo 绕过硬禁品类过滤
# ═══════════════════════════════════════════════════════════════
class TestMandatoryCargoBypass:
    """mandatory_cargo 目标应绕过 hard_banned 品类过滤。"""

    def test_mandatory_cargo_not_filtered_by_banned_category(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "mandatory_cargo",
            "target": "C001",
            "deadline_minute": 43200,
            "priority": 1,
        }]
        layer = HeuristicLayer()
        # 货源品类在 hard_banned 中，但 cargo_id 是 mandatory 目标
        items = [{
            "cargo": {
                "cargo_id": "C001",
                "cargo_name": "危险品运输",
                "price": 500,
                "cost_time_minutes": 120,
                "start": {"lat": 22.5, "lng": 114.0},
                "end": {"lat": 23.0, "lng": 113.5},
                "load_time": None,
            },
            "distance_km": 10.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        constraints = [{
            "type": "cargo_category_ban",
            "params": {"banned_categories": ["危险品运输"]},
            "severity": "hard",
        }]
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "C001" in cargo_ids, "mandatory_cargo 应绕过硬禁品类过滤"

    def test_non_mandatory_cargo_still_filtered(self):
        """非 mandatory_cargo 的硬禁品类应照常被过滤。"""
        state = StateTracker()
        state.open_tasks = []
        layer = HeuristicLayer()
        items = [{
            "cargo": {
                "cargo_id": "C002",
                "cargo_name": "危险品运输",
                "price": 500,
                "cost_time_minutes": 120,
                "start": {"lat": 22.5, "lng": 114.0},
                "end": {"lat": 23.0, "lng": 113.5},
                "load_time": None,
            },
            "distance_km": 10.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        constraints = [{
            "type": "cargo_category_ban",
            "params": {"banned_categories": ["危险品运输"]},
            "severity": "hard",
        }]
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "C002" not in cargo_ids, "非 mandatory 的硬禁品类应被过滤"


# ═══════════════════════════════════════════════════════════════
# Fix #4: monthly_visit 使用不同自然日
# ═══════════════════════════════════════════════════════════════
class TestMonthlyVisitDistinctDays:
    """monthly_visit 应统计不同自然日的到访。"""

    def test_same_day_multiple_visits_count_as_one(self):
        """同一天多次到达只算 1 天。"""
        state = StateTracker()
        state.open_tasks = [{
            "type": "monthly_visit",
            "target": "lat=22.5,lng=114.0",
            "min_visit_days": 2,
            "deadline_minute": 43200,
        }]
        # 同一天 (day=0, sim_min=100 和 500 都在 day 0) 的两条历史
        history = [
            {"position_after": {"lat": 22.5, "lng": 114.0},
             "result": {"simulation_progress_minutes": 100}},
            {"position_after": {"lat": 22.5, "lng": 114.0},
             "result": {"simulation_progress_minutes": 500}},
        ]
        state.update_task_progress(1000, history)
        # 只访问了 1 天，需要 2 天，不应完成
        assert len(state.open_tasks) == 1

    def test_different_days_completes(self):
        """不同天的到访达到 min_visit_days 应完成。"""
        state = StateTracker()
        state.open_tasks = [{
            "type": "monthly_visit",
            "target": "lat=22.5,lng=114.0",
            "min_visit_days": 2,
            "deadline_minute": 43200,
        }]
        history = [
            {"position_after": {"lat": 22.5, "lng": 114.0},
             "result": {"simulation_progress_minutes": 100}},   # day 0
            {"position_after": {"lat": 22.5, "lng": 114.0},
             "result": {"simulation_progress_minutes": 2000}},  # day 1
        ]
        state.update_task_progress(3000, history)
        assert len(state.open_tasks) == 0
        assert len(state.completed_tasks) == 1


# ═══════════════════════════════════════════════════════════════
# Fix #5: monthly_visit fallback 包含 min_visit_days
# ═══════════════════════════════════════════════════════════════
class TestMonthlyVisitFallback:
    """StrategicPlanner fallback 中 monthly_visit 应包含 min_visit_days。"""

    def test_fallback_includes_min_visit_days(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        planner = StrategicPlanner(api)
        constraints = [{
            "type": "monthly_visit_requirement",
            "params": {
                "target_lat": 22.5,
                "target_lng": 114.0,
                "min_visit_days": 5,
            },
        }]
        plan = planner._fallback_plan(constraints, 43200)
        visits = [t for t in plan["must_do_tasks"] if t["type"] == "monthly_visit"]
        assert len(visits) == 1
        assert visits[0]["min_visit_days"] == 5
        assert visits[0]["target"] == "lat=22.5,lng=114.0"


class TestUrgentTaskDeferral:
    """Future monthly tasks should not short-circuit cargo search in short runs."""

    def test_future_mandatory_cargo_not_urgent_before_activation(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "mandatory_cargo",
            "target": "240646",
            "activation_min": 3848,
            "deadline_minute": 1440,
            "priority": 1,
        }]

        assert state.has_urgent_tasks(900) == []

    def test_short_run_monthly_visit_not_urgent(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "monthly_visit",
            "target": "lat=23.13,lng=113.26",
            "min_visit_days": 5,
            "deadline_minute": 1440,
            "priority": 2,
        }]

        assert state.has_urgent_tasks(900) == []


# ═══════════════════════════════════════════════════════════════
# Fix #3: _try_reviewer_override 存在性验证
# ═══════════════════════════════════════════════════════════════
class TestReviewerIntegration:
    """验证 _try_reviewer_override 方法存在且签名正确。"""

    def test_method_exists(self):
        from agent.model_decision_service import ModelDecisionService
        assert hasattr(ModelDecisionService, "_try_reviewer_override")

    def test_method_returns_decision_on_exception(self):
        """reviewer 异常时应返回原决策。"""
        from agent.model_decision_service import ModelDecisionService
        api = type("FakeApi", (), {
            "model_chat_completion": lambda self, *a, **kw: 1/0,
            "get_driver_status": lambda self, *a, **kw: {},
            "query_decision_history": lambda self, *a, **kw: {},
        })()
        svc = ModelDecisionService(api)
        state = StateTracker()
        decision = {"action": "wait", "params": {"duration_minutes": 30}}
        status = {"simulation_progress_minutes": 1000}
        result = svc._try_reviewer_override(
            decision, [], state, status, [], 1000, "D1"
        )
        # reviewer 异常时应返回原决策
        assert result["action"] == "wait"


# ═══════════════════════════════════════════════════════════════
# Problem 1: mandatory_cargo 熟货预约过滤
# ═══════════════════════════════════════════════════════════════
class TestMandatoryCargoPickupFilter:
    """候选完单后赶不到 mandatory pickup 的应被过滤。"""

    def _make_state_and_constraints(self, activation_min, pickup_lat, pickup_lng):
        """构造有 mandatory_cargo 任务的 state 和对应 constraints。"""
        state = StateTracker()
        state.open_tasks = [{
            "type": "mandatory_cargo",
            "target": "MAND001",
            "deadline_minute": 43200,
            "priority": 1,
        }]
        constraints = [{
            "type": "mandatory_cargo",
            "params": {
                "cargo_id": "MAND001",
                "activation_min": activation_min,
                "pickup_lat": pickup_lat,
                "pickup_lng": pickup_lng,
            },
        }]
        return state, constraints

    def test_candidate_filtered_when_cant_reach_mandatory(self):
        """候选完单后赶不到 mandatory pickup → 被过滤。"""
        # mandatory pickup 在远处，activation 很近
        # 当前时间 current_min=1000, activation_min=1100（只剩100分钟）
        # 候选在远处，完单需要 120 分钟，然后到 mandatory pickup 还要 60 分钟
        # 总共需要 180 分钟 > 100 分钟 → 过滤
        state, constraints = self._make_state_and_constraints(
            activation_min=1100, pickup_lat=23.0, pickup_lng=114.0,
        )
        layer = HeuristicLayer()
        # 候选：距离 20km → deadhead ~20min, cost_time=120min
        # 候选终点在 (22.0, 113.0) → 到 mandatory pickup (23.0, 114.0) 很远
        items = [{
            "cargo": {
                "cargo_id": "C_FAR",
                "cargo_name": "普通货物",
                "price": 500,
                "cost_time_minutes": 120,
                "start": {"lat": 22.5, "lng": 113.5},
                "end": {"lat": 22.0, "lng": 113.0},  # 终点离 mandatory pickup 远
                "load_time": None,
            },
            "distance_km": 20.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "C_FAR" not in cargo_ids, "赶不到 mandatory pickup 的候选应被过滤"

    def test_candidate_kept_when_can_reach_mandatory(self):
        """候选完单后能赶到 mandatory pickup → 不被过滤。"""
        # mandatory pickup activation_min=5000（很远）
        # 候选完单后有充足时间赶到
        state, constraints = self._make_state_and_constraints(
            activation_min=5000, pickup_lat=23.0, pickup_lng=114.0,
        )
        layer = HeuristicLayer()
        items = [{
            "cargo": {
                "cargo_id": "C_NEAR",
                "cargo_name": "普通货物",
                "price": 500,
                "cost_time_minutes": 30,
                "start": {"lat": 22.5, "lng": 113.5},
                "end": {"lat": 22.6, "lng": 113.6},  # 终点离 mandatory pickup 近
                "load_time": None,
            },
            "distance_km": 5.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "C_NEAR" in cargo_ids, "能赶到 mandatory pickup 的候选不应被过滤"

    def test_mandatory_target_itself_not_filtered(self):
        """mandatory 目标货源本身不应被过滤。"""
        state, constraints = self._make_state_and_constraints(
            activation_min=1100, pickup_lat=23.0, pickup_lng=114.0,
        )
        layer = HeuristicLayer()
        items = [{
            "cargo": {
                "cargo_id": "MAND001",  # 就是 mandatory 目标
                "cargo_name": "普通货物",
                "price": 500,
                "cost_time_minutes": 120,
                "start": {"lat": 22.5, "lng": 113.5},
                "end": {"lat": 22.0, "lng": 113.0},
                "load_time": None,
            },
            "distance_km": 20.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "MAND001" in cargo_ids, "mandatory 目标本身不应被过滤"

    def test_no_mandatory_tasks_means_no_filter(self):
        """没有 mandatory 任务时，不应有额外过滤。"""
        state = StateTracker()
        state.open_tasks = []
        layer = HeuristicLayer()
        items = [{
            "cargo": {
                "cargo_id": "C_NORMAL",
                "cargo_name": "普通货物",
                "price": 500,
                "cost_time_minutes": 120,
                "start": {"lat": 22.5, "lng": 113.5},
                "end": {"lat": 22.0, "lng": 113.0},
                "load_time": None,
            },
            "distance_km": 20.0,
        }]
        status = {"simulation_progress_minutes": 1000, "simulation_horizon_minutes": 43200}
        result = layer.score_and_rank(items, status, state, [], top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]
        assert "C_NORMAL" in cargo_ids, "无 mandatory 任务时不应过滤普通候选"


# ═══════════════════════════════════════════════════════════════
# Problem 2: scheduled_event 保守完成判断
# ═══════════════════════════════════════════════════════════════
class TestScheduledEventConservativeCompletion:
    """scheduled_event 必须有到过 pickup/home 的历史证据才能确认完成。"""

    def _make_task(self, release_min=3000, home_deadline_min=2000,
                   pickup_lat=22.5, pickup_lng=114.0,
                   home_lat=23.0, home_lng=113.5):
        return {
            "type": "scheduled_event",
            "target": "event",
            "deadline_minute": home_deadline_min,
            "release_min": release_min,
            "home_deadline_min": home_deadline_min,
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lng,
            "home_lat": home_lat,
            "home_lng": home_lng,
        }

    def test_not_completed_when_only_wrong_location_wait(self):
        """release 后只在错误地点 wait → 不完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        # history: 只在远离 pickup 和 home 的地方 wait
        history = [
            {
                "action": {"action": "wait", "params": {"duration_minutes": 60}},
                "position_after": {"lat": 25.0, "lng": 116.0},  # 远离 pickup/home
                "result": {"simulation_progress_minutes": 3060},
            },
        ]
        state.update_task_progress(3100, history)
        assert len(state.open_tasks) == 1, "只在错误地点 wait 不应完成"

    def test_completed_when_visited_pickup_and_home_and_waited_at_home(self):
        """到过 pickup/home 并在 home 等到 release → 完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        history = [
            # 到过 pickup 附近
            {
                "action": {"action": "take_order", "params": {"cargo_id": "X"}},
                "position_after": {"lat": 22.5, "lng": 114.0},  # near pickup
                "result": {"simulation_progress_minutes": 1500},
            },
            # 到过 home 附近
            {
                "action": {"action": "take_order", "params": {"cargo_id": "Y"}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # near home
                "result": {"simulation_progress_minutes": 2500},
            },
            # 在 home 等待到 release 之后
            {
                "action": {"action": "wait", "params": {"duration_minutes": 600}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # still near home
                "result": {"simulation_progress_minutes": 3100},  # after release_min=3000
            },
        ]
        state.update_task_progress(3200, history)
        assert len(state.open_tasks) == 0, "到过 pickup/home 并在 home 等到 release 应完成"
        assert len(state.completed_tasks) == 1

    def test_not_completed_when_visited_pickup_home_but_no_wait_after_release(self):
        """到过 pickup/home 但没有 release 后的 home 等待证据 → 不完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        history = [
            {
                "action": {"action": "take_order", "params": {"cargo_id": "X"}},
                "position_after": {"lat": 22.5, "lng": 114.0},
                "result": {"simulation_progress_minutes": 1500},
            },
            {
                "action": {"action": "take_order", "params": {"cargo_id": "Y"}},
                "position_after": {"lat": 23.0, "lng": 113.5},
                "result": {"simulation_progress_minutes": 2500},  # before release
            },
        ]
        state.update_task_progress(3100, history)
        assert len(state.open_tasks) == 1, "没有 release 后 home 等待证据不应完成"

    def test_not_completed_when_visited_pickup_but_not_home(self):
        """只到过 pickup 没到过 home → 不完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        history = [
            {
                "action": {"action": "take_order", "params": {"cargo_id": "X"}},
                "position_after": {"lat": 22.5, "lng": 114.0},  # near pickup
                "result": {"simulation_progress_minutes": 1500},
            },
            {
                "action": {"action": "wait", "params": {"duration_minutes": 600}},
                "position_after": {"lat": 22.5, "lng": 114.0},  # still at pickup, not home
                "result": {"simulation_progress_minutes": 3100},
            },
        ]
        state.update_task_progress(3200, history)
        assert len(state.open_tasks) == 1, "没到过 home 不应完成"

    def test_not_completed_before_effective_deadline(self):
        """effective_deadline 前不应评估完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000, home_deadline_min=2000)]
        # current_min=1500 < release_min=3000
        state.update_task_progress(1500, [])
        assert len(state.open_tasks) == 1, "deadline 前不应完成"

    def test_position_after_at_home_after_release_counts(self):
        """position_after 在 home 附近且 sim_min >= release_min → 有效证据。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        history = [
            {
                "action": {"action": "take_order", "params": {"cargo_id": "X"}},
                "position_after": {"lat": 22.5, "lng": 114.0},  # pickup
                "result": {"simulation_progress_minutes": 1500},
            },
            {
                "action": {"action": "take_order", "params": {"cargo_id": "Y"}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # home
                "result": {"simulation_progress_minutes": 2500},
            },
            # 非 wait action 但 position_after 在 home 且 sim_min >= release
            {
                "action": {"action": "reposition", "params": {"latitude": 23.0, "longitude": 113.5}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # home
                "result": {"simulation_progress_minutes": 3050},  # after release
            },
        ]
        state.update_task_progress(3100, history)
        assert len(state.open_tasks) == 0, "position_after 在 home 且 >= release 应完成"

    def test_not_completed_when_home_before_pickup(self):
        """先到 home 后到 pickup（顺序反）→ 不完成。"""
        state = StateTracker()
        state.open_tasks = [self._make_task(release_min=3000)]
        history = [
            # 先到 home（sim_min=1200）
            {
                "action": {"action": "take_order", "params": {"cargo_id": "Y"}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # home
                "result": {"simulation_progress_minutes": 1200},
            },
            # 后到 pickup（sim_min=2000）
            {
                "action": {"action": "take_order", "params": {"cargo_id": "X"}},
                "position_after": {"lat": 22.5, "lng": 114.0},  # pickup
                "result": {"simulation_progress_minutes": 2000},
            },
            # release 后回 home
            {
                "action": {"action": "wait", "params": {"duration_minutes": 600}},
                "position_after": {"lat": 23.0, "lng": 113.5},  # home
                "result": {"simulation_progress_minutes": 3100},
            },
        ]
        state.update_task_progress(3200, history)
        assert len(state.open_tasks) == 1, "顺序反（home 先于 pickup）不应完成"

    def test_none_fields_handled_safely(self):
        """release_min / home_deadline_min 为 None 时不抛异常。"""
        state = StateTracker()
        state.open_tasks = [{
            "type": "scheduled_event",
            "target": "event",
            "deadline_minute": 2000,
            "release_min": None,       # LLM 计划可能给 null
            "home_deadline_min": None,
            "pickup_lat": 22.5,
            "pickup_lng": 114.0,
            "home_lat": 23.0,
            "home_lng": 113.5,
        }]
        # 不应抛 TypeError
        state.update_task_progress(3000, [])
        # None → 0, effective_deadline=0, 不会触发 time guard
        # 但没有历史证据，所以不应完成
        assert len(state.open_tasks) == 1


class TestTakeOrderConstraintValidation:
    """覆盖 ModelDecisionService 的硬接单校验主路径。"""

    def _make_service(self):
        api = type("FakeApi", (), {
            "model_chat_completion": lambda self, *a, **kw: {},
        })()
        return ModelDecisionService(api)

    def _make_items(self, cost_time_minutes=60):
        return [{
            "distance_km": 5.0,
            "cargo": {
                "cargo_id": "C1",
                "price": 500,
                "cost_time_minutes": cost_time_minutes,
                "start": {"lat": 22.5, "lng": 114.0},
                "end": {"lat": 22.6, "lng": 114.1},
                "load_time": ["2026-03-01 10:00:00", "2026-03-01 23:00:00"],
            },
        }]

    def test_take_order_constraint_validation_uses_shared_time_restriction(self):
        """带 time_restriction 时不应因找不到方法而退化成全局 wait。"""
        svc = self._make_service()
        state = StateTracker()
        status = {
            "simulation_progress_minutes": 700,
            "simulation_horizon_minutes": 1440,
            "current_lat": 22.5,
            "current_lng": 114.0,
        }
        constraints = [{
            "type": "time_restriction",
            "params": {
                "start_hour": 12,
                "end_hour": 13,
                "forbidden_actions": ["take_order", "reposition"],
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C1", self._make_items(cost_time_minutes=120), status, constraints, state
        )

        assert ok is False
        assert reason == "violates_time_restriction"


class TestFuturePositionCost:
    """Heuristic scoring should price the position needed after an order."""

    def test_home_deadline_prefers_order_ending_near_home(self):
        state = StateTracker()
        layer = HeuristicLayer()
        status = {
            "simulation_progress_minutes": 18 * 60,
            "simulation_horizon_minutes": 1440,
        }
        constraints = [{
            "type": "daily_home_deadline",
            "params": {
                "home_lat": 23.12,
                "home_lng": 113.28,
                "deadline_hour": 23,
            },
            "penalty_amount": 900,
            "severity": "hard",
        }]
        items = [
            {
                "distance_km": 1.0,
                "cargo": {
                    "cargo_id": "FAR_HOME",
                    "cargo_name": "general",
                    "price": 500,
                    "cost_time_minutes": 60,
                    "start": {"lat": 23.0, "lng": 113.0},
                    "end": {"lat": 22.0, "lng": 114.5},
                    "load_time": None,
                },
            },
            {
                "distance_km": 1.0,
                "cargo": {
                    "cargo_id": "NEAR_HOME",
                    "cargo_name": "general",
                    "price": 500,
                    "cost_time_minutes": 60,
                    "start": {"lat": 23.0, "lng": 113.0},
                    "end": {"lat": 23.12, "lng": 113.28},
                    "load_time": None,
                },
            },
        ]

        result = layer.score_and_rank(items, status, state, constraints, top_n=2)

        assert result[0]["cargo_id"] == "NEAR_HOME"
        by_id = {item["cargo_id"]: item for item in result}
        assert by_id["NEAR_HOME"]["future_position_cost"] < by_id["FAR_HOME"]["future_position_cost"]
