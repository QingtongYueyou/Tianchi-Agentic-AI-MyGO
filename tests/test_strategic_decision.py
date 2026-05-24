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
    ProfitSearchLayer,
    RuleLayer,
    MultiDriverCoordinationLayer,
    _get_day_index,
    haversine,
    _distance_to_minutes,
)
from agent.strategic_planner import StrategicPlanner
from server.bench.simulation_orchestrator import SimulationOrchestrator


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

    def test_expired_mandatory_cargo_is_not_urgent(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "mandatory_cargo",
            "target": "240646",
            "activation_min": 3763,
            "window_end": 3848,
            "deadline_minute": 44640,
            "priority": 1,
        }]

        state.update_task_progress(3900, [])

        assert state.has_urgent_tasks(3900) == []
        assert state.open_tasks == []
        assert state.completed_tasks[0]["expired"] is True


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

    def test_constraint_fallback_blocks_order_crossing_mandatory_activation(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        status = {
            "simulation_progress_minutes": 1000,
            "simulation_horizon_minutes": 43200,
            "current_lat": 23.0,
            "current_lng": 114.0,
        }
        constraints = [{
            "type": "mandatory_cargo",
            "params": {
                "cargo_id": "MAND001",
                "activation_min": 1100,
                "pickup_lat": 23.0,
                "pickup_lng": 114.0,
            },
        }]
        items = [{
            "distance_km": 0.0,
            "cargo": {
                "cargo_id": "C_OTHER",
                "cargo_name": "general",
                "price": 1000,
                "cost_time_minutes": 120,
                "start": {"lat": 23.0, "lng": 114.0},
                "end": {"lat": 22.0, "lng": 113.0},
                "load_time": None,
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C_OTHER", items, status, constraints, state
        )

        assert ok is False
        assert reason == "order_would_miss_mandatory_cargo"

    def test_completed_mandatory_constraint_does_not_keep_blocking_orders(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        state.completed_tasks = [{"type": "mandatory_cargo", "target": "MAND001"}]
        status = {
            "simulation_progress_minutes": 1000,
            "simulation_horizon_minutes": 43200,
            "current_lat": 23.0,
            "current_lng": 114.0,
        }
        constraints = [{
            "type": "mandatory_cargo",
            "params": {
                "cargo_id": "MAND001",
                "activation_min": 1100,
                "pickup_lat": 23.0,
                "pickup_lng": 114.0,
            },
        }]
        items = [{
            "distance_km": 0.0,
            "cargo": {
                "cargo_id": "C_OTHER",
                "cargo_name": "general",
                "price": 1000,
                "cost_time_minutes": 120,
                "start": {"lat": 23.0, "lng": 114.0},
                "end": {"lat": 22.0, "lng": 113.0},
                "load_time": None,
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C_OTHER", items, status, constraints, state
        )

        assert ok is True
        assert reason == ""


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

    def test_scheduled_event_rejects_order_that_cannot_complete_pickup_home_chain(self):
        svc = self._make_service()
        state = StateTracker()
        state.open_tasks = [{
            "type": "scheduled_event",
            "target": "event",
            "pickup_min": 13560,
            "home_deadline_min": 13680,
            "release_min": 18000,
            "pickup_lat": 23.21,
            "pickup_lng": 113.37,
            "home_lat": 23.19,
            "home_lng": 113.36,
            "pickup_wait_minutes": 10,
        }]
        status = {
            "simulation_progress_minutes": 13140,
            "simulation_horizon_minutes": 43200,
            "current_lat": 23.10,
            "current_lng": 113.20,
        }
        constraints = [{
            "type": "scheduled_event",
            "params": {
                "pickup_min": 13560,
                "home_deadline_min": 13680,
                "release_min": 18000,
                "pickup_lat": 23.21,
                "pickup_lng": 113.37,
                "home_lat": 23.19,
                "home_lng": 113.36,
                "pickup_wait_minutes": 10,
            },
        }]
        items = [{
            "distance_km": 0.0,
            "cargo": {
                "cargo_id": "C1",
                "price": 1000,
                "cost_time_minutes": 485,
                "start": {"lat": 23.10, "lng": 113.20},
                "end": {"lat": 22.83, "lng": 113.94},
                "load_time": None,
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C1", items, status, constraints, state
        )

        assert ok is False
        assert reason == "order_would_miss_scheduled_event"

    def test_scheduled_event_allows_short_order_when_chain_still_fits(self):
        svc = self._make_service()
        state = StateTracker()
        state.open_tasks = [{
            "type": "scheduled_event",
            "target": "event",
            "pickup_min": 13560,
            "home_deadline_min": 13680,
            "release_min": 18000,
        }]
        status = {
            "simulation_progress_minutes": 13140,
            "simulation_horizon_minutes": 43200,
            "current_lat": 23.10,
            "current_lng": 113.20,
        }
        constraints = [{
            "type": "scheduled_event",
            "params": {
                "pickup_min": 13560,
                "home_deadline_min": 13680,
                "release_min": 18000,
                "pickup_lat": 23.21,
                "pickup_lng": 113.37,
                "home_lat": 23.19,
                "home_lng": 113.36,
                "pickup_wait_minutes": 10,
            },
        }]
        items = [{
            "distance_km": 0.0,
            "cargo": {
                "cargo_id": "C1",
                "price": 1000,
                "cost_time_minutes": 60,
                "start": {"lat": 23.10, "lng": 113.20},
                "end": {"lat": 23.20, "lng": 113.37},
                "load_time": None,
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C1", items, status, constraints, state
        )

        assert ok is True
        assert reason == ""


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


class TestProfitSearchLayer:
    """Profit search should optimize constrained downstream profit, not only immediate true_net."""

    def test_downstream_value_can_beat_higher_immediate_true_net(self):
        state = StateTracker()
        state.spatial_income[(240, 1140)] = 12000.0
        layer = ProfitSearchLayer()
        status = {"simulation_progress_minutes": 0, "simulation_horizon_minutes": 1440}
        candidates = [
            {
                "cargo_id": "COLD",
                "true_net": 500.0,
                "score": 500.0,
                "net_profit": 500.0,
                "total_minutes": 60,
                "deadhead_km": 1.0,
                "has_soft_penalty": False,
                "end": {"lat": 23.0, "lng": 113.0},
            },
            {
                "cargo_id": "HOT",
                "true_net": 430.0,
                "score": 430.0,
                "net_profit": 430.0,
                "total_minutes": 60,
                "deadhead_km": 1.0,
                "has_soft_penalty": False,
                "end": {"lat": 24.0, "lng": 114.0},
            },
        ]

        ranked = layer.rank_candidates(candidates, status, state, [], None)
        best = layer.select_best(ranked)

        assert best is not None
        assert best["cargo_id"] == "HOT"
        assert best["downstream_value"] > 0
        assert best["profit_search_score"] > ranked[1]["profit_search_score"]

    def test_late_overnight_long_order_gets_large_risk_cost(self):
        layer = ProfitSearchLayer()
        state = StateTracker()
        status = {
            "simulation_progress_minutes": 8 * 1440 + 21 * 60,
            "simulation_horizon_minutes": 43200,
        }
        candidates = [{
            "cargo_id": "OVERNIGHT",
            "true_net": 2000.0,
            "score": 2000.0,
            "net_profit": 2000.0,
            "total_minutes": 900,
            "deadhead_km": 1.0,
            "has_soft_penalty": False,
            "end": {"lat": 23.0, "lng": 113.0},
        }]

        ranked = layer.rank_candidates(candidates, status, state, [], None)

        assert 0 < ranked[0]["overnight_risk_cost"] <= 900
        assert ranked[0]["profit_search_score"] > 0

    def test_positive_long_order_is_not_waited_away(self):
        layer = ProfitSearchLayer()
        candidates = [{
            "cargo_id": "LONG_BUT_PROFITABLE",
            "true_net": 220.0,
            "score": 220.0,
            "net_profit": 820.0,
            "total_minutes": 780,
            "deadhead_km": 12.0,
            "has_soft_penalty": False,
            "end": {"lat": 23.0, "lng": 113.0},
        }]

        best = layer.select_best(candidates)

        assert best is not None
        assert best["cargo_id"] == "LONG_BUT_PROFITABLE"

    def test_select_best_falls_back_when_top_fails_guard(self):
        layer = ProfitSearchLayer()
        candidates = [
            {
                "cargo_id": "TOO_THIN_FAR",
                "true_net": 50.0,
                "score": 50.0,
                "net_profit": 500.0,
                "total_minutes": 60,
                "deadhead_km": 220.0,
                "has_soft_penalty": False,
                "end": {"lat": 23.0, "lng": 113.0},
            },
            {
                "cargo_id": "RUNNABLE",
                "true_net": 160.0,
                "score": 160.0,
                "net_profit": 360.0,
                "total_minutes": 180,
                "deadhead_km": 15.0,
                "has_soft_penalty": False,
                "end": {"lat": 23.0, "lng": 113.0},
            },
        ]

        best = layer.select_best(candidates)

        assert best is not None
        assert best["cargo_id"] == "RUNNABLE"


# ═══════════════════════════════════════════════════════════════
# true_net 重构测试
# ═══════════════════════════════════════════════════════════════

class TestClassifyConstraintSeverity:
    """classify_constraint_severity 应正确区分硬/软约束。"""

    def test_mandatory_cargo_is_hard(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "mandatory_cargo"}) == "hard"

    def test_scheduled_event_is_hard(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "scheduled_event"}) == "hard"

    def test_time_restriction_is_hard(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "time_restriction"}) == "hard"

    def test_forbidden_circle_is_hard(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "spatial_restrict", "params": {"type": "forbidden_circle"}}
        ) == "hard"

    def test_daily_rest_is_soft(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "daily_rest", "severity": "soft"}) == "soft"

    def test_daily_home_deadline_default_is_soft(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "daily_home_deadline", "severity": "soft", "penalty_amount": 300}
        ) == "soft"

    def test_daily_home_deadline_hard_high_penalty_is_hard(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "daily_home_deadline", "severity": "hard", "penalty_amount": 9000}
        ) == "hard"

    def test_daily_home_deadline_hard_low_penalty_is_soft(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "daily_home_deadline", "severity": "hard", "penalty_amount": 300}
        ) == "soft"

    def test_monthly_visit_is_soft(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "monthly_visit_requirement"}) == "soft"

    def test_day_off_is_soft(self):
        assert HeuristicLayer.classify_constraint_severity(
            {"type": "day_off_requirement"}) == "soft"


class TestTrueNetScoring:
    """score_and_rank 应输出 true_net 字段，且净利润权重为 1.0。"""

    def test_output_has_true_net_field(self):
        state = StateTracker()
        layer = HeuristicLayer()
        status = {"simulation_progress_minutes": 600, "simulation_horizon_minutes": 43200}
        items = [{
            "distance_km": 5.0,
            "cargo": {
                "cargo_id": "C1", "cargo_name": "general", "price": 500,
                "cost_time_minutes": 60,
                "start": {"lat": 23.0, "lng": 113.0},
                "end": {"lat": 23.5, "lng": 113.5},
                "load_time": None,
            },
        }]
        result = layer.score_and_rank(items, status, state, [], top_n=1)
        assert len(result) == 1
        assert "true_net" in result[0]
        assert "has_soft_penalty" in result[0]
        assert "hard_penalty" in result[0]
        assert "long_order_penalty" in result[0]
        assert "deadhead_risk_cost" in result[0]

    def test_net_profit_weight_is_one(self):
        """净利润应以 1.0 权重进入 true_net（不再是 0.25）。"""
        state = StateTracker()
        layer = HeuristicLayer()
        status = {"simulation_progress_minutes": 600, "simulation_horizon_minutes": 43200}
        # 两个候选，唯一区别是价格
        items = [
            {
                "distance_km": 1.0,
                "cargo": {
                    "cargo_id": "LOW", "cargo_name": "general", "price": 200,
                    "cost_time_minutes": 30,
                    "start": {"lat": 23.0, "lng": 113.0},
                    "end": {"lat": 23.01, "lng": 113.01},
                    "load_time": None,
                },
            },
            {
                "distance_km": 1.0,
                "cargo": {
                    "cargo_id": "HIGH", "cargo_name": "general", "price": 600,
                    "cost_time_minutes": 30,
                    "start": {"lat": 23.0, "lng": 113.0},
                    "end": {"lat": 23.01, "lng": 113.01},
                    "load_time": None,
                },
            },
        ]
        result = layer.score_and_rank(items, status, state, [], top_n=2)
        by_id = {c["cargo_id"]: c for c in result}
        # HIGH 的 true_net 应比 LOW 高约 400（价格差），差距应远大于旧公式下的 100
        diff = by_id["HIGH"]["true_net"] - by_id["LOW"]["true_net"]
        assert diff > 300, f"true_net diff {diff} should reflect full profit difference"


class TestDailyRestSoftPenalty:
    """daily_rest 违反应不再硬拒，而是通过 soft_penalty_cost 惩罚。"""

    def test_daily_rest_no_longer_hard_rejects(self):
        """score_and_rank 不应因 daily_rest 过滤掉候选。"""
        state = StateTracker()
        layer = HeuristicLayer()
        # 当天已过 20 小时，只剩 4 小时，需要 5 小时连续休息
        status = {"simulation_progress_minutes": 20 * 60, "simulation_horizon_minutes": 43200}
        constraints = [{
            "type": "daily_rest",
            "severity": "soft",
            "params": {"min_continuous_minutes": 300},
            "penalty_amount": 300,
        }]
        items = [{
            "distance_km": 1.0,
            "cargo": {
                "cargo_id": "LATE", "cargo_name": "general", "price": 800,
                "cost_time_minutes": 120,
                "start": {"lat": 23.0, "lng": 113.0},
                "end": {"lat": 23.5, "lng": 113.5},
                "load_time": None,
            },
        }]
        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        # 以前会被过滤掉（返回空），现在应保留
        assert len(result) == 1
        assert result[0]["has_soft_penalty"] is True
        assert result[0]["penalty_score"] > 0


class TestMaxDeadheadHardFilter:
    """单笔赴装货点空驶上限应在候选阶段硬过滤。"""

    def test_max_deadhead_distance_filters_candidate(self):
        state = StateTracker()
        layer = HeuristicLayer()
        status = {"simulation_progress_minutes": 600, "simulation_horizon_minutes": 43200}
        constraints = [{
            "type": "max_deadhead_distance",
            "params": {"max_deadhead_km": 50},
            "severity": "hard",
        }]
        items = [{
            "distance_km": 60.0,
            "cargo": {
                "cargo_id": "TOO_FAR",
                "cargo_name": "general",
                "price": 2000,
                "cost_time_minutes": 60,
                "start": {"lat": 23.0, "lng": 113.0},
                "end": {"lat": 23.1, "lng": 113.1},
                "load_time": None,
            },
        }]

        result = layer.score_and_rank(items, status, state, constraints, top_n=5)

        assert result == []


class TestSelectBestByTrueNet:
    """_select_best_by_true_net 应正确应用 penalty_margin 决策规则。"""

    def test_safe_chosen_when_penalty_not_better_enough(self):
        candidates = [
            {"cargo_id": "SAFE", "score": 100, "true_net": 100,
             "has_soft_penalty": False, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 10},
            {"cargo_id": "PENALTY", "score": 250, "true_net": 250,
             "has_soft_penalty": True, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 10},
        ]
        # PENALTY true_net (250) > SAFE true_net (100) + margin (200) = 300? No (250 < 300)
        best = ModelDecisionService._select_best_by_true_net(candidates)
        assert best["cargo_id"] == "SAFE"

    def test_penalty_chosen_when_better_enough(self):
        candidates = [
            {"cargo_id": "SAFE", "score": 100, "true_net": 100,
             "has_soft_penalty": False, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 10},
            {"cargo_id": "PENALTY", "score": 400, "true_net": 400,
             "has_soft_penalty": True, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 10},
        ]
        # PENALTY true_net (400) > SAFE true_net (100) + margin (200) = 300? Yes
        best = ModelDecisionService._select_best_by_true_net(candidates)
        assert best["cargo_id"] == "PENALTY"

    def test_extreme_deadhead_rejected_when_profit_is_too_thin(self):
        candidates = [
            {"cargo_id": "FAR", "score": 120, "true_net": 120,
             "has_soft_penalty": False, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 200},
        ]
        # deadhead 200 > 150, true_net 500 < 800 → reject
        best = ModelDecisionService._select_best_by_true_net(candidates)
        assert best is None

    def test_extreme_deadhead_accepted_with_high_true_net(self):
        candidates = [
            {"cargo_id": "FAR_BUT_RICH", "score": 500, "true_net": 500,
             "has_soft_penalty": False, "hard_penalty": 0,
             "total_minutes": 60, "deadhead_km": 200},
        ]
        # deadhead 200 > 150, but true_net 900 > 800 → accept
        best = ModelDecisionService._select_best_by_true_net(candidates)
        assert best is not None
        assert best["cargo_id"] == "FAR_BUT_RICH"

    def test_empty_candidates_returns_none(self):
        assert ModelDecisionService._select_best_by_true_net([]) is None


class TestEstimateWaitValue:
    """estimate_wait_value 应返回合理的等待价值。"""

    def test_wait_value_with_rest_need(self):
        state = StateTracker()
        constraints = [{
            "type": "daily_rest",
            "severity": "soft",
            "params": {"min_continuous_minutes": 300},
            "penalty_amount": 300,
        }]
        # 尚未满足休息要求 → rest_value 应使 wait_value 为正
        val = HeuristicLayer.estimate_wait_value(
            600, state, constraints, 43200)
        assert val > 0, "需要休息时 wait_value 应为正"

    def test_wait_value_at_night(self):
        state = StateTracker()
        # 深夜 23:00，无休息需求
        val = HeuristicLayer.estimate_wait_value(
            23 * 60, state, [], 43200)
        # 应为负（夜间无货，时间成本 + 夜间惩罚）
        assert val < 0, "深夜无需求时 wait_value 应为负"


class TestDailyHomeDeadlineFixes:
    """每日回家 deadline 应使用真实距离，并在启发式层过滤不可回家订单。"""

    def _make_service(self):
        api = type("FakeApi", (), {
            "model_chat_completion": lambda self, *a, **kw: {},
        })()
        return ModelDecisionService(api)

    def test_urgent_home_uses_km_radius_not_degree_box(self):
        svc = self._make_service()
        state = StateTracker()
        task = {
            "type": "daily_home_deadline",
            "home_lat": 23.12,
            "home_lng": 113.28,
            "deadline_hour": 23,
            "priority": 2,
        }
        status = {
            "simulation_progress_minutes": 22 * 60 + 38,
            "current_lat": 23.11,
            "current_lng": 113.26,
        }
        constraints = [{
            "type": "daily_home_deadline",
            "params": {
                "home_lat": 23.12,
                "home_lng": 113.28,
                "deadline_hour": 23,
                "radius_km": 1.0,
            },
            "penalty_amount": 900,
            "severity": "hard",
        }]

        decision = svc._handle_urgent_tasks([task], status, state, constraints)

        assert decision["action"] == "reposition"

    def test_score_filters_orders_that_cannot_return_home(self):
        state = StateTracker()
        layer = HeuristicLayer()
        status = {
            "simulation_progress_minutes": 22 * 60,
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
                    "price": 1000,
                    "cost_time_minutes": 60,
                    "start": {"lat": 23.12, "lng": 113.28},
                    "end": {"lat": 22.0, "lng": 114.5},
                    "load_time": None,
                },
            },
            {
                "distance_km": 1.0,
                "cargo": {
                    "cargo_id": "NEAR_HOME",
                    "cargo_name": "general",
                    "price": 200,
                    "cost_time_minutes": 10,
                    "start": {"lat": 23.12, "lng": 113.28},
                    "end": {"lat": 23.12, "lng": 113.28},
                    "load_time": None,
                },
            },
        ]

        result = layer.score_and_rank(items, status, state, constraints, top_n=5)
        cargo_ids = [c["cargo_id"] for c in result]

        assert "FAR_HOME" not in cargo_ids
        assert "NEAR_HOME" in cargo_ids
        assert result[0]["hard_penalty"] == 0


class TestContinuousRestFixes:
    """连续休息应补足当前连续段，而不是补足分散历史总量。"""

    def test_forced_rest_continues_existing_wait_streak(self):
        state = StateTracker()
        state.record_wait(0, 360)
        layer = RuleLayer()
        constraints = [{
            "type": "daily_rest",
            "severity": "soft",
            "params": {"min_continuous_minutes": 480},
            "penalty_amount": 300,
        }]

        decision = layer._check_forced_rest(360, state, constraints, 1440)

        assert decision == {"action": "wait", "params": {"duration_minutes": 120}}

    def test_forced_rest_continues_across_midnight_without_query_gap(self):
        state = StateTracker()
        state.record_wait(2700, 180)
        layer = RuleLayer()
        constraints = [{
            "type": "daily_rest",
            "severity": "soft",
            "params": {"min_continuous_minutes": 240},
            "penalty_amount": 300,
        }]

        decision = layer._check_forced_rest(2880, state, constraints, 4320)

        assert decision == {"action": "wait", "params": {"duration_minutes": 240}}

    def test_forced_rest_starts_as_morning_block(self):
        state = StateTracker()
        layer = RuleLayer()
        constraints = [{
            "type": "daily_rest",
            "severity": "soft",
            "params": {"min_continuous_minutes": 240},
            "penalty_amount": 300,
        }]

        decision = layer._check_forced_rest(60, state, constraints, 43200)

        assert decision == {"action": "wait", "params": {"duration_minutes": 240}}


class TestRuntimeConstraintTaskSync:
    """运行中变为可见的关键约束应补进 open_tasks。"""

    def test_visible_mandatory_cargo_is_added_to_open_tasks(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        state.set_strategic_plan({"must_do_tasks": []})
        constraints = [{
            "type": "mandatory_cargo",
            "visible_end_time": "2026-03-03 16:08:24",
            "params": {
                "cargo_id": "240646",
                "activation_min": 3763,
                "pickup_lat": 24.81,
                "pickup_lng": 113.58,
            },
        }]

        svc._sync_constraint_tasks(state, constraints, 44640)

        assert len(state.open_tasks) == 1
        task = state.open_tasks[0]
        assert task["type"] == "mandatory_cargo"
        assert task["target"] == "240646"
        assert task["window_end"] > task["activation_min"]

    def test_sync_constraints_accepts_default_fields(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        state.set_strategic_plan({"must_do_tasks": []})
        constraints = [
            {
                "type": "scheduled_event",
                "params": {
                    "pickup_lat": 23.21,
                    "pickup_lng": 113.37,
                    "home_lat": 23.19,
                    "home_lng": 113.36,
                    "pickup_min": 13560,
                    "home_deadline_min": 14280,
                    "release_min": 18600,
                },
            },
            {
                "type": "daily_home_deadline",
                "params": {
                    "home_lat": 23.12,
                    "home_lng": 113.28,
                },
            },
        ]

        svc._sync_constraint_tasks(state, constraints, 43200)

        by_type = {task["type"]: task for task in state.open_tasks}
        assert by_type["scheduled_event"]["pickup_wait_minutes"] == 10
        assert by_type["daily_home_deadline"]["deadline_hour"] == 23

    def test_scheduled_event_remains_urgent_through_release(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "scheduled_event",
            "pickup_min": 100,
            "home_deadline_min": 200,
            "release_min": 300,
            "deadline_minute": 200,
        }]

        assert state.has_urgent_tasks(100)
        assert state.has_urgent_tasks(250)

    def test_scheduled_event_waits_at_pickup_before_home(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        task = {
            "type": "scheduled_event",
            "pickup_min": 100,
            "home_deadline_min": 200,
            "release_min": 300,
            "pickup_lat": 23.21,
            "pickup_lng": 113.37,
            "home_lat": 23.19,
            "home_lng": 113.36,
            "pickup_wait_minutes": 10,
            "priority": 1,
        }
        state.open_tasks = [task]

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(100),
            {
                "simulation_progress_minutes": 100,
                "current_lat": 23.21,
                "current_lng": 113.37,
            },
            state,
            [],
        )

        assert decision == {"action": "wait", "params": {"duration_minutes": 10}}
        assert task["pickup_wait_until"] == 110

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(110),
            {
                "simulation_progress_minutes": 110,
                "current_lat": 23.21,
                "current_lng": 113.37,
            },
            state,
            [],
        )

        assert decision["action"] == "reposition"
        assert decision["params"] == {"latitude": 23.19, "longitude": 113.36}
        assert task["pickup_wait_done"] is True

    def test_scheduled_event_does_not_pin_driver_before_travel_start(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        task = {
            "type": "scheduled_event",
            "pickup_min": 13560,
            "home_deadline_min": 13680,
            "release_min": 17880,
            "deadline_minute": 13680,
            "pickup_lat": 23.21,
            "pickup_lng": 113.37,
            "home_lat": 23.19,
            "home_lng": 113.36,
            "pickup_wait_minutes": 10,
            "priority": 1,
        }
        state.open_tasks = [task]

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(13140),
            {
                "simulation_progress_minutes": 13140,
                "current_lat": 23.65,
                "current_lng": 113.10,
            },
            state,
            [],
        )

        assert decision is None

    def test_scheduled_event_starts_by_travel_time_before_pickup(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        task = {
            "type": "scheduled_event",
            "pickup_min": 13560,
            "home_deadline_min": 13680,
            "release_min": 17880,
            "deadline_minute": 13680,
            "pickup_lat": 23.21,
            "pickup_lng": 113.37,
            "home_lat": 23.19,
            "home_lng": 113.36,
            "pickup_wait_minutes": 10,
            "priority": 1,
        }
        state.open_tasks = [task]

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(13490),
            {
                "simulation_progress_minutes": 13490,
                "current_lat": 23.65,
                "current_lng": 113.10,
            },
            state,
            [],
        )

        assert decision == {
            "action": "reposition",
            "params": {"latitude": 23.21, "longitude": 113.37},
        }

    def test_active_mandatory_cargo_at_pickup_allows_cargo_query(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        task = {
            "type": "mandatory_cargo",
            "target": "240646",
            "activation_min": 100,
            "pickup_lat": 24.81,
            "pickup_lng": 113.58,
            "priority": 1,
        }
        state.open_tasks = [task]

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(100),
            {
                "simulation_progress_minutes": 100,
                "current_lat": 24.81,
                "current_lng": 113.58,
            },
            state,
            [],
        )

        assert decision is None

    def test_visible_active_mandatory_cargo_is_taken_after_query(self):
        state = StateTracker()
        state.open_tasks = [{
            "type": "mandatory_cargo",
            "target": "240646",
        }]
        layer = RuleLayer()
        status = {
            "simulation_progress_minutes": 100,
            "simulation_horizon_minutes": 1000,
            "current_lat": 24.81,
            "current_lng": 113.58,
        }
        constraints = [{
            "type": "mandatory_cargo",
            "params": {
                "cargo_id": "240646",
                "activation_min": 100,
                "pickup_lat": 24.81,
                "pickup_lng": 113.58,
            },
        }]
        items = [{"cargo": {"cargo_id": "240646"}}]

        decision = layer.evaluate(status, state, constraints, items)

        assert decision == {"action": "take_order", "params": {"cargo_id": "240646"}}

    def test_pending_mandatory_cargo_waits_until_activation_at_pickup(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        state = StateTracker()
        task = {
            "type": "mandatory_cargo",
            "target": "240646",
            "activation_min": 100,
            "pickup_lat": 24.81,
            "pickup_lng": 113.58,
            "priority": 1,
        }
        state.open_tasks = [task]

        decision = svc._handle_urgent_tasks(
            state.has_urgent_tasks(90),
            {
                "simulation_progress_minutes": 90,
                "current_lat": 24.81,
                "current_lng": 113.58,
            },
            state,
            [],
        )

        assert decision == {"action": "wait", "params": {"duration_minutes": 10}}


class TestOutputPrecision:
    """动作日志坐标精度不能被压到两位小数。"""

    def test_float_output_keeps_six_decimals(self):
        orch = object.__new__(SimulationOrchestrator)

        out = orch._normalize_for_output({"lat": 23.27797, "lng": 114.073838})

        assert out == {"lat": 23.27797, "lng": 114.073838}


class TestRepositionSafety:
    """Long required reposition actions should progress in safe legs."""

    def test_long_reposition_is_split_instead_of_dropped(self):
        api = type("FakeApi", (), {"model_chat_completion": lambda self, *a, **kw: {}})()
        svc = ModelDecisionService(api)
        status = {
            "simulation_progress_minutes": 0,
            "simulation_horizon_minutes": 10000,
            "current_lat": 23.73,
            "current_lng": 116.16,
        }
        decision = {
            "action": "reposition",
            "params": {"latitude": 23.13, "longitude": 113.26},
        }

        validated = svc._validate_action(decision, status, [])

        assert validated is not None
        params = validated["params"]
        leg_km = haversine(
            status["current_lat"],
            status["current_lng"],
            params["latitude"],
            params["longitude"],
        )
        assert leg_km <= 300.0
        assert params["latitude"] != 23.13 or params["longitude"] != 113.26


class TestMonthlyVisitValidationRelaxed:
    """月度到访在非收官阶段不应每天硬拒订单。"""

    def test_monthly_visit_not_hard_rejected_early_in_month(self):
        api = type("FakeApi", (), {
            "model_chat_completion": lambda self, *a, **kw: {},
        })()
        svc = ModelDecisionService(api)
        state = StateTracker()
        state.open_tasks = [{
            "type": "monthly_visit",
            "target": "lat=23.13,lng=113.26",
            "min_visit_days": 5,
            "deadline_minute": 43200,
        }]
        status = {
            "simulation_progress_minutes": 600,
            "simulation_horizon_minutes": 43200,
            "current_lat": 23.12,
            "current_lng": 113.28,
        }
        constraints = [{
            "type": "monthly_visit_requirement",
            "params": {
                "target_lat": 23.13,
                "target_lng": 113.26,
                "min_visit_days": 5,
            },
        }]
        items = [{
            "distance_km": 1.0,
            "cargo": {
                "cargo_id": "C1",
                "price": 1000,
                "cost_time_minutes": 1000,
                "start": {"lat": 23.12, "lng": 113.28},
                "end": {"lat": 22.0, "lng": 114.5},
                "load_time": None,
            },
        }]

        ok, reason = svc._validate_take_order_constraints(
            "C1", items, status, constraints, state)

        assert ok is True
        assert reason == ""


class TestCoordinationLayerDefault:
    """离线独立司机评测下，默认不跨司机过滤货源。"""

    def test_competition_filter_is_disabled_by_default(self):
        layer = MultiDriverCoordinationLayer()
        layer.record_decision("D001", "C1", "take_order")
        items = [
            {"cargo": {"cargo_id": "C1"}},
            {"cargo": {"cargo_id": "C2"}},
        ]

        result = layer.filter_competitive_cargo("D002", items)

        assert result == items


class TestShortRunFastDecision:
    """Short simulations should not call the model for obvious heuristic decisions."""

    class NoModelApi:
        def __init__(self, item):
            self.model_calls = 0
            self.items = item if isinstance(item, list) else [item]
            self.status = {
                "driver_id": "D_TEST",
                "current_lat": 23.0,
                "current_lng": 113.0,
                "preferences": [],
                "simulation_progress_minutes": 0,
                "simulation_horizon_minutes": 1440,
            }

        def model_chat_completion(self, payload):
            self.model_calls += 1
            raise AssertionError("model should not be called")

        def get_driver_status(self, driver_id):
            return dict(self.status)

        def query_decision_history(self, driver_id, step):
            return {"records": []}

        def query_cargo(self, driver_id, latitude, longitude):
            return {"items": self.items}

    @staticmethod
    def _item(price, cargo_id="C_FAST", end_lat=23.01, end_lng=113.01):
        return {
            "distance_km": 1.0,
            "cargo": {
                "cargo_id": cargo_id,
                "cargo_name": "general",
                "price": price,
                "cost_time_minutes": 30,
                "start": {"lat": 23.0, "lng": 113.0},
                "end": {"lat": end_lat, "lng": end_lng},
                "load_time": None,
            },
        }

    def test_positive_true_net_uses_heuristic_take_order(self):
        api = self.NoModelApi(self._item(price=1000))
        svc = ModelDecisionService(api)
        svc._preference_engine.get_constraints = lambda driver_id, prefs: []

        decision = svc.decide("D_TEST")

        assert decision == {"action": "take_order", "params": {"cargo_id": "C_FAST"}}
        assert api.model_calls == 0

    def test_profit_search_prefers_downstream_hot_area_without_model(self):
        cold = self._item(price=1100, cargo_id="C_COLD", end_lat=23.0, end_lng=113.0)
        hot = self._item(price=980, cargo_id="C_HOT", end_lat=23.1, end_lng=113.1)
        api = self.NoModelApi([cold, hot])
        svc = ModelDecisionService(api)
        svc._preference_engine.get_constraints = lambda driver_id, prefs: []
        state = StateTracker()
        state.spatial_income[(231, 1131)] = 12000.0
        state.refresh_from_history = lambda api_obj, driver_id: None
        svc._state_tracker["D_TEST"] = state

        decision = svc.decide("D_TEST")

        assert decision == {"action": "take_order", "params": {"cargo_id": "C_HOT"}}
        assert api.model_calls == 0

    def test_non_positive_true_net_waits_without_model(self):
        item = self._item(price=1)
        item["distance_km"] = 200.0
        item["cargo"]["end"] = {"lat": 25.0, "lng": 115.0}
        api = self.NoModelApi(item)
        svc = ModelDecisionService(api)
        svc._preference_engine.get_constraints = lambda driver_id, prefs: []

        decision = svc.decide("D_TEST")

        assert decision["action"] == "wait"
        assert api.model_calls == 0
