"""Regression tests for partial-horizon income evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"
sys.path.insert(0, str(DEMO_DIR))

from calc_monthly_income import (  # noqa: E402
    DriverD009PreferenceCalculator,
    DriverD010PreferenceCalculator,
    PreferenceRuleSpec,
)


def _rule(
    content: str,
    start_minutes: int = 0,
    end_minutes: int = 0,
    penalty_amount: float = 0.0,
    penalty_cap: float | None = None,
) -> PreferenceRuleSpec:
    return PreferenceRuleSpec(
        content=content,
        start_minutes=start_minutes,
        end_minutes=end_minutes,
        penalty_amount=penalty_amount,
        penalty_cap=penalty_cap,
    )


def test_d009_future_familiar_cargo_is_deferred_in_short_run():
    rules = [
        _rule("familiar cargo", start_minutes=2_000, end_minutes=3_000, penalty_amount=10_000, penalty_cap=10_000),
        _rule("home", penalty_amount=300, penalty_cap=9_000),
        _rule("express", penalty_amount=200, penalty_cap=2_000),
    ]

    penalty, details = DriverD009PreferenceCalculator().compute([], {}, rules, simulation_duration_days=0)

    assert penalty == 0.0
    assert details["rules"][0]["penalty"] == 0.0
    assert details["rules"][0]["deferred_until_window_end"] is True


def test_d010_future_family_event_and_monthly_visit_are_deferred_in_short_run():
    rules = [
        _rule("family event", start_minutes=12_960, end_minutes=18_720, penalty_amount=9_000, penalty_cap=9_000),
        _rule("monthly visit", penalty_amount=3_000, penalty_cap=3_000),
        _rule("daily rest", penalty_amount=300, penalty_cap=9_000),
        _rule("soft cargo", penalty_amount=100, penalty_cap=1_000),
    ]

    _penalty, details = DriverD010PreferenceCalculator().compute([], {}, rules, simulation_duration_days=1)

    family_rule = details["rules"][0]
    visit_rule = details["rules"][1]
    assert family_rule["penalty"] == 0.0
    assert family_rule["deferred_until_window_end"] is True
    assert visit_rule["penalty"] == 0.0
    assert visit_rule["deferred_until_full_month"] is True
