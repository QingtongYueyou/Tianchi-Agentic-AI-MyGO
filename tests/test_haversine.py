"""Unit tests for the two haversine implementations."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_DIR))

from agent.model_decision_service import haversine
from simkit.simulation_actions import haversine_km


EARTH_CIRCUMFERENCE_HALF_KM = math.pi * 6371.0


class TestConsistency:
    @pytest.mark.parametrize(
        "lat1, lng1, lat2, lng2",
        [
            (22.5431, 114.0579, 22.5431, 114.0579),
            (0.0, 0.0, 0.0, 180.0),
            (22.5431, 114.0579, 22.5435, 114.0585),
            (0.0, 179.0, 0.0, -179.0),
            (-1.0, 36.8, 1.0, 36.8),
            (90.0, 0.0, -90.0, 0.0),
            (22.5431, 114.0579, 23.1291, 113.2644),
            (39.9042, 116.4074, 31.2304, 121.4737),
            (51.5074, -0.1278, 40.7128, -74.0060),
        ],
    )
    def test_bit_exact_equal(self, lat1, lng1, lat2, lng2):
        assert haversine_km(lat1, lng1, lat2, lng2) == haversine(lat1, lng1, lat2, lng2)


class TestBoundary:
    def test_same_point_returns_zero(self):
        assert haversine_km(22.5431, 114.0579, 22.5431, 114.0579) == 0.0
        assert haversine(22.5431, 114.0579, 22.5431, 114.0579) == 0.0

    def test_antipodal_point(self):
        dist = haversine_km(0.0, 0.0, 0.0, 180.0)
        assert abs(dist - EARTH_CIRCUMFERENCE_HALF_KM) < 0.01

    def test_short_distance(self):
        dist = haversine_km(22.5431, 114.0579, 22.5435, 114.0585)
        assert 0.01 < dist < 2.0

    def test_cross_dateline(self):
        dist = haversine_km(0.0, 179.0, 0.0, -179.0)
        expected = 2.0 * math.pi * 6371.0 * (2.0 / 360.0)
        assert abs(dist - expected) < 0.5

    def test_cross_equator(self):
        dist = haversine_km(-1.0, 36.8, 1.0, 36.8)
        expected = 2.0 * math.pi * 6371.0 * (2.0 / 360.0)
        assert abs(dist - expected) < 0.5

    def test_pole_to_pole(self):
        dist = haversine_km(90.0, 0.0, -90.0, 0.0)
        assert abs(dist - EARTH_CIRCUMFERENCE_HALF_KM) < 0.01


class TestKnownDistances:
    def test_shenzhen_to_guangzhou(self):
        dist = haversine_km(22.5431, 114.0579, 23.1291, 113.2644)
        assert 90.0 < dist < 120.0

    def test_beijing_to_shanghai(self):
        dist = haversine_km(39.9042, 116.4074, 31.2304, 121.4737)
        assert 1000.0 < dist < 1150.0


class TestMathProperties:
    POINTS = [
        (22.5431, 114.0579),
        (23.1291, 113.2644),
        (39.9042, 116.4074),
        (31.2304, 121.4737),
        (0.0, 0.0),
        (90.0, 0.0),
        (-33.8688, 151.2093),
    ]

    @pytest.mark.parametrize("idx_a", range(7))
    @pytest.mark.parametrize("idx_b", range(7))
    def test_symmetry(self, idx_a, idx_b):
        a = self.POINTS[idx_a]
        b = self.POINTS[idx_b]
        assert haversine_km(a[0], a[1], b[0], b[1]) == haversine_km(b[0], b[1], a[0], a[1])

    @pytest.mark.parametrize("idx_a", range(7))
    @pytest.mark.parametrize("idx_b", range(7))
    def test_non_negative(self, idx_a, idx_b):
        a = self.POINTS[idx_a]
        b = self.POINTS[idx_b]
        assert haversine_km(a[0], a[1], b[0], b[1]) >= 0.0

    def test_triangle_inequality(self):
        for i, a in enumerate(self.POINTS):
            for j, b in enumerate(self.POINTS):
                for k, c in enumerate(self.POINTS):
                    d_ac = haversine_km(a[0], a[1], c[0], c[1])
                    d_ab = haversine_km(a[0], a[1], b[0], b[1])
                    d_bc = haversine_km(b[0], b[1], c[0], c[1])
                    assert d_ac <= d_ab + d_bc + 1e-9, (
                        f"triangle inequality failed for {i}, {j}, {k}"
                    )
