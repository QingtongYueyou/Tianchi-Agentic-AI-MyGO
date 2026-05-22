"""haversine 距离计算函数单元测试。

同时测试两处实现的正确性与 bit-exact 一致性：
- demo/simkit/simulation_actions.py :: haversine_km
- demo/agent/model_decision_service.py :: haversine
"""

import sys
import os
import math
import importlib.util

import pytest

# 将项目根目录和 demo 目录加入 sys.path，确保导入可用
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "demo"))


def _load_function_from_file(file_path: str, function_name: str):
    """从指定文件直接加载函数，避免触发模块级别的完整导入链。"""
    spec = importlib.util.spec_from_file_location("_isolated_module", file_path)
    module = importlib.util.module_from_spec(spec)
    # 注入 math 模块，因为目标函数依赖它
    module.__builtins__ = __builtins__
    # 仅执行文件中纯函数所需的最小依赖
    import types
    fake_module = types.ModuleType("simkit")
    fake_cargo = types.ModuleType("simkit.cargo_repository")
    fake_driver = types.ModuleType("simkit.driver_state_manager")
    fake_cargo.CargoRepository = type("CargoRepository", (), {})
    fake_driver.DriverStateManager = type("DriverStateManager", (), {})
    sys.modules.setdefault("simkit", fake_module)
    sys.modules.setdefault("simkit.cargo_repository", fake_cargo)
    sys.modules.setdefault("simkit.driver_state_manager", fake_driver)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


# 直接从源文件加载 haversine 函数，避免 numpy 等重型依赖
_simkit_path = os.path.join(PROJECT_ROOT, "demo", "simkit", "simulation_actions.py")
_agent_path = os.path.join(PROJECT_ROOT, "demo", "agent", "model_decision_service.py")

haversine_km = _load_function_from_file(_simkit_path, "haversine_km")

# agent 模块可能也有依赖，用同样方式加载
def _load_agent_haversine():
    """加载 agent 中的 haversine，需要 mock 其依赖。"""
    agent_dir = os.path.join(PROJECT_ROOT, "demo", "agent")
    spec = importlib.util.spec_from_file_location(
        "_agent_module",
        _agent_path,
        submodule_search_locations=[agent_dir],
    )
    module = importlib.util.module_from_spec(spec)
    module.__builtins__ = __builtins__
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        # 如果有未满足的依赖，仅提取 haversine 纯函数源码执行
        import re
        with open(_agent_path, "r") as f:
            source = f.read()
        # 提取 haversine 函数定义
        pattern = r"(def haversine\(.*?\n(?:    .*\n)*)"
        match = re.search(pattern, source)
        if match:
            ns = {"math": math}
            exec(match.group(0), ns)
            return ns["haversine"]
        raise
    return module.haversine


haversine = _load_agent_haversine()


# ============================================================
# 辅助常量
# ============================================================
EARTH_CIRCUMFERENCE_HALF_KM = math.pi * 6371.0  # 约 20015.09 km


# ============================================================
# 一致性测试：所有用例同时验证两个函数结果 bit-exact 相等
# ============================================================
class TestConsistency:
    """两处 haversine 实现对所有测试点必须返回完全相同的浮点值。"""

    @pytest.mark.parametrize(
        "lat1, lng1, lat2, lng2",
        [
            # 同一点
            (22.5431, 114.0579, 22.5431, 114.0579),
            # 对跖点 (0,0) -> (0,180)
            (0.0, 0.0, 0.0, 180.0),
            # 短距离（深圳两个相邻点，约几百米）
            (22.5431, 114.0579, 22.5435, 114.0585),
            # 跨越日期变更线
            (0.0, 179.0, 0.0, -179.0),
            # 跨越赤道
            (-1.0, 36.8, 1.0, 36.8),
            # 北极到南极
            (90.0, 0.0, -90.0, 0.0),
            # 深圳到广州
            (22.5431, 114.0579, 23.1291, 113.2644),
            # 北京到上海
            (39.9042, 116.4074, 31.2304, 121.4737),
            # 额外：伦敦到纽约
            (51.5074, -0.1278, 40.7128, -74.0060),
        ],
        ids=[
            "same_point",
            "antipodal_0_180",
            "short_distance",
            "cross_dateline",
            "cross_equator",
            "pole_to_pole",
            "shenzhen_guangzhou",
            "beijing_shanghai",
            "london_newyork",
        ],
    )
    def test_bit_exact_equal(self, lat1, lng1, lat2, lng2):
        result_simkit = haversine_km(lat1, lng1, lat2, lng2)
        result_agent = haversine(lat1, lng1, lat2, lng2)
        assert result_simkit == result_agent, (
            f"不一致！simkit={result_simkit}, agent={result_agent}"
        )


# ============================================================
# 边界情况测试
# ============================================================
class TestBoundary:
    """边界与特殊情况测试。"""

    def test_same_point_returns_zero(self):
        """同一点距离应为 0.0"""
        assert haversine_km(22.5431, 114.0579, 22.5431, 114.0579) == 0.0
        assert haversine(22.5431, 114.0579, 22.5431, 114.0579) == 0.0

    def test_antipodal_point(self):
        """对跖点 (0,0) -> (0,180) 应约等于半个地球周长"""
        dist = haversine_km(0.0, 0.0, 0.0, 180.0)
        assert abs(dist - EARTH_CIRCUMFERENCE_HALF_KM) < 0.01

    def test_short_distance(self):
        """相距几百米的两点，结果应在合理范围（0.01~2 km）"""
        dist = haversine_km(22.5431, 114.0579, 22.5435, 114.0585)
        assert 0.01 < dist < 2.0

    def test_cross_dateline(self):
        """跨越日期变更线 (0, 179) -> (0, -179) 应约 222 km"""
        dist = haversine_km(0.0, 179.0, 0.0, -179.0)
        # 经度差实际为 2 度，赤道上 1 度约 111.19 km
        expected = 2.0 * math.pi * 6371.0 * (2.0 / 360.0)
        assert abs(dist - expected) < 0.5

    def test_cross_equator(self):
        """跨越赤道 (-1, 36.8) -> (1, 36.8) 应约 222 km"""
        dist = haversine_km(-1.0, 36.8, 1.0, 36.8)
        expected = 2.0 * math.pi * 6371.0 * (2.0 / 360.0)
        assert abs(dist - expected) < 0.5

    def test_pole_to_pole(self):
        """北极到南极应约等于半个地球周长"""
        dist = haversine_km(90.0, 0.0, -90.0, 0.0)
        assert abs(dist - EARTH_CIRCUMFERENCE_HALF_KM) < 0.01


# ============================================================
# 已知距离验证
# ============================================================
class TestKnownDistances:
    """验证已知城市对之间的距离（容差 5%）。"""

    def test_shenzhen_to_guangzhou(self):
        """深圳到广州约 104 km"""
        dist = haversine_km(22.5431, 114.0579, 23.1291, 113.2644)
        assert 90.0 < dist < 120.0, f"深圳->广州: {dist:.2f} km"

    def test_beijing_to_shanghai(self):
        """北京到上海约 1067 km"""
        dist = haversine_km(39.9042, 116.4074, 31.2304, 121.4737)
        assert 1000.0 < dist < 1150.0, f"北京->上海: {dist:.2f} km"


# ============================================================
# 数学属性测试
# ============================================================
class TestMathProperties:
    """验证 haversine 函数满足距离度量的数学属性。"""

    POINTS = [
        (22.5431, 114.0579),   # 深圳
        (23.1291, 113.2644),   # 广州
        (39.9042, 116.4074),   # 北京
        (31.2304, 121.4737),   # 上海
        (0.0, 0.0),           # 原点
        (90.0, 0.0),          # 北极
        (-33.8688, 151.2093),  # 悉尼
    ]

    @pytest.mark.parametrize("idx_a", range(7))
    @pytest.mark.parametrize("idx_b", range(7))
    def test_symmetry(self, idx_a, idx_b):
        """对称性: d(A,B) == d(B,A)"""
        a = self.POINTS[idx_a]
        b = self.POINTS[idx_b]
        d_ab = haversine_km(a[0], a[1], b[0], b[1])
        d_ba = haversine_km(b[0], b[1], a[0], a[1])
        assert d_ab == d_ba

    @pytest.mark.parametrize("idx_a", range(7))
    @pytest.mark.parametrize("idx_b", range(7))
    def test_non_negative(self, idx_a, idx_b):
        """非负性: d(A,B) >= 0"""
        a = self.POINTS[idx_a]
        b = self.POINTS[idx_b]
        assert haversine_km(a[0], a[1], b[0], b[1]) >= 0.0

    def test_triangle_inequality(self):
        """三角不等式: d(A,C) <= d(A,B) + d(B,C)"""
        for i in range(len(self.POINTS)):
            for j in range(len(self.POINTS)):
                for k in range(len(self.POINTS)):
                    a, b, c = self.POINTS[i], self.POINTS[j], self.POINTS[k]
                    d_ac = haversine_km(a[0], a[1], c[0], c[1])
                    d_ab = haversine_km(a[0], a[1], b[0], b[1])
                    d_bc = haversine_km(b[0], b[1], c[0], c[1])
                    assert d_ac <= d_ab + d_bc + 1e-9, (
                        f"三角不等式违反: d({i},{k})={d_ac:.4f} > "
                        f"d({i},{j})={d_ab:.4f} + d({j},{k})={d_bc:.4f}"
                    )
