# 智能货运 Agent 参赛方案

## 方案概述

设计一个**三层混合决策架构**的智能 Agent，核心理念：用规则处理确定性场景（省 Token），用启发式算法做候选排序（高效），用 LLM 处理复杂权衡和偏好理解（灵活）。

目标文件：`demo/agent/model_decision_service.py`

---

## 架构设计

```
┌─────────────────────────────────────────────┐
│              决策控制器 (DecisionController)    │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐ │
│  │ 规则层   │→│ 启发式层  │→│  LLM 层    │ │
│  │ (快速路径)│  │(候选评分) │  │(复杂权衡)  │ │
│  └─────────┘  └──────────┘  └────────────┘ │
│       │              │              │        │
│  ┌────┴──────────────┴──────────────┘       │
│  │          状态追踪器 (StateTracker)         │
│  │  - 累计收益/里程/接单数                     │
│  │  - 偏好合规监控                            │
│  │  - 时段统计                               │
│  └──────────────────────────────────────────┘
└─────────────────────────────────────────────┘
```

---

## Task 1: 状态追踪器 (StateTracker)

**目标**：维护跨步骤的决策上下文，避免每步都查询全部历史。

**核心数据结构**：
```python
class StateTracker:
    # 基础统计
    total_orders: int            # 累计接单数
    total_income: float          # 累计毛收入
    total_deadhead_km: float     # 累计空驶里程
    total_haul_km: float         # 累计干线里程
    
    # 时间管理
    last_rest_start: int         # 上次休息开始时间(分钟)
    daily_rest_minutes: dict     # 每天连续休息时长
    days_without_orders: set     # 无接单的日期集合
    
    # 偏好合规状态
    preference_violations: int    # 已违规次数
    preference_penalty_accrued: float  # 已累计罚分
    
    # 空间记忆
    hot_zones: list              # 高收益区域坐标
    recent_positions: deque      # 最近位置历史
```

**初始化**：首次调用 `decide()` 时通过 `query_decision_history(driver_id, step=-1)` 恢复状态（支持断点续跑）。

---

## Task 2: 偏好解析引擎 (PreferenceEngine)

**目标**：将文本偏好转化为可执行的约束检查器，仅在首次调用时用 LLM 解析一次。

**设计策略**：

1. **首次调用时**：用 LLM 将 `preferences` 文本解析为结构化约束
2. **后续每步**：用纯代码检查约束状态，零 Token 消耗

**约束分类**：
| 类型 | 示例 | 检查方式 |
|------|------|---------|
| 时间约束 | 每日休息>=8h、23-4点禁止活动 | 时间比对 |
| 空间约束 | 限定区域、禁入半径 | Haversine距离计算 |
| 数量约束 | 月度空驶<=100km、>=4天不接单 | 累计统计比对 |
| 窗口约束 | 特定时间段必须在指定位置 | 时间+位置联合检查 |

**LLM 解析 Prompt 设计**：
```
将以下司机偏好规则解析为JSON约束列表：
{preferences_text}

输出格式：
[{
  "type": "daily_rest|spatial_restrict|mileage_cap|...",
  "params": {...具体参数...},
  "penalty_per_violation": number,
  "penalty_cap": number
}]
```

这一步只需 1 次 LLM 调用（约几百 Token），之后全部用规则引擎执行。

---

## Task 3: 规则层 (RuleLayer) - 快速决策路径

**目标**：对确定性场景直接返回决策，不调用 LLM，节省 Token。

**快速路径场景**：

| 场景 | 判断条件 | 直接动作 |
|------|---------|---------|
| 强制休息 | 偏好要求连续休息，且未满足 | `wait(duration)` |
| 深夜无货 | 22:00-06:00 且附近无有效货源 | `wait(60~480)` |
| 月末收官 | 仿真剩余时间不足完成任何可见货源 | `wait(remaining)` |
| 禁行时段 | 偏好禁止当前时段活动 | `wait(until_allowed)` |
| 仅一个可选货源且明确高收益 | 候选=1，净收益>阈值 | `take_order(cargo_id)` |
| 窗口约束触发 | 必须在指定时间前到达指定位置 | `reposition(target)` |

**预计覆盖率**：约 40-60% 的决策步骤可由规则层直接处理。

---

## Task 4: 启发式层 (HeuristicLayer) - 候选评分与筛选

**目标**：用数学模型对货源候选打分排序，为 LLM 决策提供精简输入。

**评分公式**：
```python
def score_cargo(cargo, driver_status, state_tracker):
    # 基础收益评估
    price_yuan = cargo["price"] / 100
    deadhead_km = haversine(driver_pos, cargo["start"])
    total_km = deadhead_km + cargo["haul_distance_km"]
    cost = total_km * driver["cost_per_km"]
    net_profit = price_yuan - cost
    
    # 时间效率（元/分钟）
    total_minutes = ceil(deadhead_km / speed * 60) + cargo["cost_time_minutes"]
    efficiency = net_profit / total_minutes if total_minutes > 0 else -inf
    
    # 偏好合规性（扣分项）
    compliance_penalty = check_preference_compliance(cargo, driver_status)
    
    # 位置价值（终点是否在高收益区域）
    destination_value = estimate_destination_value(cargo["end"])
    
    # 综合评分
    score = (
        efficiency * W_EFFICIENCY +
        destination_value * W_DESTINATION -
        compliance_penalty * W_COMPLIANCE
    )
    return score
```

**筛选策略**：
- 过滤不合法候选（时间窗过期、无法在月内完成、违反硬约束）
- 按评分取 Top 5 传给 LLM（而非 baseline 的 20 个，大幅减少 Token）

---

## Task 5: LLM 层 (LLMLayer) - 复杂权衡决策

**目标**：仅在启发式层无法确定最优解时调用 LLM，处理需要"理解"的复杂情况。

**调用场景**：
- Top 候选评分接近（差距<15%），需权衡
- 存在偏好冲突（接单能赚钱但可能违规）
- 战略性空驶决策（当前无好货，是否移到远处高收益区域）
- 异常情况处理

**Prompt 优化设计**：
```
你是货运调度Agent。基于以下信息做最优决策。

[司机状态] 位置:(lat,lng) 时间:Day X HH:MM 累计收入:¥XXX 本日休息:Xh
[约束] {已解析的结构化约束摘要}
[Top候选-已按收益排序]
1. cargo_X: 净利¥XX, 耗时Xh, 效率¥X/h, 终点:XX区
2. cargo_Y: ...
3. ...
[当前风险] {偏好违规风险提示}

请选择最优动作，JSON格式输出：
{"action":"take_order|wait|reposition", ...params}
```

**关键优化**：
- 只传 Top 5 而非全部候选（减少 60%+ Token）
- 预计算净利/效率等数值（减少 LLM 计算负担）
- 用结构化摘要替代原始文本（减少上下文长度）
- 去掉冗余系统提示（精简到核心指令）

---

## Task 6: 决策控制器 (DecisionController) - 主入口

**目标**：协调三层系统，实现 `decide()` 方法。

```python
class ModelDecisionService:
    def decide(self, driver_id: str) -> dict:
        # 0. 初始化（首次调用）
        if not self._initialized:
            self._initialize(driver_id)
        
        # 1. 更新状态追踪
        status = self._api.get_driver_status(driver_id)
        self._state_tracker.update(status)
        
        # 2. 规则层 - 快速路径
        rule_action = self._rule_layer.check(status, self._state_tracker)
        if rule_action:
            self._state_tracker.record_action(rule_action)
            return rule_action
        
        # 3. 查询货源 + 启发式评分
        cargo_resp = self._api.query_cargo(driver_id, status["lat"], status["lng"])
        scored = self._heuristic.score_and_rank(cargo_resp["items"], status)
        
        # 4. 启发式层 - 明确最优解
        if scored and scored[0]["score"] > THRESHOLD_HIGH:
            if len(scored) < 2 or scored[0]["score"] > scored[1]["score"] * 1.3:
                action = {"action": "take_order", "cargo_id": scored[0]["cargo_id"]}
                self._state_tracker.record_action(action)
                return action
        
        # 5. LLM 层 - 复杂权衡
        top_candidates = scored[:5]
        action = self._llm_layer.decide(status, top_candidates, self._state_tracker)
        self._state_tracker.record_action(action)
        return action
```

---

## Task 7: Token 预算管理

**目标**：确保单司机 Token 消耗不超过 500 万（复赛约束）。

**预算分配**：
| 用途 | 预算占比 | 约Token数 |
|------|---------|----------|
| 偏好解析（一次性） | 1% | 5万 |
| LLM 决策调用 | 90% | 450万 |
| 异常/重试 | 9% | 45万 |

**估算**：
- 每次 LLM 调用约 800-1200 Token（精简 Prompt）
- 500万 / 1000 = 约 5000 次调用
- 月度 43200 分钟，平均每步约 30 分钟 = 约 1440 步
- 若 50% 由规则层处理 = 约 720 次 LLM 调用
- Token 预算充裕（即使全部用 LLM 也足够）

**降级策略**：当 Token 使用率 > 80% 时，提高规则层阈值，减少 LLM 调用频率。

---

## Task 8: 高收益区域学习

**目标**：动态识别货源密集/高价区域，指导 reposition 决策。

**实现方式**：
- 每次 `query_cargo()` 返回时，记录货源起点分布和价格
- 维护区域热力图（网格化），按时段统计历史货源密度和均价
- 当当前位置无好货源时，reposition 到最近的高热力值区域

---

## 实施路线

| 优先级 | 任务 | 预期收益 |
|--------|------|---------|
| P0 | Task 1 状态追踪器 | 基础设施，所有层依赖 |
| P0 | Task 2 偏好解析引擎 | 直接减少罚分（当前baseline罚分极高） |
| P0 | Task 3 规则层 | 节省 Token + 避免明显违规 |
| P1 | Task 4 启发式评分 | 提高选单质量，减少 LLM 输入量 |
| P1 | Task 5 LLM 层优化 | 精简 Prompt，提升决策质量 |
| P1 | Task 6 决策控制器 | 整合所有组件 |
| P2 | Task 7 Token 预算管理 | 复赛必要，初赛可简化 |
| P2 | Task 8 区域学习 | 提升 reposition 决策质量 |

---

## 关键风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 偏好文本变化导致解析失败 | 罚分大增 | LLM 解析 + 正则兜底，不硬编码 |
| LLM 返回格式异常 | 动作无法执行 | 多层解析 + 默认 wait 兜底 |
| 货源评分模型偏差 | 选单质量差 | 动态调整权重，参考历史收益反馈 |
| Token 超限 | 被终止 | 监控用量 + 自动降级到纯规则模式 |
| 仿真结束前接了无法完成的单 | income_eligible=False | 预检剩余时间 vs 运输耗时 |
