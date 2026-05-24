# Tianchi Agentic AI MyGO

本项目包含司机调度决策、RL 训练、离线仿真和月收入计算流程。下面的命令默认在 Windows PowerShell 中从项目根目录执行：

```powershell
cd D:\wyfzzz\PyCharm\MyProjects\Tianchi-Agentic-AI-MyGO
```

所有项目命令默认使用 `minimind` conda 环境。

## 1. 跑测试

```powershell
conda run -n minimind python -m pytest
```

Windows 上不要并行运行多个 `conda run -n minimind ...`，conda 临时激活文件可能会互相冲突。

## 2. RL 训练

### 自定义 PPO（原有路径）

短烟测训练：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 1
```

中等试训：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 100
```

默认 Phase 2 PPO 训练：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2
```

### MaskablePPO（推荐路径）

基于 sb3-contrib 的 MaskablePPO，支持 action masking、启发式教师预训练：

```powershell
# Phase 2: 共享模型训练
conda run -n minimind python demo/agent/train.py --phase 2 --algo maskable_ppo --episodes 10

# Phase 3: 基于共享模型的 per-driver 微调
conda run -n minimind python demo/agent/train.py --phase 3 --algo maskable_ppo --episodes 5 --base-model demo/agent/models/policy_maskable_best.zip
```

默认训练轮数来自 `demo/agent/configs/rl_config.yaml`：

```yaml
training:
  ppo_episodes: 1000
  finetune:
    base_model: "demo/agent/models/policy_maskable_best.zip"
    learning_rate_factor: 0.5
    regression_threshold: 0.95
```

训练会读取：

```text
demo/agent/configs/rl_config.yaml
```

训练产物：

```text
# 自定义 PPO
demo/agent/models/policy_best.npz
demo/agent/models/policy_final.npz

# MaskablePPO
demo/agent/models/policy_maskable_best.zip
demo/agent/models/policy_maskable_final.zip

# Per-driver 微调
demo/agent/models/{driver_id}/maskable/policy_maskable_final.zip

# 训练日志
demo/results/rl_logs/
```

### 模型导出

将 MaskablePPO 模型导出为 numpy .npz 格式（部署时不需要 sb3）：

```powershell
conda run -n minimind python -m agent.sb3_export --model demo/agent/models/policy_maskable_best.zip --output demo/agent/models/policy_maskable_best_np.npz
```

### A/B 评估

对比多种策略在相同 seed 下的表现：

```powershell
conda run -n minimind python -m agent.ab_eval --config demo/agent/configs/rl_config.yaml --seeds 42,123,456
```

输出 JSON 报告到 `demo/results/rl_logs/ab_eval_{timestamp}.json`，自动检查验收标准。

## 3. 启动完整月仿真

## 3. 启动完整月仿真

完整月仿真会自动加载当前训练出的模型（按优先级）：

```text
demo/agent/models/policy_maskable_best.zip    # MaskablePPO（优先）
demo/agent/models/policy_maskable_best_np.npz # MaskablePPO numpy 导出
demo/agent/models/policy_best.npz             # 自定义 PPO
```

从项目根目录运行：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/server/main.py
```

也可以从 `demo/server/` 目录运行：

```powershell
cd demo\server
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python main.py
```

仿真配置在：

```text
demo/server/config/config.json
```

关键字段：

```json
{
  "simulation_duration_days": 31,
  "simulation_max_steps": 20000,
  "results_dir": "results"
}
```

仿真输出：

```text
demo/results/actions_202603_D001_*.jsonl
demo/results/actions_202603_D002_*.jsonl
...
demo/results/run_summary_202603.json
demo/results/logs/
```

每次启动完整仿真时，旧的 `demo/results/` 文件会自动归档到：

```text
demo/results/history/<timestamp>/
```

这是正常行为。

## 4. 计算月收入

仿真完成后，从项目根目录运行：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/calc_monthly_income.py
```

也可以从 `demo/` 目录运行：

```powershell
cd demo
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python calc_monthly_income.py
```

收益计算输出：

```text
demo/results/monthly_income_202603.json
```

重点查看：

```text
summary.total_net_income_all_drivers
summary.failed_driver_count
summary.failed_drivers
drivers[*].income.net_income
drivers[*].income.preference_penalty
drivers[*].calculation_aborted
drivers[*].validation_error
```

如果 `failed_driver_count` 大于 0，优先看对应司机的 `validation_error`，说明动作日志有时间、位置、货源有效期或参数一致性问题。

## 5. 推荐完整流程

### MaskablePPO 流程（推荐）

```powershell
# 1. 短烟测确认环境
conda run -n minimind python demo/agent/train.py --phase 2 --algo maskable_ppo --episodes 1

# 2. 训练共享模型
conda run -n minimind python demo/agent/train.py --phase 2 --algo maskable_ppo --episodes 100

# 3. Per-driver 微调
conda run -n minimind python demo/agent/train.py --phase 3 --algo maskable_ppo --episodes 10 --base-model demo/agent/models/policy_maskable_best.zip

# 4. 导出模型（可选）
conda run -n minimind python -m agent.sb3_export --model demo/agent/models/policy_maskable_best.zip --output demo/agent/models/policy_maskable_best_np.npz

# 5. A/B 评估
conda run -n minimind python -m agent.ab_eval --seeds 42,123

# 6. 跑完整月仿真
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/server/main.py

# 7. 计算月收入
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/calc_monthly_income.py
```

### 自定义 PPO 流程

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 100
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/server/main.py
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/calc_monthly_income.py
```

## 6. 常见情况

如果看到：

```text
MaskableRLDecisionLayer loaded sb3 model from ... policy_maskable_best.zip
```

或：

```text
PolicyNetworkNumpy loaded from ... policy_best.npz
RLDecisionLayer loaded ...
```

说明仿真已经正常加载 RL 模型并开始跑。MaskablePPO 优先于自定义 PPO 被加载。

如果看到 PyTorch 的 `pynvml` warning，通常不影响训练或仿真。

如果 `conda run` 因中文输出触发 Windows 编码错误，使用本 README 中的：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output ...
```

如果完整月仿真长时间没有 step 日志，优先检查模型 API 配置、网络连通性和 `demo/results/logs/server_runtime.log`。
