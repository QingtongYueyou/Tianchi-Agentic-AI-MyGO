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

短烟测训练，用来确认训练入口、数据读取、模型输出都能跑通：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 1
```

中等试训，适合先看收益趋势：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 100
```

默认 Phase 2 PPO 训练：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2
```

默认训练轮数来自 `demo/agent/configs/rl_config.yaml`：

```yaml
training:
  ppo_episodes: 1000
```

单司机/个性化微调：

```powershell
conda run -n minimind python demo/agent/train.py --phase 3
```

如果当前目录已经切到 `demo/`，也可以这样训练：

```powershell
conda run -n minimind python agent/train.py --phase 2
```

训练会读取：

```text
demo/agent/configs/rl_config.yaml
```

训练产物会写到：

```text
demo/agent/models/policy_best.npz
demo/agent/models/policy_final.npz
demo/results/rl_logs/latest_phase2.json
```

注意：当前训练命令不会从已有 `policy_best.npz` 断点续训，而是重新初始化策略并覆盖最新模型文件。

## 3. 启动完整月仿真

完整月仿真会自动加载当前训练出的：

```text
demo/agent/models/policy_best.npz
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

先跑短训练确认环境：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 1
```

再跑 100 轮试训：

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --episodes 100
```

用最新模型跑完整月仿真：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/server/main.py
```

计算最终月收入：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output -n minimind python demo/calc_monthly_income.py
```

## 6. 常见情况

如果看到：

```text
PolicyNetworkNumpy loaded from ... policy_best.npz
RLDecisionLayer loaded ...
simulation start marked
driver loop begin driver_id=D001
```

说明仿真已经正常加载 RL 模型并开始跑。

如果看到 PyTorch 的 `pynvml` warning，通常不影响训练或仿真。

如果 `conda run` 因中文输出触发 Windows 编码错误，使用本 README 中的：

```powershell
$env:PYTHONIOENCODING="utf-8"
conda run --no-capture-output ...
```

如果完整月仿真长时间没有 step 日志，优先检查模型 API 配置、网络连通性和 `demo/results/logs/server_runtime.log`。
