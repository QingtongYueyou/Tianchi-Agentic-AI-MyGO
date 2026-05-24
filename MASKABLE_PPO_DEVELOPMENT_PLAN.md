# Maskable PPO + Heuristic Fallback Development Plan

## Goal

Build an aggressive revenue-oriented RL stack:

```text
heuristic candidate generation
-> revenue/risk action mask
-> MaskablePPO as primary decision maker
-> heuristic veto/fallback
-> per-driver fine-tuning
```

The target is not just to swap PPO implementations. The target is higher monthly net income with stronger control over invalid, low-profit, high-deadhead, and penalty-prone decisions.

## Phase 0: Freeze Baseline

Establish a measurable baseline before changing strategy behavior.

- Run the current test suite with:
  `conda run -n minimind python -m pytest`
- Run the current benchmark and record:
  - total income
  - net income
  - penalties
  - deadhead mileage
  - accepted orders
  - wait count and wait duration
  - per-driver income
- Save the baseline under `demo/results/rl_logs/`.

Success criteria:

- A baseline report exists.
- Later experiments can compare against the same seeds and metrics.

## Phase 1: MaskablePPO Main Training Path

Add a first-class `sb3-contrib` MaskablePPO training path while keeping the current trainer available for comparison.

Planned changes:

- Add `sb3-contrib>=2.1.0` to `demo/agent/requirements.txt`.
- Add `LogisticsDriverEnv.action_masks()` as the official MaskablePPO-compatible mask method.
- Add `demo/agent/sb3_maskable_trainer.py`.
- Add `--algo maskable_ppo` to `demo/agent/train.py`.
- Save SB3 models as:
  - `policy_maskable_final.zip`
  - `policy_maskable_best.zip`
- Write training summaries through the existing summary writer.

Smoke command:

```powershell
conda run -n minimind python demo/agent/train.py --phase 2 --algo maskable_ppo --episodes 1
```

Success criteria:

- The command completes.
- A MaskablePPO model is saved.
- Existing PPO training still works.

## Phase 2: Expand Action Space

Increase the decision surface from the current 9 actions to a richer high-revenue action set.

Target actions:

```text
0    wait_15
1    wait_30
2    wait_60
3-12 cargo_0 ... cargo_9
13   reposition_home
14   reposition_hotzone
15   reposition_supply_zone
16   reposition_high_value_dest
17   force_rest
```

Planned changes:

- Increase `_TOP_K` from 5 to 10.
- Increase `_ACTION_DIM` from 9 to 18.
- Increase state dimension from 49 to 79.
- Update `encode_state()` and `_encode_state()` to encode Top10 cargo candidates.
- Implement the new wait, reposition, and rest actions in `step()`.

Success criteria:

- Masks open only existing cargo actions.
- Reposition actions are masked when no target exists.
- Rest and wait actions update rest state correctly.
- Existing tests are updated for the new dimensions.

## Phase 3: Revenue/Risk Action Mask

Upgrade action masks from "legal action" masks to "profitable and safe action" masks.

Planned module:

- `demo/agent/action_masking.py`

Hard mask rules:

- Mask orders with negative estimated true net income.
- Mask long low-margin orders.
- Mask extreme deadhead orders unless destination value justifies them.
- Mask orders whose penalty risk is not covered by profit.
- Mask ordinary cargo when hard constraints are near deadline.
- Mask repeated waiting after too many consecutive waits.
- Always leave at least one safe action available.

Success criteria:

- `info["action_mask_reasons"]` explains blocked actions.
- No all-false mask is possible.
- Tests cover negative profit, long orders, deadhead, penalty risk, and repeated waits.

## Phase 4: Heuristic Teacher

Use the heuristic stack as a teacher, not merely as a fallback.

Planned module:

- `demo/agent/heuristic_teacher.py`

Training flow:

```text
collect teacher states/actions
-> supervised warm start
-> MaskablePPO training
-> evaluation
```

Success criteria:

- Teacher action data can be generated from the same environment.
- Warm-start policy reaches useful teacher top-1 accuracy.
- PPO starts from a stronger policy than random exploration.

## Phase 5: Revenue-Oriented Reward

Reshape reward around monthly net income and opportunity cost.

Target reward components:

```text
actual_net_income
+ time_efficiency_bonus
+ destination_future_value_bonus
+ constraint_completion_bonus
- deadhead_budget_penalty
- penalty_risk_penalty
- bad_wait_penalty
+ terminal_monthly_net_income
```

Success criteria:

- `info["reward_breakdown"]` records reward components.
- Reward correlates with net income in evaluation.
- Reward shaping does not make negative-income orders attractive.

## Phase 6: RL-First Online Decision Layer (DONE)

Make MaskablePPO the primary decision maker in online inference.

Implemented module:

- `demo/agent/maskable_rl_integration.py`

Decision flow:

```text
build TopK candidates
-> encode state
-> compute action mask
-> MaskablePPO predicts ranked actions
-> veto unsafe actions
-> execute first accepted action
-> fallback to heuristic if needed
```

Logged fields:

- `rl_action`
- `rl_prob`
- `mask_reasons`
- `veto_reason`
- `fallback_used`

Success criteria:

- RL can drive the default path.
- Any exception falls back to heuristic.
- Veto catches negative-profit, high-penalty, and runaway-wait actions.

## Phase 7: Model Export (DONE)

Make the MaskablePPO model deployable without requiring SB3 at inference time.

Implemented modules:

- `demo/agent/sb3_export.py`
- `demo/agent/maskable_policy_numpy.py`

Success criteria:

- Exported numpy policy matches SB3 top-1 action on sampled states.
- Inference can run with mask-aware numpy policy.
- Missing model falls back to heuristic.

## Phase 8: Per-Driver Fine-Tuning (DONE)

Fine-tune policy variants by driver profile.

Implemented changes:

- `demo/agent/sb3_maskable_trainer.py`: added `base_model_path` to `MaskablePPOConfig`; `train()` loads pre-trained model when set
- `demo/agent/train.py`: added `--base-model` CLI arg; phase 3 passes base model to per-driver configs; regression check after each driver
- `demo/agent/configs/rl_config.yaml`: added `training.finetune` section with `base_model`, `learning_rate_factor`, `regression_threshold`

Model naming:

- `policy_maskable_best.zip` -- Phase 2 shared base model
- `policy_maskable_{driver_id}.zip` -- Phase 8 per-driver fine-tuned models

Regression check: fine-tuned model income must be >= base income * 0.95, otherwise discarded.

## Phase 9: A/B Evaluation (DONE)

Compare all major strategy versions on identical seeds.

Implemented module: `demo/agent/ab_eval.py`

Strategy groups:

- `heuristic` -- current heuristic-only strategy
- `maskable_integrated` -- MaskablePPO with RL-first fallback
- `maskable_per_driver` -- per-driver fine-tuned MaskablePPO

CLI: `python -m agent.ab_eval --config demo/agent/configs/rl_config.yaml --seeds 42,123,456`

Acceptance criteria checked automatically:

```text
average net income >= baseline * 1.08
worst seed net income >= baseline * 0.98
penalty <= baseline * 1.10
fallback rate < 35%
```

Output: JSON report to `demo/results/rl_logs/ab_eval_{timestamp}.json` + console summary table.

## Integration: model_decision_service.py (DONE)

- `MaskableRLDecisionLayer` wired as primary decision path
- `_try_load_maskable_rl_layer()` auto-discovers models in `demo/agent/models/`
- MaskablePPO `decide()` called before existing RL reranking; falls through to heuristic on `None`

## Current Implementation Focus

All phases (0-9) are complete. Next steps:

- Train shared MaskablePPO model (Phase 2): `python demo/agent/train.py --phase 2 --algo maskable_ppo`
- Fine-tune per driver (Phase 3): `python demo/agent/train.py --phase 3 --algo maskable_ppo --base-model demo/agent/models/policy_maskable_best.zip`
- Run A/B evaluation: `python -m agent.ab_eval --seeds 42,123,456`
- Export best model for deployment: `python -m agent.sb3_export --model demo/agent/models/policy_maskable_best.zip --output demo/agent/models/policy_maskable_best_np.npz`
