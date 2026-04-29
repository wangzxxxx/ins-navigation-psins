# Current Iteration State

_Updated: 2026-03-28_

## Mainline Comparison Rule

- Mainline method comparison must use:
  - the same noisy dataset
  - the same noise strength
  - the same seed
  - and a standard KF baseline as reference
- Cross-noise comparisons are allowed only as robustness / regime studies, not as mainline winner selection.

## 1x Mainline

- Current best: `Round61_Hybrid`
- Method: `psins_method_bench/methods/markov/method_42state_gm1_round61_h_scd_state20_microtight_commit.py`
- Result: `psins_method_bench/results/R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.json`
- Role: 当前默认主线 best

## Ultra-low Regime Branch

- Current branch best: `Round62`
- Method: `psins_method_bench/methods/markov/method_42state_gm1_round62_ultralow_alpha_guard_commit.py`
- Result: `psins_method_bench/results/R62_42state_gm1_round62_ultralow_alpha_guard_commit_param_errors.json`
- Role: 特殊 shared-noise ultra-low 分支 best，不等同于 1x 主线 best

## Non-promoted Exploratory Branches

### Round63
- Topic: ultra-low SCD gating / combo
- Status: probe-only, not promoted
- Reason: 没有对 Round62 / Round61 形成 clean winner

### Round64
- Topic: mainline trust-map / covariance-schedule rebalance
- Status: probe-only, not promoted
- Best signal candidate: `r64_cov_sched_static_meas_plus`
- Why not promoted:
  - `dKg_yy` 轻微回退
  - `rx_y` 轻微回退
  - `mean` 轻微回退

## Round65 (same-dataset ICG probe)

- Topic: innovation-consistency gated Round61 (ICG)
- Constraint: strict same dataset / same noise / same seed (`D_ref_mainline`, seed=42)
- Ladder used: `KF baseline -> Markov -> Markov+SCD -> Round61 -> Round65 candidates`
- Status: probe-only, not promoted
- Why not promoted:
  - no candidate cleanly beat Round61 under protected-metric gate
  - strongest candidate (`r65_icg_feedback_priority`) only improved `dKg_zz` but regressed `dKg_xy / dKg_yy / dKa_xx / rx_y`
  - `mean` and `max` did not improve versus Round61

## Next Best Follow-up

### Preferred next move
- Keep the ICG mechanism, but split gate channels:
  - `feedback_gate` for selected-state trust feedback
  - `scd_gate` for cross-cov suppression
- Add explicit yy/Ka_xx protection floor in feedback branch
- Keep same fixed mainline dataset and deterministic batch size (3-5)

### Not preferred next move
- Do not declare Round65 winner (no clean same-dataset win over Round61)
- Do not switch back to ultra-low branch logic as mainline direction
- Do not mix wide trust-map reshaping and new ICG channel-split in one batch
