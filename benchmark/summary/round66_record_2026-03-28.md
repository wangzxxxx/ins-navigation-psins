# Round66 Record (new mechanism probe)

## A. Round 基本信息
- Round name: Round66_SCD_XXZZ_ConsistencyOnly
- Round type: `new mechanism probe`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. 本轮目标
- This round is a cleaner new-direction probe, not a Round65-B split-gate repair continuation.
- Freeze Round61 feedback body as much as possible (feedback gate fixed no-op at 1.0).
- Apply consistency-adaptive control ONLY to SCD xx/zz subpath (iter2 once-per-phase SCD target=`xxzz_pair`).

## C. Allowed knobs
- knob group 1: SCD consistency gate stats (EMA/slope/floor/warmup/power)
- knob group 2: SCD alpha base around Round61 neighborhood
- locked/no-change: Round61 feedback route, yy/Ka_xx/lever feedback protections

## D. Protected metrics and clean-win gate
- hard-protected metrics: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z
- clean-win gate vs Round61: mean<0, max<=0, dKg_xx<0 and hard-protected no regression
- formalize gate: only clean winner can be promoted

## E. Candidate design (3-5 max)
### candidate 1
- name: `r66_scd_xxzz_consis_mild_f90`
- rationale: Most conservative new-direction probe: only allow tiny SCD modulation on xx/zz while preserving Round61 feedback body unchanged.
- scd_channel: `{"target_nis": 1.0, "ema_beta": 0.1, "slope": 1.0, "gate_floor": 0.9, "warmup_static_meas": 8, "power": 1.0, "apply_floor": 0.9}`
- scd: `{"mode": "once_per_phase", "alpha": 0.999, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`

### candidate 2
- name: `r66_scd_xxzz_consis_bal_f82`
- rationale: Test whether moderate SCD-only adaptation can recover dKg_zz without leaking regressions to yy/Ka_xx-protected paths.
- scd_channel: `{"target_nis": 1.0, "ema_beta": 0.09, "slope": 1.45, "gate_floor": 0.82, "warmup_static_meas": 8, "power": 1.2, "apply_floor": 0.82}`
- scd: `{"mode": "once_per_phase", "alpha": 0.9989, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`

### candidate 3
- name: `r66_scd_xxzz_consis_slowema_f85`
- rationale: Reduce gate jitter to keep mechanism deterministic/interpretable and avoid overreacting to transient innovation fluctuations.
- scd_channel: `{"target_nis": 1.0, "ema_beta": 0.05, "slope": 1.35, "gate_floor": 0.85, "warmup_static_meas": 10, "power": 1.15, "apply_floor": 0.85}`
- scd: `{"mode": "once_per_phase", "alpha": 0.9988, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`

### candidate 4
- name: `r66_scd_xxzz_consis_push_f75`
- rationale: Probe upper bound of narrow SCD-only adaptation strength to check whether a stronger xx/zz correction can beat Round61 cleanly.
- scd_channel: `{"target_nis": 1.0, "ema_beta": 0.1, "slope": 1.8, "gate_floor": 0.75, "warmup_static_meas": 8, "power": 1.35, "apply_floor": 0.75}`
- scd: `{"mode": "once_per_phase", "alpha": 0.9987, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`

## F. Result summary
- winner: none
- result class: `no useful signal`
- one-line conclusion: Round66 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.
- strongest signal: best candidate r66_scd_xxzz_consis_mild_f90: dKg_xx Δ=0.703550, dKg_zz Δ=-0.568316, mean Δ=0.143796, max Δ=0.703550

## G. Mechanism learning and next move
- mechanism learning: Freezing Round61 feedback while adapting only xx/zz SCD keeps causality interpretable, but current settings did not yet deliver a clean no-regression same-dataset win over Round61.
- next best move: Keep feedback fully frozen; run an even tighter one-knob SCD-only alpha ladder around the best Round66 candidate (single-floor/single-power perturbation) and verify if dKg_xx/mean can improve without yy/Ka_xx/rx_y leakage.

## H. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round66_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round66_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round66_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
