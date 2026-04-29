# Round67 Record (new mechanism probe)

## A. Round 基本信息
- Round name: Round67_OBS_GROUPED_SCHEDULE
- Round type: `new mechanism probe`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65 / Round66)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. 本轮目标
- Preserve Round61 backbone (same trust-feedback + iter2 once-per-phase SCD body).
- New mechanism axis: observability-aware static measurement scheduling + grouped conservative feedback fusion.
- Avoid consistency-gating variants and avoid multi-knob mixed redesign.

## C. Allowed knobs
- knob group 1: static measurement reinforcement schedule by observability ratio (threshold/factor/window).
- knob group 2: grouped feedback multipliers (dominant xx/xy/zz vs protected yy/Ka_xx, with lever post-guard).
- locked/no-change: Round61 dataset, Round61 base feedback route, Round61 SCD core path.

## D. Protected metrics and clean-win gate
- hard-protected metrics: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z
- clean-win gate vs Round61: mean<0, max<=0, dKg_xx<0 and hard-protected no regression
- formalize gate: only clean winner can be promoted

## E. Candidate design (3-5 max)
### candidate 1
- name: `r67_obs_sched_reinf_mild`
- rationale: Isolate one new knob: reinforce extra static measurement only when dominant-scale observability exceeds protected-group observability.
- obs_schedule: `{"reinforce_enabled": true, "ratio_threshold": 1.02, "factor_base": 0.1, "factor_gain": 0.18, "factor_min": 0.08, "factor_max": 0.2, "static_start": 12, "static_end": 2200, "r_scale": 1.0}`
- grouped_feedback: `{"enabled": false, "ratio_ref": 1.0, "dom_gain": 0.0, "dom_max": 1.0, "xy_guard_gain": 0.0, "xy_min": 1.0, "prot_gain": 0.0, "prot_min": 1.0, "lever_guard": 1.0, "kg_yy_guard": 1.0, "ka_xx_guard": 1.0}`

### candidate 2
- name: `r67_obs_sched_reinf_bal_grouplite`
- rationale: Add a small grouped-feedback layer to protect yy/Ka_xx while keeping xx/zz reinforcement active.
- obs_schedule: `{"reinforce_enabled": true, "ratio_threshold": 1.0, "factor_base": 0.12, "factor_gain": 0.22, "factor_min": 0.1, "factor_max": 0.24, "static_start": 10, "static_end": 2400, "r_scale": 1.0}`
- grouped_feedback: `{"enabled": true, "ratio_ref": 1.0, "dom_gain": 0.15, "dom_max": 1.06, "xy_guard_gain": 0.1, "xy_min": 0.9, "prot_gain": 0.22, "prot_min": 0.88, "lever_guard": 0.85, "kg_yy_guard": 0.92, "ka_xx_guard": 0.92}`

### candidate 3
- name: `r67_grouped_conservative_guard`
- rationale: Paper-friendly grouped-update baseline: keep measurement schedule unchanged and make feedback fusion conservative for protected groups.
- obs_schedule: `{"reinforce_enabled": false, "ratio_threshold": 9.99, "factor_base": 0.0, "factor_gain": 0.0, "factor_min": 0.0, "factor_max": 0.0, "static_start": 999999, "static_end": 0, "r_scale": 1.0}`
- grouped_feedback: `{"enabled": true, "ratio_ref": 1.0, "dom_gain": 0.18, "dom_max": 1.05, "xy_guard_gain": 0.2, "xy_min": 0.86, "prot_gain": 0.34, "prot_min": 0.78, "lever_guard": 0.6, "kg_yy_guard": 0.7, "ka_xx_guard": 0.72}`

### candidate 4
- name: `r67_async_static_window_guard`
- rationale: Test grouped/asynchronous update scheduling by enabling reinforcement only after static evidence stabilizes.
- obs_schedule: `{"reinforce_enabled": true, "ratio_threshold": 1.05, "factor_base": 0.08, "factor_gain": 0.16, "factor_min": 0.06, "factor_max": 0.18, "static_start": 40, "static_end": 2600, "r_scale": 1.05}`
- grouped_feedback: `{"enabled": true, "ratio_ref": 1.02, "dom_gain": 0.12, "dom_max": 1.04, "xy_guard_gain": 0.16, "xy_min": 0.88, "prot_gain": 0.28, "prot_min": 0.82, "lever_guard": 0.7, "kg_yy_guard": 0.8, "ka_xx_guard": 0.8}`

## F. Result summary
- winner: none
- result class: `no useful signal`
- one-line conclusion: Round67 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.
- strongest signal: best candidate r67_obs_sched_reinf_mild: dKg_xx Δ=-7.535691, dKg_zz Δ=3.879025, mean Δ=2.423254, max Δ=12.726855

## G. Mechanism learning and next move
- mechanism learning: Round67 keeps Round61 unchanged at backbone level and injects only a classical observability-aware static schedule plus grouped conservative fusion; current batch shows whether this axis can improve xx/xy/zz while preserving yy/Ka_xx/lever.
- next best move: If no clean winner, keep the best Round67 candidate as seed and run a one-knob refinement: fix grouped-feedback map and sweep only static reinforcement factor window (base/max) to test clean-gate feasibility.

## H. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round67_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round67_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round67_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
