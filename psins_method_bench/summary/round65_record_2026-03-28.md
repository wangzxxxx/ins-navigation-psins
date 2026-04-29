# Round65 Record (filled)

## A. Round 基本信息
- Round name: Round65_Mainline_ICG
- Round type: `new mechanism probe`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (round53/61-family mainline fixed noisy dataset)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`
- Seed: `42`

## B. 本轮目标
- Chosen innovation direction: `innovation-consistency gated Round61`（用 NIS/innovation consistency gate 调制 internalized feedback 与 SCD 强度）
- Primary goal: 在 Round61 上验证 innovation-consistency gating 是否能在同数据口径下带来 clean no-regression 增益。
- Secondary goal: 通过 feedback-path vs SCD-path 门控分离，形成可解释消融证据。
- This round is NOT trying to do: 不做 ultra-low branch；不做宽范围 trust-map 搜索。

## C. Allowed knobs
- knob group 1: innovation gate statistics (NIS target/EMA/slope/floor/warmup)
- knob group 2: gate coupling map (feedback_gate_power/floor, scd_gate_power/floor)

## D. Protected metrics
- must hold: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z
- can tolerate tiny regression: dKg_zz / median (仅在 mean+max+dKg_xx 同时改善时考虑)
- absolutely cannot regress: 与 Round61 同数据对比出现明显保护项回退

## E. Candidate design
### candidate 1
- name: `r65_icg_balanced`
- changed knobs: innovation_gate = `{"target_nis": 1.0, "ema_beta": 0.08, "slope": 1.4, "gate_floor": 0.72, "warmup_static_meas": 8, "feedback_gate_power": 1.0, "feedback_gate_floor": 0.72, "scd_gate_power": 1.0, "scd_gate_floor": 0.76}`
- rationale: Core mechanism test: consistency-derived gate controls both internalized feedback aggressiveness and SCD suppression depth.
- expected benefit: 在 innovation 不一致时收敛 feedback/SCD 作用强度，降低局部好看但保护项回退。
- possible risk: 过门控导致主目标修复不够（mean/max/dKg_xx 不提升）。

### candidate 2
- name: `r65_icg_feedback_priority`
- changed knobs: innovation_gate = `{"target_nis": 1.0, "ema_beta": 0.08, "slope": 1.65, "gate_floor": 0.68, "warmup_static_meas": 8, "feedback_gate_power": 1.25, "feedback_gate_floor": 0.65, "scd_gate_power": 0.7, "scd_gate_floor": 0.84}`
- rationale: Ablation-style emphasis: test whether over-correction is mainly from feedback route rather than SCD route.
- expected benefit: 在 innovation 不一致时收敛 feedback/SCD 作用强度，降低局部好看但保护项回退。
- possible risk: 过门控导致主目标修复不够（mean/max/dKg_xx 不提升）。

### candidate 3
- name: `r65_icg_scd_priority`
- changed knobs: innovation_gate = `{"target_nis": 1.0, "ema_beta": 0.08, "slope": 1.65, "gate_floor": 0.68, "warmup_static_meas": 8, "feedback_gate_power": 0.72, "feedback_gate_floor": 0.84, "scd_gate_power": 1.3, "scd_gate_floor": 0.64}`
- rationale: Counter-ablation: test whether mis-match risk is dominated by cross-cov suppression timing/strength.
- expected benefit: 在 innovation 不一致时收敛 feedback/SCD 作用强度，降低局部好看但保护项回退。
- possible risk: 过门控导致主目标修复不够（mean/max/dKg_xx 不提升）。

### candidate 4
- name: `r65_icg_slow_ema_guarded`
- changed knobs: innovation_gate = `{"target_nis": 1.0, "ema_beta": 0.05, "slope": 1.55, "gate_floor": 0.74, "warmup_static_meas": 10, "feedback_gate_power": 1.0, "feedback_gate_floor": 0.74, "scd_gate_power": 1.0, "scd_gate_floor": 0.78}`
- rationale: Robustness-style variant: reduce gate jitter and see whether smoother consistency estimate helps protected metrics.
- expected benefit: 在 innovation 不一致时收敛 feedback/SCD 作用强度，降低局部好看但保护项回退。
- possible risk: 过门控导致主目标修复不够（mean/max/dKg_xx 不提升）。

## F. Scoring / gate
- clean win gate: 同数据下对 Round61 满足 mean<0, max<=0, dKg_xx<0 且无硬保护项回退。
- partial signal definition: 至少一个目标项改善，但 clean gate 未满足。
- no useful signal definition: 目标项无稳定改善或保护项回退。
- formalize gate: 仅 clean win 才 formalize 方法文件与 param json。

## G. Result summary
- winner: none
- result class: `no useful signal`
- one-line conclusion: Round65 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.

## H. Metric deltas vs base (Round61)
- key improves: best partial signal on r65_icg_feedback_priority: dKg_xx Δ=0.231524, dKg_zz Δ=-0.076755, mean Δ=0.031875, max Δ=0.231524
- key regressions: [{'metric': 'dKg_xy', 'delta': 0.1868742610232328}, {'metric': 'dKg_yy', 'delta': 0.4765395200662894}, {'metric': 'dKa_xx', 'delta': 0.20065147543582285}, {'metric': 'rx_y', 'delta': 0.0197149002117456}]

## I. Mechanism learning
- what probably worked: Innovation-driven dynamic gate is inspectable and does produce coherent movement on target stats in some variants.
- what probably did not work: Coupling one scalar gate to both feedback and SCD can still over-transfer suppression and leak into protected metrics.
- is this gain structural or just redistribution? Current evidence is mechanism-level partial signal, not yet a structural clean improvement over Round61.

## J. Next experiment generation
- keep: NIS-EMA consistency gate as named mechanism and logging interface (innovation_gate_log).
- remove: Overly aggressive shared gate settings that jointly drag feedback and SCD below safe protected thresholds.
- next best repair direction: Keep innovation-consistency gating mechanism but repair protected regressions via split gate map: freeze feedback gate floor on yy/Ka_xx path while keeping SCD gate adaptive on xx/zz path.
- next best new-mechanism direction: Try dual-channel consistency gate (feedback-channel and SCD-channel with separate target deviations).
- should formalize now? no

## K. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round65_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round65_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round65_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
