# Round65-B Record (repair branch)

## A. Round 基本信息
- Round name: Round65B_DualGate_Repair
- Round type: `repair branch`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (same as Round65, fixed noisy dataset)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. 本轮目标
- Keep innovation-consistency mechanism, repair Round65 protected regressions via dual-channel split gate.
- Primary repair targets: `dKg_yy / dKa_xx / rx_y` (same-dataset vs Round61).
- Secondary target: keep SCD adaptation mainly on `xx/zz` path, avoid re-opening broad trust-map search.

## C. Allowed knobs
- knob group 1: dual-channel innovation gate config (feedback channel vs SCD channel with separate EMA/slope/floor).
- knob group 2: narrow yy/Ka_xx local feedback guard and optional micro rx_y post-guard.
- knob group 3: SCD target scope (`xxzz_pair` vs bounded `scale_block nav-only`).

## D. Protected metrics
- must hold: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z
- repair-first protected set: dKg_yy / dKa_xx / rx_y
- absolutely cannot regress for clean win: any hard-protected metric vs Round61 > 0

## E. Candidate design
### candidate 1
- name: `r65b_split_xxzz_fb92_guard`
- changed knobs: feedback_channel=`{"target_nis": 1.0, "ema_beta": 0.04, "slope": 1.1, "gate_floor": 0.92, "warmup_static_meas": 8, "power": 1.0, "apply_floor": 0.92}`, scd_channel=`{"target_nis": 1.0, "ema_beta": 0.12, "slope": 1.9, "gate_floor": 0.6, "warmup_static_meas": 8, "power": 1.2, "apply_floor": 0.6}`, scd=`{"mode": "once_per_phase", "alpha": 0.9988, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`
- rationale: First repair move: decouple channels and keep feedback close to Round61 on protected paths; reserve adaptation mainly for xx/zz SCD suppression.
- expected benefit: 修复 Round65 的 yy/Ka_xx/rx_y 回退，同时保留 innovation-consistency 机制与 SCD 自适应能力。
- possible risk: 修复有效但主目标 mean/max/dKg_xx 不够，仍无法通过 clean-win gate。

### candidate 2
- name: `r65b_split_xxzz_fb96_frozen`
- changed knobs: feedback_channel=`{"target_nis": 1.0, "ema_beta": 0.03, "slope": 1.0, "gate_floor": 0.96, "warmup_static_meas": 8, "power": 1.0, "apply_floor": 0.96}`, scd_channel=`{"target_nis": 1.0, "ema_beta": 0.12, "slope": 2.1, "gate_floor": 0.55, "warmup_static_meas": 8, "power": 1.35, "apply_floor": 0.55}`, scd=`{"mode": "once_per_phase", "alpha": 0.9987, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`
- rationale: Stress-test whether most regressions came from feedback-path gate drift; keep feedback near baseline and move adaptation burden to xx/zz SCD only.
- expected benefit: 修复 Round65 的 yy/Ka_xx/rx_y 回退，同时保留 innovation-consistency 机制与 SCD 自适应能力。
- possible risk: 修复有效但主目标 mean/max/dKg_xx 不够，仍无法通过 clean-win gate。

### candidate 3
- name: `r65b_split_xxzz_fb94_rxguard`
- changed knobs: feedback_channel=`{"target_nis": 1.0, "ema_beta": 0.04, "slope": 1.2, "gate_floor": 0.94, "warmup_static_meas": 8, "power": 1.0, "apply_floor": 0.94}`, scd_channel=`{"target_nis": 1.0, "ema_beta": 0.11, "slope": 1.95, "gate_floor": 0.58, "warmup_static_meas": 8, "power": 1.25, "apply_floor": 0.58}`, scd=`{"mode": "once_per_phase", "alpha": 0.9988, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`
- rationale: Keep split-gate body unchanged and test whether a minimal lever post-guard can repair rx_y without broad retuning.
- expected benefit: 修复 Round65 的 yy/Ka_xx/rx_y 回退，同时保留 innovation-consistency 机制与 SCD 自适应能力。
- possible risk: 修复有效但主目标 mean/max/dKg_xx 不够，仍无法通过 clean-win gate。

### candidate 4
- name: `r65b_split_scale_navonly_guarded`
- changed knobs: feedback_channel=`{"target_nis": 1.0, "ema_beta": 0.04, "slope": 1.18, "gate_floor": 0.94, "warmup_static_meas": 8, "power": 1.0, "apply_floor": 0.94}`, scd_channel=`{"target_nis": 1.0, "ema_beta": 0.11, "slope": 1.7, "gate_floor": 0.62, "warmup_static_meas": 8, "power": 1.2, "apply_floor": 0.62}`, scd=`{"mode": "once_per_phase", "alpha": 0.999, "transition_duration": 2.0, "target": "scale_block", "bias_to_target": false, "apply_policy_names": ["iter2_commit"]}`
- rationale: Ablation-style check: if xx/zz-only SCD is too narrow, test a bounded scale-block/nav-only version without reopening bias coupling.
- expected benefit: 修复 Round65 的 yy/Ka_xx/rx_y 回退，同时保留 innovation-consistency 机制与 SCD 自适应能力。
- possible risk: 修复有效但主目标 mean/max/dKg_xx 不够，仍无法通过 clean-win gate。

## F. Scoring / gate
- clean win gate: same-dataset vs Round61 满足 mean<0, max<=0, dKg_xx<0，且 hard-protected 无回退，并且 dKg_yy/dKa_xx/rx_y 全部 <0。
- partial signal: 关键修复项（yy/Ka_xx/rx_y）至少一项改善，但 clean gate 未满足。
- no useful signal: 修复项无改善或 protected 回退明显。
- formalize gate: 仅 clean win 才 formalize 方法文件和正式 param_errors。

## G. Result summary
- winner: none
- result class: `no useful signal`
- one-line conclusion: Round65-B did not produce a useful repair signal or a clean promotable winner over Round61.

## H. Metric deltas vs base (Round61)
- strongest signal: best (still regressive) candidate r65b_split_xxzz_fb96_frozen: dKg_yy Δ=2.139571, dKa_xx Δ=1.264940, rx_y Δ=0.230824, dKg_xx Δ=0.695172, mean Δ=0.143454
- key regressions: [{'metric': 'dKg_xy', 'delta': 1.2980849932881071}, {'metric': 'dKg_yy', 'delta': 2.139571446677728}, {'metric': 'dKa_xx', 'delta': 1.2649403213828965}, {'metric': 'rx_y', 'delta': 0.23082418061353493}]

## I. Mechanism learning
- what worked: Dual-channel split gate can decouple feedback and SCD adaptation, and provides direct per-channel diagnostics (feedback gate vs SCD gate logs).
- what did not work enough: Even with split channels, finding a configuration that simultaneously repairs yy/Ka_xx/rx_y and improves mainline mean/max/dKg_xx over Round61 remains difficult.
- structural or redistribution: Current evidence should be treated as repair-signal exploration; only clean no-regression improvement qualifies as structural mainline gain.

## J. Next experiment generation
- keep: Dual-channel innovation-consistency mechanism and per-channel logs.
- remove: Shared scalar gate coupling that simultaneously drifts feedback and SCD response.
- next best repair direction: Lock feedback gate even closer to Round61 on iter2 protected states (yy/Ka_xx), and run a tighter xx/zz-only SCD alpha ladder (one notch around best candidate) without changing any other knob.
- next best new-mechanism direction: If repair saturates, try dual-target innovation channels with different target_nis for feedback vs SCD.
- should formalize now? no

## K. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round65b_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round65b_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round65b_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
