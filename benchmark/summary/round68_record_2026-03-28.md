# Round68 Record (global family-balanced calibration probe)

## A. Round 基本信息
- Round name: Round68_Global_Family_Balanced
- Round type: `new mechanism probe`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65/66/67)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. Chosen mechanism / global objective framing
- mechanism: `global family-balanced grouped reconciliation on top of Round61 backbone`
- framing: calibration quality is a global multi-family objective, not a local patch objective.
- objective priority: global mean / median / max + family-dispersion reduction.
- protected diagnostics (guard only, not sole objective): dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z.

## C. Clean-win gate
- mean<0, median<=0, max<=0, dKg_xx<0
- family_dispersion_delta_vs_round61 <= 0
- protected_peak_regression <= 0.15 pct-point
- only clean winner can be formalized

## D. Candidate batch (deterministic, narrow)
### candidate 1
- name: `r68_balanced_iso_mild`
- rationale: 最小改动验证：先看统一 family 归一化目标是否能在不破坏主干的前提下提升全局误差形态。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.92, "mult_max": 1.1, "weak_consensus_blend": 0.22, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.9}}`

### candidate 2
- name: `r68_balanced_strong_anchor`
- rationale: 把全局目标优先放在强可观参数家族，避免“只修弱参数外观”造成的伪全局改进。
- family_balance_cfg: `{"gamma": 0.18, "blend": 0.78, "mult_min": 0.9, "mult_max": 1.12, "weak_consensus_blend": 0.35, "weights": {"kg_diag": 1.15, "kg_offdiag": 1.15, "ka_diag": 1.1, "ka_offdiag": 1.1, "gyro_bias": 1.05, "acc_bias": 1.05, "ka2": 0.75, "lever": 0.7}}`

### candidate 3
- name: `r68_balanced_weak_consensus`
- rationale: 把 weak families（Ka2/lever）与强家族一致性绑定，测试“全局协调 + 冻结弱扰动”是否更稳。
- family_balance_cfg: `{"gamma": 0.2, "blend": 0.8, "mult_min": 0.88, "mult_max": 1.14, "weak_consensus_blend": 0.55, "weights": {"kg_diag": 1.1, "kg_offdiag": 1.05, "ka_diag": 1.1, "ka_offdiag": 1.05, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.85, "lever": 0.8}}`

### candidate 4
- name: `r68_balanced_tightcap`
- rationale: 以更紧的乘子上限约束全局重分配，验证“窄幅但系统化”的 family 归一是否更可靠。
- family_balance_cfg: `{"gamma": 0.16, "blend": 0.68, "mult_min": 0.95, "mult_max": 1.06, "weak_consensus_blend": 0.28, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 0.95, "acc_bias": 0.95, "ka2": 0.85, "lever": 0.85}}`

## E. Result summary
- winner: none
- result class: `partial signal`
- one-line conclusion: Round68 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.
- strongest signal: best candidate r68_balanced_iso_mild: mean Δ=0.462893, median Δ=1.178165, max Δ=-0.374012, dKg_xx Δ=-0.374012, family_dispersion Δ=-0.325153, protected_peak Δ=2.791209

## F. Mechanism learning and next move
- mechanism learning: Round68 keeps Round61 as a stable estimation backbone and moves adaptation to a post-estimation global family-balance reconciliation layer, which is globally motivated and interpretable by family-level RMS multipliers.
- next best move: Fix the best Round68 candidate and run a one-knob narrow sweep on weak_consensus_blend or multiplier cap only, to see whether protected_peak regression can be pushed below the clean gate without losing global mean/median/max.

## G. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round68_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round68_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round68_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
