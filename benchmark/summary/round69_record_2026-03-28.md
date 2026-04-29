# Round69 Record (global family-balanced continuation narrow sweep)

## A. Round 基本信息
- Round name: Round69_Global_Family_Balanced
- Round type: `repair branch`
- Base candidate (Round61 anchor): `r61_s20_08988_ryz00116`
- Base candidate (Round68 center): `r68_balanced_iso_mild`
- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65/66/67/68)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. Chosen mechanism / sweep knobs
- mechanism: `global family-balanced grouped reconciliation continuation on Round61 backbone`
- objective: preserve Round68 global structure signal while reducing over-pull on dKg_xy / dKg_zz / rx_y / ry_z.
- allowed sweep knobs (narrow only):
  - `weak_consensus_blend`
  - multiplier envelope (`mult_min`, `mult_max`)
  - one tiny family weight trim (optional, single candidate only)

## C. Clean-win gate
- same-dataset vs Round61: mean<0, median<=0, max<=0, dKg_xx<0
- family_dispersion_delta_vs_round61 <= 0
- protected_peak_regression <= 0.15 pct-point
- only clean winner can be formalized

## D. Candidate batch (deterministic, narrow)
### candidate 1
- name: `r69_iso_ref_anchor`
- rationale: 以 Round68 best (r68_balanced_iso_mild) 作为中心锚点，确保 Round69 是延续而非换轴。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.92, "mult_max": 1.1, "weak_consensus_blend": 0.22, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.9}}`

### candidate 2
- name: `r69_wcb012_relax`
- rationale: 只降 weak_consensus_blend，测试是否能减轻 dKg_xy / rx_y / ry_z 过拉。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.92, "mult_max": 1.1, "weak_consensus_blend": 0.12, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.9}}`

### candidate 3
- name: `r69_wcb006_relax_more`
- rationale: 继续单旋钮降低 weak_consensus_blend，观察过拉指标是否继续回落。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.92, "mult_max": 1.1, "weak_consensus_blend": 0.06, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.9}}`

### candidate 4
- name: `r69_cap108_wcb012`
- rationale: 同步小幅收紧 multiplier envelope，抑制重分配幅度并配合弱一致性放松。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.94, "mult_max": 1.08, "weak_consensus_blend": 0.12, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.0, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.9}}`

### candidate 5
- name: `r69_cap106_wcb010_trim`
- rationale: 在窄 cap 基础上仅做微小 family 权重修剪，尝试缓和 dKg_xy 与 lever 过拉。
- family_balance_cfg: `{"gamma": 0.14, "blend": 0.72, "mult_min": 0.95, "mult_max": 1.06, "weak_consensus_blend": 0.1, "weights": {"kg_diag": 1.0, "kg_offdiag": 1.05, "ka_diag": 1.0, "ka_offdiag": 1.0, "gyro_bias": 1.0, "acc_bias": 1.0, "ka2": 0.9, "lever": 0.95}}`

## E. Result summary
- winner: none
- result class: `partial signal`
- one-line conclusion: Round69 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.
- strongest signal: best candidate r69_iso_ref_anchor: vsR61 max Δ=-0.374012, dKg_xx Δ=-0.374012, dKg_yy Δ=-0.538377, dKa_xx Δ=-1.019925, family_dispersion Δ=-0.325153, protected_peak Δ=2.791209; vsR68best repair Δ(dKg_xy/dKg_zz/rx_y/ry_z)=(0.000000, 0.000000, 0.000000, 0.000000)

## F. Mechanism learning and next move
- mechanism learning: Round69 confirms the Round68 global-family reconciliation axis can be narrowed deterministically by weak_consensus_blend and cap envelope controls; the trade-off remains global-shape retention vs targeted over-pull repair.
- next best move: If no clean winner, keep the best Round69 candidate and run one more ultra-narrow 1D sweep on a single knob (prefer weak_consensus_blend only) with no extra weight trim, to isolate causality on dKg_xy/rx_y/ry_z.

## G. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round69_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round69_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round69_probe_2026-03-28.md`
- formal_method_file: `None`
- formal_result_json: `None`
