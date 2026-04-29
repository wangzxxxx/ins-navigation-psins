# Round65-C Record (split-point search)

## A. Round 基本信息
- Round name: Round65C_SplitPoint_Search
- Round type: `repair-axis search`
- Base candidate: `r61_s20_08988_ryz00116`
- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65 / Round65-B)
- D_ref_mainline definition:
  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`
  - arw = `0.005 * dpsh`
  - vrw = `5.0 * ugpsHz`
  - bi_g = `0.002 * dph`
  - bi_a = `5.0 * ug`
  - tau_g = tau_a = `300.0`
  - seed = `42`

## B. 本轮目标
- 不再散着试 dual-channel patch；把“分离点”压缩成一条可解释的 1D 轴。
- 搜索轴定义：`gap = feedback_floor - scd_floor`。
- 约束：平均 floor 固定为 `0.755`，其余 dual-channel skeleton 固定，避免混入额外自由度。

## C. Fixed skeleton
- feedback base cfg: `{"target_nis": 1.0, "ema_beta": 0.03, "slope": 1.0, "warmup_static_meas": 8, "power": 1.0}`
- scd gate base cfg: `{"target_nis": 1.0, "ema_beta": 0.12, "slope": 2.1, "warmup_static_meas": 8, "power": 1.35}`
- scd cfg: `{"mode": "once_per_phase", "alpha": 0.9987, "transition_duration": 2.0, "target": "xxzz_pair", "bias_to_target": true, "apply_policy_names": ["iter2_commit"]}`
- iter patch: `{"1": {"state_alpha_mult": {"16": 1.014, "21": 1.012}}}`

## D. Candidate design
### candidate 1
- name: `r65c_split_gap_m24`
- gap: `-0.24`
- feedback floor: `0.635`
- scd floor: `0.875`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 2
- name: `r65c_split_gap_m18`
- gap: `-0.18`
- feedback floor: `0.665`
- scd floor: `0.845`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 3
- name: `r65c_split_gap_m12`
- gap: `-0.12`
- feedback floor: `0.695`
- scd floor: `0.815`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 4
- name: `r65c_split_gap_m06`
- gap: `-0.06`
- feedback floor: `0.725`
- scd floor: `0.785`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 5
- name: `r65c_split_gap_p00`
- gap: `+0.00`
- feedback floor: `0.755`
- scd floor: `0.755`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 6
- name: `r65c_split_gap_p06`
- gap: `+0.06`
- feedback floor: `0.785`
- scd floor: `0.725`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 7
- name: `r65c_split_gap_p12`
- gap: `+0.12`
- feedback floor: `0.815`
- scd floor: `0.695`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 8
- name: `r65c_split_gap_p18`
- gap: `+0.18`
- feedback floor: `0.845`
- scd floor: `0.665`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 9
- name: `r65c_split_gap_p24`
- gap: `+0.24`
- feedback floor: `0.875`
- scd floor: `0.635`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 10
- name: `r65c_split_gap_p32`
- gap: `+0.32`
- feedback floor: `0.915`
- scd floor: `0.595`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

### candidate 11
- name: `r65c_split_gap_p40`
- gap: `+0.40`
- feedback floor: `0.955`
- scd floor: `0.555`
- rationale: Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, so the optimum really answers where the split point should sit on this mechanism axis.

## E. Clean-win gate
- same-dataset vs Round61: mean<0, max<=0, dKg_xx<0
- hard-protected metrics must not regress: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z
- repair-first protected set: dKg_yy / dKa_xx / rx_y

## F. Result summary
- result class: `no useful signal`
- winner: none
- conclusion: Round65-C found the best split point in this family, but it still does not cleanly beat Round61.
- strongest signal: best split candidate r65c_split_gap_p40: gap=+0.40, fb=0.955, scd=0.555; dKg_xy Δ=1.297958, dKg_yy Δ=2.143841, dKa_xx Δ=1.265498, rx_y Δ=0.231398, dKg_xx Δ=0.697953, dKg_zz Δ=-0.534175, mean Δ=0.143415, max Δ=0.697953

## G. Interpretation
- 在固定 dual-channel skeleton 与固定平均 floor 的前提下，当前最优 split point 落在 **偏 feedback 一侧（feedback 更放、SCD 更收）**；也就是说，这条轴上最好的结果对应 gap=+0.40，而不是盲目把 feedback 和 SCD 拉得越开越好。
- next move: Lock the best gap `+0.40` as center, then run one ultra-narrow local refinement on ±0.03 / ±0.06 only, or conclude that the split-point axis itself has saturated if even the best point remains clearly behind Round61.

## H. Artifacts
- candidate_json: `/root/.openclaw/workspace/psins_method_bench/results/round65c_splitpoint_candidates.json`
- summary_json: `/root/.openclaw/workspace/psins_method_bench/results/round65c_splitpoint_probe_summary.json`
- report_md: `/root/.openclaw/workspace/reports/psins_round65c_splitpoint_probe_2026-03-29.md`
- round_record_md: `/root/.openclaw/workspace/psins_method_bench/summary/round65c_splitpoint_record_2026-03-29.md`
