# Round65 Plan (pre-run)

## A. Round 基本信息

- Round name: `Round65_Mainline_ICG`
- Round type: `new mechanism probe`
- Base candidate: `r61_s20_08988_ryz00116` (Round61 mainline best)
- Dataset / regime: `D_ref_mainline_1x`（固定 noisy dataset，round53/61 family）
- Seed: `42`

## B. 本轮目标

- Primary goal: 在 **同数据/同噪声/同 seed** 下验证“innovation-consistency gate”是否可在 Round61 上形成 clean no-regression 改进。
- Secondary goal: 做 feedback-path 与 SCD-path 的门控耦合消融（同一机制内）。
- This round is NOT trying to do:
  - 不做 ultra-low/special-regime 分支
  - 不做大范围 trust-map 重塑
  - 不改 Round61 SCD cadence（仍是 iter2 `once_per_phase`, `alpha=0.999`）

## C. Allowed knobs

- knob group 1: innovation gate statistics（`target_nis`, `ema_beta`, `slope`, `gate_floor`, `warmup_static_meas`）
- knob group 2: gate coupling map（`feedback_gate_power/floor`, `scd_gate_power/floor`）

## D. Protected metrics

- must hold: `dKg_xy`, `dKg_yy`, `dKa_xx`, `rx_y`, `ry_z`
- can tolerate tiny regression: `dKg_zz`, `median`（仅在 mean/max/dKg_xx 同时改善时）
- absolutely cannot regress: 硬保护项出现明确正向回退且无主目标补偿

## E. Candidate design（deterministic, 4 candidates）

1. `r65_icg_balanced`
   - changed knobs: 平衡门控（feedback 与 SCD 同等门控）
   - rationale: 先验证主机制在中等强度下的可行性
   - expected benefit: 降低不一致静止段的过修正
   - possible risk: 门控偏弱，收益不足

2. `r65_icg_feedback_priority`
   - changed knobs: feedback 门控更强，SCD 门控更温和
   - rationale: 检查回退是否主要来自 feedback 通道
   - expected benefit: 保护 `yy/Ka_xx/rx_y`
   - possible risk: 主目标修复力度下降

3. `r65_icg_scd_priority`
   - changed knobs: SCD 门控更强，feedback 更温和
   - rationale: 检查回退是否主要来自 SCD 压制强度
   - expected benefit: 减少 cross-cov 过抑制造成的副作用
   - possible risk: dKg_xx/mean 改善不稳定

4. `r65_icg_slow_ema_guarded`
   - changed knobs: 更慢 EMA + 中等 slope
   - rationale: 降低门控抖动，做稳健性版本
   - expected benefit: 门控更平滑，保护项更稳
   - possible risk: 响应过慢，难以产生显著改进

## F. Scoring / gate

- clean win definition:
  - 对 Round61：`mean<0`, `max<=0`, `dKg_xx<0`
  - 且硬保护项无回退
- partial signal definition:
  - 至少一项主目标改善，但未满足 clean gate
- no useful signal definition:
  - 主目标无稳定改善，或保护项回退不可接受
- formalize gate:
  - 仅 clean win 才升格 formal method/result

---

Status: **planned, not executed yet**.
