# Split-Point Global Sweep Round Record

## 1. Goal
- 用户要求：尽可能找出 dual-channel split-gate 的“全局最优分离点”。
- 本轮把“分离点”严格 operationalize 为 **feedback freeze point / feedback gate floor**。
- 其余旋钮全部固定在 `r65b_split_xxzz_fb96_frozen` 体型上，不混入额外大改。

## 2. Sweep axis
- split_point values: `0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00`
- fixed knobs: Round65-B best body except feedback floor / apply_floor
- fixed dataset: same `D_ref_mainline`, seed=42

## 3. Best result
- best split point: `1.00`
- best candidate name: `splitpoint_sp100`
- score: `-4921.782285`
- delta vs Round61: `{"dKg_xx": 0.6793427753701664, "dKg_xy": 1.2980903368024936, "dKg_yy": 2.1228904312199237, "dKg_zz": -0.530219164130326, "dKa_xx": 1.2626413226137512, "rx_y": 0.23120581553284403, "ry_z": -0.03650285989100199, "mean_pct_error": 0.14272677846565074, "median_pct_error": -0.08293658744108745, "max_pct_error": 0.6793427753701664}`
- penalties: `[{"metric": "dKg_xy", "delta": 1.2980903368024936}, {"metric": "dKg_yy", "delta": 2.1228904312199237}, {"metric": "dKa_xx", "delta": 1.2626413226137512}, {"metric": "rx_y", "delta": 0.23120581553284403}]`

## 4. Judgment
- conclusion: Split-point global sweep did not produce a clean winner over Round61; best score occurs at split_point=1.00.
- trend summary: Across split_point=0.88..1.00, the top score is -4921.782 at 1.00. Best point lands on the upper boundary, so the sweep suggests the mechanism prefers almost fully frozen feedback.
- next best move: If the user wants one more refinement, keep the winning split point fixed and only run a tiny xx/zz SCD alpha ladder (e.g. ±0.0001 around the current alpha) to see whether a local no-regression pocket exists.

