# Adaptive Robust Methods

Runnable wrappers around the 36-state calibration KF with adaptive/robust measurement update strategies.

## Methods

| File | Method | Status |
|---|---|---|
| `method_adaptive_rq.py` | Adaptive R/Q with LLM-driven R inflation | ✅ runs (3 iters) |
| `method_huber_robust.py` | Huber robust KF update (Mahalanobis distance gating) | ✅ runs (5 iters) |
| `method_innovation_gating.py` | Innovation gating (scout pass + toxic zone skip) | ✅ runs (5 iters) |
| `method_attention_inflation.py` | 43-state attention-driven covariance inflation / shadow manager | ❌ too slow (>15 min, LLM + 43-state dual KF) |

## Key files

- `common_setup.py` — shared `build_dataset()` used by adaptive_rq / huber / gating
- `run_adaptive_robust_methods.py` — batch runner, saves to `results/adaptive_robust_results.json`

## Notes

- All 3 working methods are 36-state KF, sourced from `tmp_psins_py/psins_py/`.
- LLM calls fall back gracefully (default gamma/schedule) when API key is missing.
- The inflation script uses `clbtkfinit` (43-state) with `shadow_manager_inflation` — fundamentally different state vector. Needs dedicated rework to benchmark separately.
