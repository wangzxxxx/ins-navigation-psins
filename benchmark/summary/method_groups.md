# Method Groups

## 1. baseline
- test_system_calibration_19pos.py
- test_system_calibration_19pos_baseline.py
- test_calibration_observability_19pos.py
- analyze_observability_19pos.py
- analyze_observability_19pos_42state.py

## 2. correlation_decay / SCD
- correlation_decay_llm/test_calibration_correlation_decay.py
- correlation_decay_llm/test_calibration_scd_graded.py
- correlation_decay_llm/test_calibration_scd_isotropic.py
- correlation_decay_llm/test_calibration_scd_optimal.py
- correlation_decay_llm/test_calibration_scd_per_rotation.py
- correlation_decay_llm/test_calibration_scd_surgical.py
- correlation_decay_llm/test_calibration_scd_sweep.py
- correlation_decay_llm/test_calibration_scd_v3.py
- correlation_decay_llm/test_alpha_sweep.py
- correlation_decay_llm/test_progressive_alpha.py

## 3. markov noise
- test_calibration_markov_noise.py
- test_calibration_markov_noise_46state.py
- test_calibration_markov_noise_rts_fast.py
- test_calibration_markov_llm.py
- test_calibration_markov_pruned.py
- test_calibration_markov_pruned_llm.py

## 4. adaptive / robust
- test_calibration_adaptive_rq_llm.py
- huber_robust_kf_llm/test_calibration_huber_robust.py
- innovation_gating_llm/test_calibration_innovation_gating.py
- test_system_calibration_19pos_inflation.py

## 5. schmidt / shadow
- test_calibration_schmidt_llm.py
- shadow_kf.py
- shadow_manager.py
- shadow_manager_hybrid.py
- shadow_manager_inflation.py
- test_shadow_observer.py

## 6. hybrid / staged
- staged_calibration_llm/test_calibration_staged_llm.py
- test_system_calibration_19pos_hybrid.py
- hybrid_calibration_strategy.md

## 7. llm tuning
- calibration_path_optimizer_llm.py
- test_calibration_llm_hyperparam_tuner.py
- test_calibration_q_tuning_llm.py
- test_system_calibration_19pos_llm.py

## 8. rl
- train_calibration_rl.py
- evaluate_rl_model.py
- dual_axis_calib_ppo.zip
- plot_rl_progress.py
