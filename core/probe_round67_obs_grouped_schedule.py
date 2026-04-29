from __future__ import annotations

import copy
import gc
import json
import sys
import types
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SUMMARY_DIR = ROOT / 'psins_method_bench' / 'summary'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

ROUND61_REF_JSON = RESULTS_DIR / 'R65_mainline_round61_param_errors.json'

OUTPUT_JSON = RESULTS_DIR / 'round67_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round67_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round67_probe_2026-03-28.md'
ROUND67_RECORD_MD = SUMMARY_DIR / 'round67_record_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _apply_hybrid_scd, _resolve_target_indices
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate
from probe_round65_mainline_innovation_consistency import _build_shared_dataset

ROUND61_BASE_NAME = 'r61_s20_08988_ryz00116'

HARD_PROTECTED_KEYS = ['dKg_xy', 'dKg_yy', 'dKa_xx', 'rx_y', 'ry_z']
TARGET_KEYS = ['mean_pct_error', 'max_pct_error', 'dKg_xx', 'dKg_zz']

ROUND67_CANDIDATES = [
    {
        'name': 'r67_obs_sched_reinf_mild',
        'description': 'Observability-aware static-measurement reinforcement only (no grouped feedback change).',
        'rationale': 'Isolate one new knob: reinforce extra static measurement only when dominant-scale observability exceeds protected-group observability.',
        'obs_schedule': {
            'reinforce_enabled': True,
            'ratio_threshold': 1.02,
            'factor_base': 0.10,
            'factor_gain': 0.18,
            'factor_min': 0.08,
            'factor_max': 0.20,
            'static_start': 12,
            'static_end': 2200,
            'r_scale': 1.00,
        },
        'grouped_feedback': {
            'enabled': False,
            'ratio_ref': 1.00,
            'dom_gain': 0.0,
            'dom_max': 1.00,
            'xy_guard_gain': 0.0,
            'xy_min': 1.00,
            'prot_gain': 0.0,
            'prot_min': 1.00,
            'lever_guard': 1.00,
            'kg_yy_guard': 1.00,
            'ka_xx_guard': 1.00,
        },
    },
    {
        'name': 'r67_obs_sched_reinf_bal_grouplite',
        'description': 'Mild observability reinforcement plus light grouped feedback scaling by parameter family.',
        'rationale': 'Add a small grouped-feedback layer to protect yy/Ka_xx while keeping xx/zz reinforcement active.',
        'obs_schedule': {
            'reinforce_enabled': True,
            'ratio_threshold': 1.00,
            'factor_base': 0.12,
            'factor_gain': 0.22,
            'factor_min': 0.10,
            'factor_max': 0.24,
            'static_start': 10,
            'static_end': 2400,
            'r_scale': 1.00,
        },
        'grouped_feedback': {
            'enabled': True,
            'ratio_ref': 1.00,
            'dom_gain': 0.15,
            'dom_max': 1.06,
            'xy_guard_gain': 0.10,
            'xy_min': 0.90,
            'prot_gain': 0.22,
            'prot_min': 0.88,
            'lever_guard': 0.85,
            'kg_yy_guard': 0.92,
            'ka_xx_guard': 0.92,
        },
    },
    {
        'name': 'r67_grouped_conservative_guard',
        'description': 'Grouped conservative fusion only (no extra measurement), with explicit yy/Ka_xx/lever guard.',
        'rationale': 'Paper-friendly grouped-update baseline: keep measurement schedule unchanged and make feedback fusion conservative for protected groups.',
        'obs_schedule': {
            'reinforce_enabled': False,
            'ratio_threshold': 9.99,
            'factor_base': 0.0,
            'factor_gain': 0.0,
            'factor_min': 0.0,
            'factor_max': 0.0,
            'static_start': 999999,
            'static_end': 0,
            'r_scale': 1.00,
        },
        'grouped_feedback': {
            'enabled': True,
            'ratio_ref': 1.00,
            'dom_gain': 0.18,
            'dom_max': 1.05,
            'xy_guard_gain': 0.20,
            'xy_min': 0.86,
            'prot_gain': 0.34,
            'prot_min': 0.78,
            'lever_guard': 0.60,
            'kg_yy_guard': 0.70,
            'ka_xx_guard': 0.72,
        },
    },
    {
        'name': 'r67_async_static_window_guard',
        'description': 'Asynchronous static-window reinforcement (skip early windows) plus protected grouped guard.',
        'rationale': 'Test grouped/asynchronous update scheduling by enabling reinforcement only after static evidence stabilizes.',
        'obs_schedule': {
            'reinforce_enabled': True,
            'ratio_threshold': 1.05,
            'factor_base': 0.08,
            'factor_gain': 0.16,
            'factor_min': 0.06,
            'factor_max': 0.18,
            'static_start': 40,
            'static_end': 2600,
            'r_scale': 1.05,
        },
        'grouped_feedback': {
            'enabled': True,
            'ratio_ref': 1.02,
            'dom_gain': 0.12,
            'dom_max': 1.04,
            'xy_guard_gain': 0.16,
            'xy_min': 0.88,
            'prot_gain': 0.28,
            'prot_min': 0.82,
            'lever_guard': 0.70,
            'kg_yy_guard': 0.80,
            'ka_xx_guard': 0.80,
        },
    },
]


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _merge_round67_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['obs_schedule'] = copy.deepcopy(extra_candidate['obs_schedule'])
    merged['grouped_feedback'] = copy.deepcopy(extra_candidate['grouped_feedback'])
    merged['round67_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _state_group_observability(mod, kf, state_indices: list[int]):
    H = mod.np.asarray(kf.get('Hk'), dtype=float)
    P = mod.np.asarray(kf.get('Pxk'), dtype=float)
    if H.size == 0 or P.size == 0:
        return 0.0

    HP = H @ P @ H.T
    denom = float(mod.np.sqrt(max(float(mod.np.trace(HP)), 1e-30)))
    denom = max(denom, 1e-12)

    vals = []
    for idx in state_indices:
        try:
            v = H @ P[:, int(idx)]
            vals.append(float(mod.np.linalg.norm(v)) / denom)
        except Exception:
            continue
    if not vals:
        return 0.0
    return float(mod.np.mean(vals))


def _apply_grouped_feedback_policy(policy: dict, grouped_cfg: dict, obs_ratio_iter: float):
    out = copy.deepcopy(policy)
    if not grouped_cfg.get('enabled', False):
        return out, {
            'enabled': False,
            'obs_ratio_iter': float(obs_ratio_iter),
            'dom_mult': 1.0,
            'xy_mult': 1.0,
            'prot_mult': 1.0,
        }

    delta = max(float(obs_ratio_iter) - float(grouped_cfg['ratio_ref']), 0.0)

    dom_mult = min(float(grouped_cfg['dom_max']), 1.0 + float(grouped_cfg['dom_gain']) * delta)
    dom_mult = max(dom_mult, 1.0)

    xy_mult = 1.0 - float(grouped_cfg['xy_guard_gain']) * delta
    xy_mult = max(float(grouped_cfg['xy_min']), min(1.0, xy_mult))

    prot_mult = 1.0 - float(grouped_cfg['prot_gain']) * delta
    prot_mult = max(float(grouped_cfg['prot_min']), min(1.0, prot_mult))

    sam = copy.deepcopy(out.get('state_alpha_mult', {}))
    sam[12] = float(sam.get(12, 1.0)) * dom_mult
    sam[20] = float(sam.get(20, 1.0)) * dom_mult
    sam[15] = float(sam.get(15, 1.0)) * xy_mult
    sam[16] = float(sam.get(16, 1.0)) * prot_mult
    sam[21] = float(sam.get(21, 1.0)) * prot_mult
    out['state_alpha_mult'] = sam

    return out, {
        'enabled': True,
        'obs_ratio_iter': float(obs_ratio_iter),
        'dom_mult': float(dom_mult),
        'xy_mult': float(xy_mult),
        'prot_mult': float(prot_mult),
    }


def _apply_post_feedback_protected_guard(mod, clbt_before: dict, clbt_after: dict, grouped_cfg: dict):
    if not grouped_cfg.get('enabled', False):
        return clbt_after

    out = copy.deepcopy(clbt_after)

    lever_guard = float(grouped_cfg.get('lever_guard', 1.0))
    lever_guard = max(0.0, min(1.0, lever_guard))
    out['rx'][1] = clbt_before['rx'][1] + lever_guard * (out['rx'][1] - clbt_before['rx'][1])
    out['ry'][2] = clbt_before['ry'][2] + lever_guard * (out['ry'][2] - clbt_before['ry'][2])

    kg_yy_guard = float(grouped_cfg.get('kg_yy_guard', 1.0))
    kg_yy_guard = max(0.0, min(1.0, kg_yy_guard))
    out['Kg'][1, 1] = clbt_before['Kg'][1, 1] + kg_yy_guard * (out['Kg'][1, 1] - clbt_before['Kg'][1, 1])

    ka_xx_guard = float(grouped_cfg.get('ka_xx_guard', 1.0))
    ka_xx_guard = max(0.0, min(1.0, ka_xx_guard))
    out['Ka'][0, 0] = clbt_before['Ka'][0, 0] + ka_xx_guard * (out['Ka'][0, 0] - clbt_before['Ka'][0, 0])

    return out


def _run_internalized_hybrid_scd_obs_grouped(
    method_mod,
    mod,
    imu1,
    pos0,
    ts,
    bi_g,
    bi_a,
    tau_g,
    tau_a,
    label,
    scd_cfg,
    obs_sched_cfg,
    grouped_cfg,
):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    nn, _, nts, _ = mod.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

    k = frq2
    for k in range(frq2, min(5 * 60 * 2 * frq2, len(imu1)), 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    kstatic = k - 3 * frq2

    clbt = {
        'Kg': mod.np.eye(3), 'Ka': mod.np.eye(3), 'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3), 'db': mod.np.zeros(3),
        'rx': mod.np.zeros(3), 'ry': mod.np.zeros(3),
    }

    length = len(imu1)
    dotwf = mod.imudot(imu1, 5.0)
    P_trace, X_trace, iter_bounds = [], [], []
    feedback_log = []
    schedule_log = []
    scd_log = []
    obs_log = []

    target_indices = _resolve_target_indices(method_mod, scd_cfg['target'])
    apply_policy_names = set(scd_cfg.get('apply_policy_names', []))

    dominant_states = [12, 15, 20]
    protected_states = [16, 21, 31, 35]

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(method_mod.ITERATION_POLICIES):
        print(f'  [{label}] {policy["name"]} ({it+1}/{len(method_mod.ITERATION_POLICIES)})')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        prior_diag = method_mod._configure_iteration_prior(mod, kf, policy)
        base_q = mod.np.array(kf['Qk'], dtype=float).copy()
        base_r = mod.np.array(kf['Rk'], dtype=float).copy()

        if policy.get('readout_only'):
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0

        n_static_meas = 0
        n_dynamic_sched = 0
        n_late_sched = 0
        n_scd = 0
        n_transition_eligible = 0
        n_reinforced = 0

        was_rotating = False
        time_since_rot_stop = 999.0
        scd_applied_this_phase = False
        scd_enabled_here = (policy['name'] in apply_policy_names) and (not policy.get('readout_only'))

        ratio_trace = []
        reinforce_factor_trace = []

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]
            vm = imu1[k:k1+1, 3:6]
            dwb = mod.np.mean(dotwf[k:k1+1, 0:3], axis=0)

            phim, dvbm = mod.cnscl(mod.np.hstack((wm, vm)))
            phim = clbt['Kg'] @ phim - clbt['eb'] * nts
            dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
            wb = phim / nts
            fb = dvbm / nts

            SS = mod.imulvS(wb, dwb, Cba)
            fL = SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry']))
            fn = mod.qmulv(qnb, fb - clbt['Ka2'] * (fb**2) - fL)
            vn = vn + (mod.rotv(-wnie * nts / 2, fn) + gn) * nts
            qnb = mod.qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            Ft = mod.getFt_42(fb, wb, mod.q2mat(qnb), wnie, SS, tau_g, tau_a)
            kf['Phikk_1'] = mod.np.eye(42) + Ft * nts
            kf = mod.kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                is_static = bool(mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph)
                progress = float(k) / float(length)
                method_mod._set_cov_schedule(kf, base_q, base_r, policy, progress, is_static)

                if not is_static:
                    was_rotating = True
                    time_since_rot_stop = 0.0
                    scd_applied_this_phase = False
                else:
                    if was_rotating:
                        was_rotating = False
                        time_since_rot_stop = 0.0
                    else:
                        time_since_rot_stop += 0.2

                if is_static:
                    n_static_meas += 1

                    obs_dom = _state_group_observability(mod, kf, dominant_states)
                    obs_prot = _state_group_observability(mod, kf, protected_states)
                    obs_ratio = float(obs_dom / max(obs_prot, 1e-12))
                    ratio_trace.append(obs_ratio)

                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')

                    if bool(obs_sched_cfg.get('reinforce_enabled', False)):
                        in_window = (
                            int(obs_sched_cfg['static_start']) <= n_static_meas <= int(obs_sched_cfg['static_end'])
                        )
                        if in_window and obs_ratio >= float(obs_sched_cfg['ratio_threshold']):
                            fac = (
                                float(obs_sched_cfg['factor_base'])
                                + float(obs_sched_cfg['factor_gain']) * (obs_ratio - float(obs_sched_cfg['ratio_threshold']))
                            )
                            fac = max(float(obs_sched_cfg['factor_min']), min(float(obs_sched_cfg['factor_max']), fac))
                            r_scale = float(obs_sched_cfg.get('r_scale', 1.0))

                            if abs(r_scale - 1.0) > 1e-12:
                                r_keep = mod.np.array(kf['Rk'], dtype=float).copy()
                                kf['Rk'][:, :] = r_keep * r_scale
                                kf = mod.kfupdate(kf, yk=fac * vn, TimeMeasBoth='M')
                                kf['Rk'][:, :] = r_keep
                            else:
                                kf = mod.kfupdate(kf, yk=fac * vn, TimeMeasBoth='M')

                            n_reinforced += 1
                            reinforce_factor_trace.append(float(fac))

                    if scd_enabled_here and time_since_rot_stop >= float(scd_cfg['transition_duration']):
                        n_transition_eligible += 1
                        if scd_cfg['mode'] == 'once_per_phase':
                            if not scd_applied_this_phase:
                                _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices)
                                scd_applied_this_phase = True
                                n_scd += 1
                        elif scd_cfg['mode'] == 'repeat_after_transition':
                            _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices)
                            scd_applied_this_phase = True
                            n_scd += 1
                        else:
                            raise KeyError(f"Unknown SCD mode: {scd_cfg['mode']}")
                else:
                    n_dynamic_sched += 1

                if progress >= policy.get('late_release_frac', 2.0):
                    n_late_sched += 1
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        obs_ratio_iter = float(mod.np.median(ratio_trace)) if ratio_trace else 1.0
        feedback_policy, grouped_meta = _apply_grouped_feedback_policy(policy, grouped_cfg, obs_ratio_iter)

        if not policy.get('readout_only'):
            clbt_before = copy.deepcopy(clbt)
            clbt, meta = method_mod._apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, feedback_policy)
            clbt = _apply_post_feedback_protected_guard(mod, clbt_before, clbt, grouped_cfg)

            meta = dict(meta)
            meta['round67_obs_ratio_iter'] = float(obs_ratio_iter)
            meta['round67_grouped_meta'] = grouped_meta
            meta['round67_grouped_cfg'] = copy.deepcopy(grouped_cfg)
            feedback_log.append(meta)

        schedule_log.append({
            'policy_name': policy['name'],
            'n_static_meas': int(n_static_meas),
            'n_dynamic_sched': int(n_dynamic_sched),
            'n_late_sched': int(n_late_sched),
            'late_release_frac': float(policy.get('late_release_frac', 1.0)),
            'n_reinforced': int(n_reinforced),
        })
        scd_log.append({
            'policy_name': policy['name'],
            'enabled': bool(scd_enabled_here),
            'mode': scd_cfg['mode'],
            'alpha': float(scd_cfg['alpha']),
            'transition_duration': float(scd_cfg['transition_duration']),
            'target': scd_cfg['target'],
            'target_indices': [int(x) for x in target_indices],
            'bias_to_target': bool(scd_cfg.get('bias_to_target', True)),
            'n_transition_eligible': int(n_transition_eligible),
            'n_scd': int(n_scd),
        })
        obs_log.append({
            'policy_name': policy['name'],
            'obs_ratio_median': float(mod.np.median(ratio_trace)) if ratio_trace else None,
            'obs_ratio_mean': float(mod.np.mean(ratio_trace)) if ratio_trace else None,
            'obs_ratio_max': float(mod.np.max(ratio_trace)) if ratio_trace else None,
            'reinforce_factor_mean': float(mod.np.mean(reinforce_factor_trace)) if reinforce_factor_trace else None,
            'reinforce_factor_max': float(mod.np.max(reinforce_factor_trace)) if reinforce_factor_trace else None,
            'obs_schedule_cfg': copy.deepcopy(obs_sched_cfg),
            'grouped_feedback_cfg': copy.deepcopy(grouped_cfg),
            'grouped_feedback_meta': grouped_meta,
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
        'iteration_policies': method_mod.ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'scd_log': scd_log,
        'obs_group_log': obs_log,
        'policy': (
            'Round67 keeps the Round61 backbone and adds two classical structures: '
            '(1) observability-aware static measurement reinforcement; '
            '(2) grouped conservative feedback scaling for dominant vs protected parameter families.'
        ),
    }


def _compute_payload(source_mod, clbt, variant: str, method_file: str, extra: dict | None = None):
    param_errors, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': list(param_errors.keys()),
        'param_errors': param_errors,
        'focus_scale_pct': focus,
        'lever_guard_pct': lever,
        'overall': overall,
        'extra': extra or {},
    }


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _score_candidate(delta_vs_r61: dict):
    penalties = []
    for key in HARD_PROTECTED_KEYS:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.30 * float(delta_vs_r61['mean_pct_error'])
    score += -1.00 * float(delta_vs_r61['max_pct_error'])
    score += -0.85 * float(delta_vs_r61['dKg_xx'])
    score += -0.55 * float(delta_vs_r61['dKg_zz'])
    score += -0.35 * float(delta_vs_r61['median_pct_error'])
    score += -0.25 * float(delta_vs_r61['dKg_xy'])
    score += -0.30 * float(delta_vs_r61['dKg_yy'])
    score += -0.25 * float(delta_vs_r61['dKa_xx'])
    score += -0.20 * float(delta_vs_r61['rx_y'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])

    return float(score), penalties


def _is_clean_winner(delta_vs_r61: dict, penalties: list[dict]):
    if penalties:
        return False
    return (
        float(delta_vs_r61['mean_pct_error']) < 0.0
        and float(delta_vs_r61['max_pct_error']) <= 0.0
        and float(delta_vs_r61['dKg_xx']) < 0.0
    )


def _selection_note(delta_vs_r61: dict, penalties: list[dict]):
    if _is_clean_winner(delta_vs_r61, penalties):
        return 'Clean same-dataset winner over Round61 with observability-aware grouped scheduling.'

    if penalties:
        return f'Protected regression detected vs Round61: {penalties}'

    improved = [k for k in TARGET_KEYS if float(delta_vs_r61[k]) < 0.0]
    if improved:
        return f'Partial signal on {improved}, but clean gate (mean/max/dKg_xx + no protected regressions) not satisfied.'
    return 'No useful same-dataset signal over Round61.'


def _relative_improvement_block(baseline_payload: dict, candidate_payload: dict, keys: list[str]):
    out = {}
    for key in keys:
        if key in candidate_payload['param_errors']:
            b = float(baseline_payload['param_errors'][key]['pct_error'])
            c = float(candidate_payload['param_errors'][key]['pct_error'])
        else:
            b = float(baseline_payload['overall'][key])
            c = float(candidate_payload['overall'][key])
        out[key] = {
            'baseline_pct_error': b,
            'candidate_pct_error': c,
            'delta_pct_points': b - c,
            'relative_improvement_pct': ((b - c) / b * 100.0) if abs(b) > 1e-15 else None,
        }
    return out


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round67 换到新轴：在 Round61 稳定主干上，只引入 **observability-aware static measurement scheduling + grouped conservative feedback fusion**。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed mainline dataset (same as Round65 / Round66)')
    lines.append('')
    lines.append(f"- seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- arw: `{summary['dataset']['noise_config']['arw_dpsh']} dps/√h`")
    lines.append(f"- vrw: `{summary['dataset']['noise_config']['vrw_ugpsHz']} ug/√Hz`")
    lines.append(f"- bi_g: `{summary['dataset']['noise_config']['bi_g_dph']} dph`, bi_a: `{summary['dataset']['noise_config']['bi_a_ug']} ug`")
    lines.append('- source trajectory family: `round53_internalized_trustcov_release::_build_dataset`')
    lines.append('')
    lines.append('## 2. Round67 candidates vs Round61')
    lines.append('')
    lines.append('| candidate | dKg_xx Δ | dKg_xy Δ | dKg_yy Δ | dKg_zz Δ | dKa_xx Δ | rx_y Δ | ry_z Δ | mean Δ | median Δ | max Δ | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d = cand['delta_vs_round61']
        lines.append(
            f"| `{name}` | {d['dKg_xx']:.6f} | {d['dKg_xy']:.6f} | {d['dKg_yy']:.6f} | {d['dKg_zz']:.6f} | {d['dKa_xx']:.6f} | {d['rx_y']:.6f} | {d['ry_z']:.6f} | {d['mean_pct_error']:.6f} | {d['median_pct_error']:.6f} | {d['max_pct_error']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Decision')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append('- decision: formalize as Round67 method')
        lines.append(f"- reason: {summary['winner']['reason']}")
    else:
        lines.append('- winner: **none**')
        lines.append('- decision: keep probe-only')
        lines.append(f"- reason: {summary['no_winner_reason']}")
    lines.append(f"- strongest signal: `{summary['strongest_signal']['name']}` / {summary['strongest_signal']['signal']}")
    lines.append(f"- next best move: {summary['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict):
    lines = []
    lines.append('# Round67 Record (new mechanism probe)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `new mechanism probe`')
    lines.append(f"- Base candidate: `{ROUND61_BASE_NAME}`")
    lines.append('- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65 / Round66)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. 本轮目标')
    lines.append('- Preserve Round61 backbone (same trust-feedback + iter2 once-per-phase SCD body).')
    lines.append('- New mechanism axis: observability-aware static measurement scheduling + grouped conservative feedback fusion.')
    lines.append('- Avoid consistency-gating variants and avoid multi-knob mixed redesign.')
    lines.append('')
    lines.append('## C. Allowed knobs')
    lines.append('- knob group 1: static measurement reinforcement schedule by observability ratio (threshold/factor/window).')
    lines.append('- knob group 2: grouped feedback multipliers (dominant xx/xy/zz vs protected yy/Ka_xx, with lever post-guard).')
    lines.append('- locked/no-change: Round61 dataset, Round61 base feedback route, Round61 SCD core path.')
    lines.append('')
    lines.append('## D. Protected metrics and clean-win gate')
    lines.append('- hard-protected metrics: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z')
    lines.append('- clean-win gate vs Round61: mean<0, max<=0, dKg_xx<0 and hard-protected no regression')
    lines.append('- formalize gate: only clean winner can be promoted')
    lines.append('')
    lines.append('## E. Candidate design (3-5 max)')
    for idx, candidate in enumerate(ROUND67_CANDIDATES, start=1):
        lines.append(f'### candidate {idx}')
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append(f"- obs_schedule: `{json.dumps(candidate['obs_schedule'], ensure_ascii=False)}`")
        lines.append(f"- grouped_feedback: `{json.dumps(candidate['grouped_feedback'], ensure_ascii=False)}`")
        lines.append('')
    lines.append('## F. Result summary')
    lines.append(f"- winner: `{summary['winner']['name']}`" if summary['winner'] else '- winner: none')
    lines.append(f"- result class: `{summary['result_classification']}`")
    lines.append(f"- one-line conclusion: {summary['conclusion_line']}")
    lines.append(f"- strongest signal: {summary['strongest_signal']['signal']}")
    lines.append('')
    lines.append('## G. Mechanism learning and next move')
    lines.append(f"- mechanism learning: {summary['mechanism_learning']}")
    lines.append(f"- next best move: {summary['next_best_move']}")
    lines.append('')
    lines.append('## H. Artifacts')
    lines.append(f"- candidate_json: `{CANDIDATE_JSON}`")
    lines.append(f"- summary_json: `{OUTPUT_JSON}`")
    lines.append(f"- report_md: `{REPORT_MD}`")
    lines.append(f"- formal_method_file: `{summary.get('formal_method_file')}`")
    lines.append(f"- formal_result_json: `{summary.get('formal_result_json')}`")
    lines.append('')
    return '\n'.join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))

    source_mod = load_module('markov_pruned_source_round67', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round67_OBS_GROUPED_SCHEDULE',
        'round_type': 'new mechanism probe',
        'mechanism_axis': 'observability-aware static measurement scheduling + grouped conservative feedback fusion on Round61 backbone',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65/66 mainline',
        },
        'protected_metrics': HARD_PROTECTED_KEYS,
        'clean_win_gate': 'mean<0, max<=0, dKg_xx<0 and no hard-protected regression vs Round61',
        'round67_candidates': ROUND67_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round67_OBS_GROUPED_SCHEDULE',
        'round_type': 'new mechanism probe',
        'mechanism': 'Round61 backbone + observability-aware static scheduling + grouped conservative feedback fusion',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND67_CANDIDATES],
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
        'result_classification': None,
        'strongest_signal': None,
        'next_best_move': None,
        'formal_method_file': None,
        'formal_result_json': None,
    }

    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    for idx, candidate in enumerate(ROUND67_CANDIDATES, start=1):
        merged_candidate = _merge_round67_candidate(candidate)

        method_mod = load_module(f'markov_method_round67_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)

        result = list(_run_internalized_hybrid_scd_obs_grouped(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R67-OBS-GROUP-{idx}',
            scd_cfg=merged_candidate['scd'],
            obs_sched_cfg=merged_candidate['obs_schedule'],
            grouped_cfg=merged_candidate['grouped_feedback'],
        ))
        clbt_candidate = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
            'obs_group_log': result[4].get('obs_group_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"r67_obs_grouped_{candidate['name']}",
            method_file='probe_round67_obs_grouped_schedule::obs_grouped_round61_backbone',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'obs_schedule': copy.deepcopy(candidate['obs_schedule']),
                'grouped_feedback': copy.deepcopy(candidate['grouped_feedback']),
                'scd_cfg': copy.deepcopy(merged_candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R67_obs_grouped_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }

        score, penalties = _score_candidate(delta_vs_r61)
        note = _selection_note(delta_vs_r61, penalties)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'obs_schedule': copy.deepcopy(candidate['obs_schedule']),
            'grouped_feedback': copy.deepcopy(candidate['grouped_feedback']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
            'runtime_log': payload_candidate['extra']['runtime_log'],
            'vs_round61_relative_improvement': _relative_improvement_block(
                round61_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'score': score,
            'penalties': penalties,
            'note': note,
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best = out['candidates'][best_name]
    best_delta = best['delta_vs_round61']
    best_penalties = best['selection']['penalties']

    if _is_clean_winner(best_delta, best_penalties):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset winner over Round61 under observability-aware grouped scheduling.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round67 produced a clean same-dataset winner over Round61.'

        formal_method_file = METHOD_DIR / f"method_42state_gm1_round67_obs_grouped_{best_name}.py"
        formal_result_json = RESULTS_DIR / f"R67_42state_gm1_round67_obs_grouped_{best_name}_param_errors.json"
        formal_method_file.write_text(
            (
                '# Auto-generated Round67 formalization placeholder.\n'
                '# Winner configuration is recorded in results/round67_probe_summary.json and round67_candidates.json.\n'
                '# For reproducibility, use psins_method_bench/scripts/probe_round67_obs_grouped_schedule.py with the winner candidate.\n'
            ),
            encoding='utf-8',
        )
        formal_result_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
        out['formal_method_file'] = str(formal_method_file)
        out['formal_result_json'] = str(formal_result_json)
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate passed the same-dataset Round61 clean-win gate under the Round67 observability/grouped scheduling axis.'
        improved_targets = [k for k in TARGET_KEYS if float(best_delta[k]) < 0.0]
        if best_penalties:
            out['result_classification'] = 'no useful signal'
        else:
            out['result_classification'] = 'partial signal' if improved_targets else 'no useful signal'
        out['conclusion_line'] = 'Round67 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.'

    out['strongest_signal'] = {
        'name': best_name,
        'signal': (
            f"best candidate {best_name}: "
            f"dKg_xx Δ={best_delta['dKg_xx']:.6f}, dKg_zz Δ={best_delta['dKg_zz']:.6f}, "
            f"mean Δ={best_delta['mean_pct_error']:.6f}, max Δ={best_delta['max_pct_error']:.6f}"
        ),
        'regressions': str(best_penalties),
    }

    out['mechanism_learning'] = (
        'Round67 keeps Round61 unchanged at backbone level and injects only a classical observability-aware static schedule '
        'plus grouped conservative fusion; current batch shows whether this axis can improve xx/xy/zz while preserving yy/Ka_xx/lever.'
    )
    out['next_best_move'] = (
        'If no clean winner, keep the best Round67 candidate as seed and run a one-knob refinement: '
        'fix grouped-feedback map and sweep only static reinforcement factor window (base/max) to test clean-gate feasibility.'
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND67_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND67_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND67_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
