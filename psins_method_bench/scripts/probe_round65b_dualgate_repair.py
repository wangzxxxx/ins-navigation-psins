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

ROUND65_SUMMARY_JSON = RESULTS_DIR / 'round65_probe_summary.json'
ROUND61_REF_JSON = RESULTS_DIR / 'R65_mainline_round61_param_errors.json'

OUTPUT_JSON = RESULTS_DIR / 'round65b_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round65b_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round65b_probe_2026-03-28.md'
ROUND65B_RECORD_MD = SUMMARY_DIR / 'round65b_record_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _apply_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate
from probe_round65_mainline_innovation_consistency import _build_shared_dataset


ROUND61_BASE_NAME = 'r61_s20_08988_ryz00116'

PRIMARY_REPAIR_KEYS = ['dKg_yy', 'dKa_xx', 'rx_y']
HARD_PROTECTED_KEYS = ['dKg_xy', 'dKg_yy', 'dKa_xx', 'rx_y', 'ry_z']


ROUND65B_CANDIDATES = [
    {
        'name': 'r65b_split_xxzz_fb92_guard',
        'description': 'Dual-channel split gate: high-floor feedback gate with yy/Ka_xx local guard, while SCD gate stays adaptive on xx/zz only.',
        'rationale': 'First repair move: decouple channels and keep feedback close to Round61 on protected paths; reserve adaptation mainly for xx/zz SCD suppression.',
        'feedback_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.04,
            'slope': 1.10,
            'gate_floor': 0.92,
            'warmup_static_meas': 8,
            'power': 1.0,
            'apply_floor': 0.92,
        },
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.12,
            'slope': 1.90,
            'gate_floor': 0.60,
            'warmup_static_meas': 8,
            'power': 1.20,
            'apply_floor': 0.60,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9988,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {16: 1.010, 21: 1.008},
            },
        },
    },
    {
        'name': 'r65b_split_xxzz_fb96_frozen',
        'description': 'More frozen feedback channel (floor=0.96) with stronger adaptive xx/zz SCD gate.',
        'rationale': 'Stress-test whether most regressions came from feedback-path gate drift; keep feedback near baseline and move adaptation burden to xx/zz SCD only.',
        'feedback_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.03,
            'slope': 1.00,
            'gate_floor': 0.96,
            'warmup_static_meas': 8,
            'power': 1.0,
            'apply_floor': 0.96,
        },
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.12,
            'slope': 2.10,
            'gate_floor': 0.55,
            'warmup_static_meas': 8,
            'power': 1.35,
            'apply_floor': 0.55,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9987,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {16: 1.014, 21: 1.012},
            },
        },
    },
    {
        'name': 'r65b_split_xxzz_fb94_rxguard',
        'description': 'Split gate with moderate feedback floor plus a narrow rx_y post-guard.',
        'rationale': 'Keep split-gate body unchanged and test whether a minimal lever post-guard can repair rx_y without broad retuning.',
        'feedback_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.04,
            'slope': 1.20,
            'gate_floor': 0.94,
            'warmup_static_meas': 8,
            'power': 1.0,
            'apply_floor': 0.94,
        },
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.11,
            'slope': 1.95,
            'gate_floor': 0.58,
            'warmup_static_meas': 8,
            'power': 1.25,
            'apply_floor': 0.58,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9988,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {16: 1.012, 21: 1.010},
            },
        },
        'post_rx_y_mult': 1.00045,
    },
    {
        'name': 'r65b_split_scale_navonly_guarded',
        'description': 'Feedback guard kept, SCD remains adaptive but expanded to scale-block nav-only suppression.',
        'rationale': 'Ablation-style check: if xx/zz-only SCD is too narrow, test a bounded scale-block/nav-only version without reopening bias coupling.',
        'feedback_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.04,
            'slope': 1.18,
            'gate_floor': 0.94,
            'warmup_static_meas': 8,
            'power': 1.0,
            'apply_floor': 0.94,
        },
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.11,
            'slope': 1.70,
            'gate_floor': 0.62,
            'warmup_static_meas': 8,
            'power': 1.20,
            'apply_floor': 0.62,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9990,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': False,
            'apply_policy_names': ['iter2_commit'],
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {16: 1.012, 21: 1.012},
            },
        },
    },
]


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _merge_round65b_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']

    merged_patches = copy.deepcopy(merged.get('iter_patches', {}))
    for iter_idx, patch in extra_candidate.get('iter_patches', {}).items():
        dst = merged_patches.setdefault(iter_idx, {})
        for key, value in patch.items():
            if isinstance(value, dict):
                current = copy.deepcopy(dst.get(key, {}))
                current.update(copy.deepcopy(value))
                dst[key] = current
            else:
                dst[key] = copy.deepcopy(value)
    merged['iter_patches'] = merged_patches

    if extra_candidate.get('post_rx_y_mult') is not None:
        merged['post_rx_y_mult'] = float(extra_candidate['post_rx_y_mult'])
    if extra_candidate.get('post_ry_z_mult') is not None:
        merged['post_ry_z_mult'] = float(extra_candidate['post_ry_z_mult'])

    merged['feedback_channel'] = copy.deepcopy(extra_candidate['feedback_channel'])
    merged['scd_channel'] = copy.deepcopy(extra_candidate['scd_channel'])
    merged['scd'] = copy.deepcopy(extra_candidate['scd'])
    merged['round65b_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _build_feedback_policy(policy: dict, feedback_gate: float):
    gated = copy.deepcopy(policy)

    def _blend_to_unity(value: float, gate: float):
        return 1.0 + gate * (float(value) - 1.0)

    if 'selected_alpha_floor' in gated:
        gated['selected_alpha_floor'] = _blend_to_unity(float(gated['selected_alpha_floor']), feedback_gate)
    if 'selected_alpha_span' in gated:
        gated['selected_alpha_span'] = float(gated['selected_alpha_span']) * float(feedback_gate)

    for key in ['other_scale_alpha', 'ka2_alpha', 'lever_alpha', 'markov_alpha']:
        if key in gated:
            gated[key] = _blend_to_unity(float(gated[key]), feedback_gate)
    return gated


def _resolve_target_indices(method_mod, target_name: str):
    if target_name == 'xxzz_pair':
        return [12, 20]
    if target_name == 'selected':
        return [int(idx) for idx in method_mod.SELECTED_SCALE_STATES]
    if target_name == 'scale_block':
        return list(range(12, 27))
    raise KeyError(f'Unknown SCD target: {target_name}')


def _innovation_norm(mod, kf):
    rk = kf.get('rk')
    if rk is None:
        return None
    rk = mod.np.asarray(rk, dtype=float).reshape(-1)
    if rk.size == 0:
        return None

    Rk = mod.np.asarray(kf['Rk'], dtype=float)
    try:
        nis = float(rk.T @ mod.np.linalg.solve(Rk, rk))
    except Exception:
        nis = float(rk.T @ mod.np.linalg.pinv(Rk) @ rk)
    return nis / float(max(rk.size, 1))


def _channel_gate_from_nis(nis_norm: float, cfg: dict, ema_key: str, count_key: str, gate_state: dict):
    beta = float(cfg['ema_beta'])
    gate_state[ema_key] = (1.0 - beta) * float(gate_state[ema_key]) + beta * float(nis_norm)
    gate_state[count_key] += 1

    dev = abs(float(gate_state[ema_key]) - float(cfg['target_nis']))
    raw_gate = 1.0 / (1.0 + float(cfg['slope']) * dev)
    raw_gate = max(0.0, min(1.0, float(raw_gate)))

    if int(gate_state[count_key]) <= int(cfg.get('warmup_static_meas', 0)):
        eff = 1.0
    else:
        eff = max(float(cfg['gate_floor']), raw_gate)
    return float(raw_gate), float(eff), float(gate_state[ema_key])


def _run_internalized_hybrid_scd_dualgate(
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
    feedback_cfg,
    scd_gate_cfg,
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
    dual_gate_log = []

    target_indices = _resolve_target_indices(method_mod, scd_cfg['target'])
    apply_policy_names = set(scd_cfg.get('apply_policy_names', []))

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

        was_rotating = False
        time_since_rot_stop = 999.0
        scd_applied_this_phase = False
        scd_enabled_here = (policy['name'] in apply_policy_names) and (not policy.get('readout_only'))

        gate_state = {
            'feedback_ema': 1.0,
            'scd_ema': 1.0,
            'feedback_count': 0,
            'scd_count': 0,
            'nis': [],
            'feedback_gate_raw': [],
            'feedback_gate_effective': [],
            'scd_gate_raw': [],
            'scd_gate_effective': [],
            'scd_gate_applied': [],
            'scd_alpha_applied': [],
        }

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
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                    nis_norm = _innovation_norm(mod, kf)
                    if nis_norm is None:
                        nis_norm = 1.0
                    gate_state['nis'].append(float(nis_norm))

                    fb_raw, fb_eff, fb_ema = _channel_gate_from_nis(
                        nis_norm,
                        feedback_cfg,
                        ema_key='feedback_ema',
                        count_key='feedback_count',
                        gate_state=gate_state,
                    )
                    scd_raw, scd_eff, scd_ema = _channel_gate_from_nis(
                        nis_norm,
                        scd_gate_cfg,
                        ema_key='scd_ema',
                        count_key='scd_count',
                        gate_state=gate_state,
                    )
                    gate_state['feedback_gate_raw'].append(float(fb_raw))
                    gate_state['feedback_gate_effective'].append(float(fb_eff))
                    gate_state['scd_gate_raw'].append(float(scd_raw))
                    gate_state['scd_gate_effective'].append(float(scd_eff))

                    if scd_enabled_here and time_since_rot_stop >= float(scd_cfg['transition_duration']):
                        n_transition_eligible += 1
                        scd_gate = max(
                            float(scd_gate_cfg.get('apply_floor', scd_gate_cfg['gate_floor'])),
                            float(scd_eff) ** float(scd_gate_cfg.get('power', 1.0)),
                        )
                        alpha_eff = 1.0 - scd_gate * (1.0 - float(scd_cfg['alpha']))
                        scd_cfg_eff = dict(scd_cfg)
                        scd_cfg_eff['alpha'] = float(alpha_eff)

                        if scd_cfg['mode'] == 'once_per_phase':
                            if not scd_applied_this_phase:
                                _apply_hybrid_scd(method_mod, kf, scd_cfg_eff, target_indices)
                                scd_applied_this_phase = True
                                n_scd += 1
                                gate_state['scd_gate_applied'].append(float(scd_gate))
                                gate_state['scd_alpha_applied'].append(float(alpha_eff))
                        elif scd_cfg['mode'] == 'repeat_after_transition':
                            _apply_hybrid_scd(method_mod, kf, scd_cfg_eff, target_indices)
                            scd_applied_this_phase = True
                            n_scd += 1
                            gate_state['scd_gate_applied'].append(float(scd_gate))
                            gate_state['scd_alpha_applied'].append(float(alpha_eff))
                        else:
                            raise KeyError(f"Unknown SCD mode: {scd_cfg['mode']}")
                else:
                    n_dynamic_sched += 1

                if progress >= policy.get('late_release_frac', 2.0):
                    n_late_sched += 1
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if not policy.get('readout_only'):
            fb_values = gate_state['feedback_gate_effective']
            feedback_gate = float(mod.np.median(fb_values)) if len(fb_values) > 0 else 1.0
            feedback_gate = max(
                float(feedback_cfg.get('apply_floor', feedback_cfg['gate_floor'])),
                feedback_gate ** float(feedback_cfg.get('power', 1.0)),
            )
            gated_policy = _build_feedback_policy(policy, feedback_gate)
            clbt, meta = method_mod._apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, gated_policy)
            meta = dict(meta)
            meta['dual_gate_feedback_applied'] = float(feedback_gate)
            meta['dual_gate_feedback_cfg'] = copy.deepcopy(feedback_cfg)
            feedback_log.append(meta)
        else:
            feedback_gate = 1.0
            fb_ema = gate_state['feedback_ema']
            scd_ema = gate_state['scd_ema']

        schedule_log.append({
            'policy_name': policy['name'],
            'n_static_meas': int(n_static_meas),
            'n_dynamic_sched': int(n_dynamic_sched),
            'n_late_sched': int(n_late_sched),
            'late_release_frac': float(policy.get('late_release_frac', 1.0)),
        })
        scd_log.append({
            'policy_name': policy['name'],
            'enabled': bool(scd_enabled_here),
            'mode': scd_cfg['mode'],
            'alpha_base': float(scd_cfg['alpha']),
            'transition_duration': float(scd_cfg['transition_duration']),
            'target': scd_cfg['target'],
            'target_indices': [int(x) for x in target_indices],
            'bias_to_target': bool(scd_cfg.get('bias_to_target', True)),
            'n_transition_eligible': int(n_transition_eligible),
            'n_scd': int(n_scd),
            'scd_gate_applied_mean': float(mod.np.mean(gate_state['scd_gate_applied'])) if len(gate_state['scd_gate_applied']) > 0 else None,
            'alpha_effective_mean': float(mod.np.mean(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
            'alpha_effective_min': float(mod.np.min(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
            'alpha_effective_max': float(mod.np.max(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
        })
        dual_gate_log.append({
            'policy_name': policy['name'],
            'feedback_cfg': copy.deepcopy(feedback_cfg),
            'scd_gate_cfg': copy.deepcopy(scd_gate_cfg),
            'feedback_ema_final': float(gate_state['feedback_ema']),
            'scd_ema_final': float(gate_state['scd_ema']),
            'nis_mean': float(mod.np.mean(gate_state['nis'])) if len(gate_state['nis']) > 0 else None,
            'feedback_gate_mean': float(mod.np.mean(gate_state['feedback_gate_effective'])) if len(gate_state['feedback_gate_effective']) > 0 else None,
            'feedback_gate_median': float(mod.np.median(gate_state['feedback_gate_effective'])) if len(gate_state['feedback_gate_effective']) > 0 else None,
            'feedback_gate_min': float(mod.np.min(gate_state['feedback_gate_effective'])) if len(gate_state['feedback_gate_effective']) > 0 else None,
            'feedback_gate_max': float(mod.np.max(gate_state['feedback_gate_effective'])) if len(gate_state['feedback_gate_effective']) > 0 else None,
            'feedback_gate_applied': float(feedback_gate),
            'scd_gate_mean': float(mod.np.mean(gate_state['scd_gate_effective'])) if len(gate_state['scd_gate_effective']) > 0 else None,
            'scd_gate_median': float(mod.np.median(gate_state['scd_gate_effective'])) if len(gate_state['scd_gate_effective']) > 0 else None,
            'scd_gate_min': float(mod.np.min(gate_state['scd_gate_effective'])) if len(gate_state['scd_gate_effective']) > 0 else None,
            'scd_gate_max': float(mod.np.max(gate_state['scd_gate_effective'])) if len(gate_state['scd_gate_effective']) > 0 else None,
            'scd_gate_applied_mean': float(mod.np.mean(gate_state['scd_gate_applied'])) if len(gate_state['scd_gate_applied']) > 0 else None,
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
        'iteration_policies': method_mod.ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'scd_log': scd_log,
        'dual_gate_log': dual_gate_log,
        'policy': 'Round65-B repair branch: split innovation-consistency gate into dual channels (feedback vs SCD), keep high-floor feedback guard on yy/Ka_xx path and keep adaptive SCD mainly on xx/zz path.',
    }


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _score_candidate(delta_vs_r61: dict, delta_vs_r65_ref: dict):
    penalties = []
    for key in HARD_PROTECTED_KEYS:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.35 * float(delta_vs_r61['dKg_yy'])
    score += -1.20 * float(delta_vs_r61['dKa_xx'])
    score += -1.20 * float(delta_vs_r61['rx_y'])
    score += -0.75 * float(delta_vs_r61['dKg_xx'])
    score += -0.55 * float(delta_vs_r61['dKg_zz'])
    score += -0.65 * float(delta_vs_r61['mean_pct_error'])
    score += -0.40 * float(delta_vs_r61['max_pct_error'])
    score += -0.25 * float(delta_vs_r61['dKg_xy'])
    score += -0.20 * float(delta_vs_r61['ry_z'])

    # Encourage repair relative to the strongest Round65 reference.
    score += -0.50 * float(delta_vs_r65_ref['dKg_yy'])
    score += -0.45 * float(delta_vs_r65_ref['dKa_xx'])
    score += -0.45 * float(delta_vs_r65_ref['rx_y'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])

    return float(score), penalties


def _is_clean_winner(delta_vs_r61: dict, penalties: list[dict]):
    if penalties:
        return False
    if not all(float(delta_vs_r61[k]) < 0.0 for k in PRIMARY_REPAIR_KEYS):
        return False
    return (
        float(delta_vs_r61['mean_pct_error']) < 0.0
        and float(delta_vs_r61['max_pct_error']) <= 0.0
        and float(delta_vs_r61['dKg_xx']) < 0.0
    )


def _selection_note(delta_vs_r61: dict, delta_vs_r65_ref: dict, penalties: list[dict]):
    repaired_keys = [k for k in PRIMARY_REPAIR_KEYS if float(delta_vs_r61[k]) < 0.0]

    if _is_clean_winner(delta_vs_r61, penalties):
        return 'Clean same-dataset win over Round61 with all repair-protected metrics improved.'

    if repaired_keys:
        if penalties:
            return (
                f"Repair signal on {repaired_keys} and better than Round65-ref on key regressions "
                f"(yy {delta_vs_r65_ref['dKg_yy']:.6f}, Ka_xx {delta_vs_r65_ref['dKa_xx']:.6f}, rx_y {delta_vs_r65_ref['rx_y']:.6f}), "
                f"but protected regression vs Round61 remains: {penalties}"
            )
        return f'Partial repair signal: improved {repaired_keys} vs Round61, but clean-win mainline gate (mean/max/dKg_xx) not met.'

    if penalties:
        return f'No useful repair signal; protected regression remains vs Round61: {penalties}'
    return 'Near-neutral variant; does not repair the key Round65 regression trio.'


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🩹" background-color="light-blue">')
    lines.append('Round65-B 是 **repair branch**：保持 innovation-consistency 机制，但把 feedback gate 与 SCD gate 分离，并把反馈侧保护在 yy/Ka_xx 路径上。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed setup (same-dataset only)')
    lines.append('')
    lines.append(f"- base Round61 reference: `{summary['base_round61_json']}`")
    lines.append(f"- Round65 reference used for repair delta: `{summary['base_round65_reference_name']}`")
    lines.append(f"- dataset seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append('- noise: arw=0.005*dpsh, vrw=5.0*ugpsHz, bi_g=0.002*dph, bi_a=5.0*ug, tau=300')
    lines.append('')
    lines.append('## 2. Round65-B candidates vs Round61 / Round65-ref')
    lines.append('')
    lines.append('| candidate | dKg_yy ΔR61 | dKa_xx ΔR61 | rx_y ΔR61 | dKg_xx ΔR61 | dKg_zz ΔR61 | mean ΔR61 | max ΔR61 | yy ΔR65ref | Ka_xx ΔR65ref | rx_y ΔR65ref | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d61 = cand['delta_vs_round61']
        d65 = cand['delta_vs_round65_ref']
        lines.append(
            f"| `{name}` | {d61['dKg_yy']:.6f} | {d61['dKa_xx']:.6f} | {d61['rx_y']:.6f} | {d61['dKg_xx']:.6f} | {d61['dKg_zz']:.6f} | {d61['mean_pct_error']:.6f} | {d61['max_pct_error']:.6f} | {d65['dKg_yy']:.6f} | {d65['dKa_xx']:.6f} | {d65['rx_y']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Winner / branch decision')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append('- decision: promote and formalize')
        lines.append(f"- reason: {summary['winner']['reason']}")
    else:
        lines.append('- winner: **none**')
        lines.append('- decision: keep probe-only (no formal promotion)')
        lines.append(f"- reason: {summary['no_winner_reason']}")
    lines.append(f"- strongest candidate (still regressive): `{summary['strongest_signal']['name']}` / {summary['strongest_signal']['signal']}")
    lines.append(f"- next move: {summary['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict):
    lines = []
    lines.append('# Round65-B Record (repair branch)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `repair branch`')
    lines.append(f"- Base candidate: `{ROUND61_BASE_NAME}`")
    lines.append('- Dataset / regime: `D_ref_mainline` (same as Round65, fixed noisy dataset)')
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
    lines.append('- Keep innovation-consistency mechanism, repair Round65 protected regressions via dual-channel split gate.')
    lines.append('- Primary repair targets: `dKg_yy / dKa_xx / rx_y` (same-dataset vs Round61).')
    lines.append('- Secondary target: keep SCD adaptation mainly on `xx/zz` path, avoid re-opening broad trust-map search.')
    lines.append('')
    lines.append('## C. Allowed knobs')
    lines.append('- knob group 1: dual-channel innovation gate config (feedback channel vs SCD channel with separate EMA/slope/floor).')
    lines.append('- knob group 2: narrow yy/Ka_xx local feedback guard and optional micro rx_y post-guard.')
    lines.append('- knob group 3: SCD target scope (`xxzz_pair` vs bounded `scale_block nav-only`).')
    lines.append('')
    lines.append('## D. Protected metrics')
    lines.append('- must hold: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z')
    lines.append('- repair-first protected set: dKg_yy / dKa_xx / rx_y')
    lines.append('- absolutely cannot regress for clean win: any hard-protected metric vs Round61 > 0')
    lines.append('')
    lines.append('## E. Candidate design')
    for idx, candidate in enumerate(ROUND65B_CANDIDATES, start=1):
        lines.append(f'### candidate {idx}')
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- changed knobs: feedback_channel=`{json.dumps(candidate['feedback_channel'], ensure_ascii=False)}`, scd_channel=`{json.dumps(candidate['scd_channel'], ensure_ascii=False)}`, scd=`{json.dumps(candidate['scd'], ensure_ascii=False)}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append('- expected benefit: 修复 Round65 的 yy/Ka_xx/rx_y 回退，同时保留 innovation-consistency 机制与 SCD 自适应能力。')
        lines.append('- possible risk: 修复有效但主目标 mean/max/dKg_xx 不够，仍无法通过 clean-win gate。')
        lines.append('')
    lines.append('## F. Scoring / gate')
    lines.append('- clean win gate: same-dataset vs Round61 满足 mean<0, max<=0, dKg_xx<0，且 hard-protected 无回退，并且 dKg_yy/dKa_xx/rx_y 全部 <0。')
    lines.append('- partial signal: 关键修复项（yy/Ka_xx/rx_y）至少一项改善，但 clean gate 未满足。')
    lines.append('- no useful signal: 修复项无改善或 protected 回退明显。')
    lines.append('- formalize gate: 仅 clean win 才 formalize 方法文件和正式 param_errors。')
    lines.append('')
    lines.append('## G. Result summary')
    lines.append(f"- winner: `{summary['winner']['name']}`" if summary['winner'] else '- winner: none')
    lines.append(f"- result class: `{summary['result_classification']}`")
    lines.append(f"- one-line conclusion: {summary['conclusion_line']}")
    lines.append('')
    lines.append('## H. Metric deltas vs base (Round61)')
    lines.append(f"- strongest signal: {summary['strongest_signal']['signal']}")
    lines.append(f"- key regressions: {summary['strongest_signal']['regressions']}")
    lines.append('')
    lines.append('## I. Mechanism learning')
    lines.append(f"- what worked: {summary['mechanism_learning']['worked']}")
    lines.append(f"- what did not work enough: {summary['mechanism_learning']['not_worked']}")
    lines.append(f"- structural or redistribution: {summary['mechanism_learning']['structural_or_redistribution']}")
    lines.append('')
    lines.append('## J. Next experiment generation')
    lines.append(f"- keep: {summary['next_actions']['keep']}")
    lines.append(f"- remove: {summary['next_actions']['remove']}")
    lines.append(f"- next best repair direction: {summary['next_actions']['repair_direction']}")
    lines.append(f"- next best new-mechanism direction: {summary['next_actions']['new_mechanism_direction']}")
    lines.append(f"- should formalize now? {'yes' if summary['winner'] else 'no'}")
    lines.append('')
    lines.append('## K. Artifacts')
    lines.append(f"- candidate_json: `{CANDIDATE_JSON}`")
    lines.append(f"- summary_json: `{OUTPUT_JSON}`")
    lines.append(f"- report_md: `{REPORT_MD}`")
    lines.append(f"- formal_method_file: `{summary.get('formal_method_file')}`")
    lines.append(f"- formal_result_json: `{summary.get('formal_result_json')}`")
    lines.append('')
    return '\n'.join(lines)


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round65_summary = json.loads(ROUND65_SUMMARY_JSON.read_text(encoding='utf-8'))
    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))

    round65_ref_name = round65_summary['strongest_signal']['name']
    round65_ref_json = Path(round65_summary['candidates'][round65_ref_name]['param_errors_json'])
    round65_ref_payload = json.loads(round65_ref_json.read_text(encoding='utf-8'))

    source_mod = load_module('markov_pruned_source_round65b', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round65B_DualGate_Repair',
        'round_type': 'repair branch',
        'innovation_direction': 'dual-channel innovation-consistency gated Round61 repair',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'same_dataset_round65_reference': {
            'name': round65_ref_name,
            'json': str(round65_ref_json),
        },
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65 mainline',
        },
        'protected_metrics': HARD_PROTECTED_KEYS,
        'repair_metrics': PRIMARY_REPAIR_KEYS,
        'round65b_candidates': ROUND65B_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round65B_DualGate_Repair',
        'round_type': 'repair branch',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'base_round65_reference_name': round65_ref_name,
        'base_round65_reference_json': str(round65_ref_json),
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND65B_CANDIDATES],
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

    for idx, candidate in enumerate(ROUND65B_CANDIDATES, start=1):
        merged_candidate = _merge_round65b_candidate(candidate)

        method_mod = load_module(f'markov_method_round65b_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)

        result = list(_run_internalized_hybrid_scd_dualgate(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R65B-DUAL-{idx}',
            scd_cfg=merged_candidate['scd'],
            feedback_cfg=merged_candidate['feedback_channel'],
            scd_gate_cfg=merged_candidate['scd_channel'],
        ))
        clbt_candidate = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
            'dual_gate_log': result[4].get('dual_gate_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"r65b_dualgate_{candidate['name']}",
            method_file='probe_round65b_dualgate_repair::dual_channel_icg',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
                'scd_channel': copy.deepcopy(candidate['scd_channel']),
                'scd_cfg': copy.deepcopy(candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R65B_dualgate_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }
        delta_vs_r65_ref = {
            **_delta_block(payload_candidate['focus_scale_pct'], round65_ref_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round65_ref_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round65_ref_payload['overall']),
        }

        score, penalties = _score_candidate(delta_vs_r61, delta_vs_r65_ref)
        note = _selection_note(delta_vs_r61, delta_vs_r65_ref, penalties)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
            'scd_channel': copy.deepcopy(candidate['scd_channel']),
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round65_ref': delta_vs_r65_ref,
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
            'vs_round65_ref_relative_improvement': _relative_improvement_block(
                round65_ref_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round65_ref': delta_vs_r65_ref,
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
    best_delta_r61 = best['delta_vs_round61']
    best_penalties = best['selection']['penalties']

    if _is_clean_winner(best_delta_r61, best_penalties):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset winner over Round61 under Round65-B dual-channel repair gate.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round65-B produced a clean same-dataset winner over Round61.'

        # Formalization path (only if clean win).
        formal_method_file = METHOD_DIR / f"method_42state_gm1_round65b_dualgate_repair_{best_name}.py"
        formal_result_json = RESULTS_DIR / f"R65B_42state_gm1_round65b_dualgate_repair_{best_name}_param_errors.json"
        formal_method_file.write_text(
            (
                '# Auto-generated Round65-B formalization placeholder.\n'
                '# Winner configuration is recorded in results/round65b_probe_summary.json and round65b_candidates.json.\n'
                '# For reproducibility, use psins_method_bench/scripts/probe_round65b_dualgate_repair.py with the winner candidate.\n'
            ),
            encoding='utf-8',
        )
        formal_result_json.write_text(
            json.dumps(best, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        out['formal_method_file'] = str(formal_method_file)
        out['formal_result_json'] = str(formal_result_json)
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate passed the Round61 clean-win gate under the same dataset; keep Round65-B as probe-only repair evidence.'
        out['result_classification'] = 'partial signal' if any(float(best_delta_r61[k]) < 0.0 for k in PRIMARY_REPAIR_KEYS) else 'no useful signal'
        out['conclusion_line'] = 'Round65-B did not produce a useful repair signal or a clean promotable winner over Round61.'

    out['strongest_signal'] = {
        'name': best_name,
        'signal': (
            f"best (still regressive) candidate {best_name}: "
            f"dKg_yy Δ={best_delta_r61['dKg_yy']:.6f}, "
            f"dKa_xx Δ={best_delta_r61['dKa_xx']:.6f}, "
            f"rx_y Δ={best_delta_r61['rx_y']:.6f}, "
            f"dKg_xx Δ={best_delta_r61['dKg_xx']:.6f}, mean Δ={best_delta_r61['mean_pct_error']:.6f}"
        ),
        'regressions': str(best_penalties),
    }

    out['mechanism_learning'] = {
        'worked': 'Dual-channel split gate can decouple feedback and SCD adaptation, and provides direct per-channel diagnostics (feedback gate vs SCD gate logs).',
        'not_worked': 'Even with split channels, finding a configuration that simultaneously repairs yy/Ka_xx/rx_y and improves mainline mean/max/dKg_xx over Round61 remains difficult.',
        'structural_or_redistribution': 'Current evidence should be treated as repair-signal exploration; only clean no-regression improvement qualifies as structural mainline gain.',
    }
    out['next_best_move'] = (
        'Lock feedback gate even closer to Round61 on iter2 protected states (yy/Ka_xx), '
        'and run a tighter xx/zz-only SCD alpha ladder (one notch around best candidate) without changing any other knob.'
    )
    out['next_actions'] = {
        'keep': 'Dual-channel innovation-consistency mechanism and per-channel logs.',
        'remove': 'Shared scalar gate coupling that simultaneously drifts feedback and SCD response.',
        'repair_direction': out['next_best_move'],
        'new_mechanism_direction': 'If repair saturates, try dual-target innovation channels with different target_nis for feedback vs SCD.',
    }

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND65B_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND65B_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND65B_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
