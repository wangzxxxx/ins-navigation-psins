from __future__ import annotations

import copy
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
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R58_JSON = RESULTS_DIR / 'R58_42state_gm1_round58_llm_guided_alpha12_lever_plus_param_errors.json'
SCD_JSON = RESULTS_DIR / 'S_scd_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round59_h_scd_hybrid_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round59_h_scd_hybrid_candidates.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round58_llm_guided import ROUND58_LLM_CANDIDATES, _merge_round58_candidate


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']
ROUND58_BASE_NAME = 'llm_alpha12_plus_lever_plus'


HYBRID_CANDIDATES = [
    {
        'name': 'scd_sel_once_a0999_biaslink_commit',
        'description': 'R58 + once-per-static-phase gentle SCD on the five protected/targeted selected scale states during iter2 only.',
        'rationale': 'Safest first hybrid: only trim nav/bias coupling into the exact R58-selected states, once per stop, with optimal-style mild alpha.',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'scd_sel_once_a0998_biaslink_commit',
        'description': 'Same as candidate A, but a notch stronger alpha.',
        'rationale': 'Tests whether R58 still has a little room for dKg_xx/max repair if the selected-state coupling cut is made slightly stronger.',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.998,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'scd_sel_once_a0999_navonly_commit',
        'description': 'Same gentle selected-state decay, but preserve bias↔target coupling and only trim nav↔target leakage.',
        'rationale': 'Checks whether the hybrid is safer if it cuts only the ZUPT/nav noise pipeline while leaving bias observability routes untouched.',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': False,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'scd_scale_once_a0999_biaslink_commit',
        'description': 'Expand the once-per-phase cut from selected states to the full dKg/dKa block, still keeping Ka2/lever outside the decay target.',
        'rationale': 'Tests whether the hybrid helps more if the SCD idea covers the whole scale/misalignment block while still protecting the lever terms R58 already holds well.',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'scd_sel_repeat_a09998_biaslink_commit',
        'description': 'Ultra-gentle repeated post-transition decay on selected states during iter2 only.',
        'rationale': 'Checks the other transition style: instead of one cut per stop, apply a nearly no-op repeated suppression after the grace period.',
        'scd': {
            'mode': 'repeat_after_transition',
            'alpha': 0.9998,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
]


def _load_round58_base_candidate():
    for candidate in ROUND58_LLM_CANDIDATES:
        if candidate['name'] == ROUND58_BASE_NAME:
            return _merge_round58_candidate(candidate)
    raise KeyError(ROUND58_BASE_NAME)


def _merge_hybrid_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round58_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['scd'] = copy.deepcopy(extra_candidate['scd'])
    return merged


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


def _resolve_target_indices(method_mod, target_name: str):
    if target_name == 'selected':
        return [int(idx) for idx in method_mod.SELECTED_SCALE_STATES]
    if target_name == 'scale_block':
        return list(range(12, 27))
    raise KeyError(f'Unknown SCD target: {target_name}')


def _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices):
    alpha = float(scd_cfg['alpha'])
    P = kf['Pxk']

    P[0:6, target_indices] *= alpha
    P[target_indices, 0:6] *= alpha

    if scd_cfg.get('bias_to_target', True):
        P[6:12, target_indices] *= alpha
        P[target_indices, 6:12] *= alpha


# Reuses the Round53/R58-internalized loop, but adds a very mild SCD-style
# cross-covariance suppression after static ZUPT updates.
def _run_internalized_hybrid_scd(method_mod, mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label, scd_cfg):
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

        if not policy.get('readout_only'):
            clbt, meta = method_mod._apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, policy)
            feedback_log.append(meta)

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
            'alpha': float(scd_cfg['alpha']),
            'transition_duration': float(scd_cfg['transition_duration']),
            'target': scd_cfg['target'],
            'target_indices': [int(x) for x in target_indices],
            'bias_to_target': bool(scd_cfg.get('bias_to_target', True)),
            'n_transition_eligible': int(n_transition_eligible),
            'n_scd': int(n_scd),
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
        'iteration_policies': method_mod.ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'scd_log': scd_log,
        'policy': 'Round59-H probe keeps the full Round58 internalized feedback stack intact and adds only a very mild post-ZUPT SCD-style cross-covariance suppression in iter2 under tightly bounded alpha/scope rules.',
    }


def _run_candidate(candidate: dict, idx: int):
    merged_candidate = _merge_hybrid_candidate(candidate)
    method_mod = load_module(f'markov_method_round59_h_probe_{idx}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    method_mod.METHOD = f"42-state GM1 round59-h probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round59_h_probe_{merged_candidate['name']}"

    source_mod = load_module(f'markov_pruned_source_for_round59_h_probe_{idx}', str(SOURCE_FILE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = method_mod._build_dataset(source_mod)
    result = _run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label=f'42-GM1-R59H-{idx}',
        scd_cfg=merged_candidate['scd'],
    )
    clbt = result[0]
    extra = result[4] if len(result) >= 5 else {}
    _, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return merged_candidate, result, focus, lever, overall, extra


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r58_payload = json.loads(R58_JSON.read_text(encoding='utf-8'))
    scd_payload = json.loads(SCD_JSON.read_text(encoding='utf-8')) if SCD_JSON.exists() else None

    candidate_dump = {
        'baseline': ROUND58_BASE_NAME,
        'hybrid_candidates': HYBRID_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'baseline_r58': {
            'focus': r58_payload['focus_scale_pct'],
            'lever': r58_payload['lever_guard_pct'],
            'overall': r58_payload['overall'],
        },
        'baseline_name': ROUND58_BASE_NAME,
        'pure_scd_reference': {
            'summary': scd_payload.get('summary') if scd_payload else None,
            'method': scd_payload.get('method') if scd_payload else None,
            'note': 'S_scd is kept only as a directional reference; its state definition / scaling is not fully apples-to-apples with the 42-state internalized R58/R59-H bench.',
        },
        'candidates': {},
    }

    for idx, candidate in enumerate(HYBRID_CANDIDATES, start=1):
        merged_candidate, result, focus, lever, overall, extra = _run_candidate(candidate, idx)
        probe_info = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round58_candidate': ROUND58_BASE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r58': {
                **_delta_block(focus, r58_payload['focus_scale_pct']),
                **_delta_block(lever, r58_payload['lever_guard_pct']),
                **_delta_block(overall, r58_payload['overall']),
            },
            'key_round59h_delta': {
                'repair_dKg_xx': float(focus['dKg_xx'] - r58_payload['focus_scale_pct']['dKg_xx']),
                'protect_dKg_xy': float(focus['dKg_xy'] - r58_payload['focus_scale_pct']['dKg_xy']),
                'protect_dKg_yy': float(focus['dKg_yy'] - r58_payload['focus_scale_pct']['dKg_yy']),
                'protect_dKg_zz': float(focus['dKg_zz'] - r58_payload['focus_scale_pct']['dKg_zz']),
                'protect_dKa_xx': float(focus['dKa_xx'] - r58_payload['focus_scale_pct']['dKa_xx']),
                'protect_rx_y': float(lever['rx_y'] - r58_payload['lever_guard_pct']['rx_y']),
                'protect_ry_z': float(lever['ry_z'] - r58_payload['lever_guard_pct']['ry_z']),
                'repair_mean': float(overall['mean_pct_error'] - r58_payload['overall']['mean_pct_error']),
                'repair_max': float(overall['max_pct_error'] - r58_payload['overall']['max_pct_error']),
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
                'scd_log': extra.get('scd_log'),
            },
        }
        out['candidates'][merged_candidate['name']] = probe_info
        print(merged_candidate['name'], json.dumps({
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r58': probe_info['delta_vs_r58'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
