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
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
OUTPUT_JSON = RESULTS_DIR / 'round62_alpha_guard_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round62_alpha_guard_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round62_alpha_guard_probe_2026-03-27.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate

BASE_R61_CANDIDATE_NAME = 'r61_s20_08988_ryz00116'
PROBE_SCALES = [1.0, 0.10, 0.08, 0.05, 0.03]
TARGET_KEYS = ['dKg_yy', 'dKa_xx', 'rx_y']
PROTECT_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_zz', 'mean_pct_error']

ROUND62_CANDIDATES = [
    {
        'name': 'r62_ultralow_guard_soft_neutral',
        'description': 'Ultra-low-only neutralizer: leave 1x/0.10x/0.08x untouched, and below 0.08x smoothly pull the iter2 dKg_yy / dKa_xx alphas back toward 1.0 while fading out the tiny rx_y post-confirmation.',
        'rationale': 'Smallest regime-aware fix consistent with the latest analysis: the main Round61 gains come from xx/xy/zz, so only yy / Ka_xx / rx_y should be softened when the shared-noise regime is very small.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.0, 21: 1.0},
            'rx_y_post_pull': 1.0,
        },
    },
    {
        'name': 'r62_ultralow_guard_yy_heavier',
        'description': 'Same ultra-low regime gate, but pull dKg_yy a little harder than dKa_xx and let rx_y confirmation fade slightly past neutral at the floor.',
        'rationale': 'The 0.03x failure is dominated by dKg_yy, so this candidate checks whether a slightly stronger yy-only anti-overshoot is enough without touching the Round61 xx/xy/zz core.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.35, 21: 1.10},
            'rx_y_post_pull': 1.10,
        },
    },
    {
        'name': 'r62_ultralow_guard_late_onset',
        'description': 'Later-onset ultra-low gate: stay fully inert through 0.08x, intervene mainly at 0.05x/0.03x, and keep the sweet spot almost exactly Round61-shaped.',
        'rationale': 'If 0.05x still benefits from the Round61 aggressiveness, the safer move is to localize the repair even more tightly around the deepest ultra-low regime.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.07,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.30, 21: 1.05},
            'rx_y_post_pull': 1.0,
        },
    },
]


def _load_r61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == BASE_R61_CANDIDATE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(BASE_R61_CANDIDATE_NAME)


BASE_ARW = 0.005
BASE_VRW = 5.0
BASE_BI_G = 0.002
BASE_BI_A = 5.0
TAU_G = 300.0
TAU_A = 300.0


def make_suffix(noise_scale: float) -> str:
    mapping = {
        1.0: 'noise1x',
        1.0 / 3.0: 'noise1over3',
        0.2: 'noise1over5',
        0.5: 'noise0p5',
        2.0: 'noise2p0',
    }
    for key, value in mapping.items():
        if abs(noise_scale - key) < 1e-12:
            return value
    return f"noise{str(noise_scale).replace('.', 'p')}"


def _baseline_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'M_markov_42state_gm1_shared_{make_suffix(noise_scale)}_param_errors.json'


def _round61_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_{make_suffix(noise_scale)}_param_errors.json'


def _compare_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'compare_baseline_vs_round61_shared_{make_suffix(noise_scale)}.json'


def build_shared_dataset(mod, noise_scale: float):
    ts = 0.01
    att0 = mod.np.array([1.0, -91.0, -91.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70], [2, 0, 1, 0, 90, 9, 20, 20], [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20], [5, 0, 1, 0, -90, 9, 20, 20], [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20], [8, 1, 0, 0, 90, 9, 20, 20], [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20], [11, -1, 0, 0, 90, 9, 20, 20], [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20], [14, 0, 0, 1, 90, 9, 20, 20], [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20], [17, 0, 0, -1, 90, 9, 20, 20], [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = BASE_ARW * noise_scale * mod.glv.dpsh
    vrw = BASE_VRW * noise_scale * mod.glv.ugpsHz
    bi_g = BASE_BI_G * noise_scale * mod.glv.dph
    bi_a = BASE_BI_A * noise_scale * mod.glv.ug
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=arw, vrw=vrw,
        bi_g=bi_g, tau_g=TAU_G,
        bi_a=bi_a, tau_a=TAU_A, seed=42,
    )
    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': TAU_G,
        'tau_a': TAU_A,
        'noise_scale': noise_scale,
        'noise_config': {
            'arw_dpsh': BASE_ARW * noise_scale,
            'vrw_ugpsHz': BASE_VRW * noise_scale,
            'bi_g_dph': BASE_BI_G * noise_scale,
            'bi_a_ug': BASE_BI_A * noise_scale,
            'tau_g': TAU_G,
            'tau_a': TAU_A,
            'seed': 42,
            'base_family': 'round62_shared_probe',
        },
    }


def _merge_round62_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_r61_base_candidate())
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

    merged['round62_guard'] = copy.deepcopy(extra_candidate.get('round62_guard', {}))
    merged['round62_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _guard_strength(noise_scale: float, guard_cfg: dict) -> float:
    start = float(guard_cfg.get('start_noise_scale', 0.08))
    full = float(guard_cfg.get('full_noise_scale', 0.03))
    power = float(guard_cfg.get('power', 1.0))
    if start <= full:
        raise ValueError(f'Invalid guard config: start={start} full={full}')
    if noise_scale >= start:
        return 0.0
    if noise_scale <= full:
        return 1.0
    raw = (start - noise_scale) / (start - full)
    return float(raw ** power)


def _build_round62_method(base_method_mod, merged_candidate: dict, noise_scale: float):
    method_mod = _build_patched_method(base_method_mod, merged_candidate)
    guard_cfg = copy.deepcopy(merged_candidate.get('round62_guard', {}))
    apply_policy_names = set(guard_cfg.get('apply_policy_names', []))
    guard_strength = _guard_strength(noise_scale, guard_cfg) if guard_cfg else 0.0

    def patched_apply_feedback(mod, base_clbt, kf, prior_diag, policy):
        out = method_mod._copy_clbt(mod, base_clbt)
        xk = mod.np.array(kf['xk'], dtype=float).copy()
        xfb = xk.copy()
        trust_log = {}

        alpha_mult = {int(k): float(v) for k, v in policy.get('state_alpha_mult', {}).items()}
        alpha_add = {int(k): float(v) for k, v in policy.get('state_alpha_add', {}).items()}
        alpha_clip = policy.get('state_alpha_clip', (0.70, 1.08))
        low_clip = float(alpha_clip[0])
        high_clip = float(alpha_clip[1])

        active_guard_strength = 0.0
        guard_triggered = (policy['name'] in apply_policy_names) and guard_strength > 0.0
        if guard_triggered:
            active_guard_strength = guard_strength

        for idx in method_mod.SELECTED_SCALE_STATES:
            tc = method_mod._trust_components(mod, kf, prior_diag, idx, policy)
            base_alpha = float(tc['alpha'])
            adj_alpha = base_alpha * alpha_mult.get(idx, 1.0) + alpha_add.get(idx, 0.0)
            if adj_alpha < low_clip:
                adj_alpha = low_clip
            if adj_alpha > high_clip:
                adj_alpha = high_clip

            pull = None
            guarded_alpha = adj_alpha
            if guard_triggered and idx in guard_cfg.get('state_alpha_pull', {}):
                pull = float(guard_cfg['state_alpha_pull'][idx])
                guarded_alpha = 1.0 + (adj_alpha - 1.0) * (1.0 - active_guard_strength * pull)
                if guarded_alpha < low_clip:
                    guarded_alpha = low_clip
                if guarded_alpha > high_clip:
                    guarded_alpha = high_clip

            xfb[idx] = guarded_alpha * xk[idx]
            tc['label'] = method_mod.SELECTED_STATE_LABELS[str(idx)]
            tc['base_alpha'] = base_alpha
            tc['alpha_mult'] = float(alpha_mult.get(idx, 1.0))
            tc['alpha_add'] = float(alpha_add.get(idx, 0.0))
            tc['alpha_pre_guard'] = float(adj_alpha)
            tc['round62_guard_pull'] = pull
            tc['round62_guard_strength'] = float(active_guard_strength)
            tc['alpha'] = float(guarded_alpha)
            tc['x_feedback'] = float(xfb[idx])
            trust_log[str(idx)] = tc

        for idx in method_mod.OTHER_SCALE_STATES:
            xfb[idx] = policy['other_scale_alpha'] * xk[idx]

        dKg = xfb[12:21].reshape(3, 3).T
        out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']

        dKa = mod.Ka_from_upper(xfb[21:27])
        out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']

        out['Ka2'] = out['Ka2'] + policy['ka2_alpha'] * xfb[27:30]
        out['eb'] = out['eb'] + xfb[6:9]
        out['db'] = out['db'] + xfb[9:12]
        out['rx'] = out['rx'] + policy['lever_alpha'] * xfb[30:33]
        out['ry'] = out['ry'] + policy['lever_alpha'] * xfb[33:36]
        out['eb'] = out['eb'] + policy['markov_alpha'] * xfb[36:39]
        out['db'] = out['db'] + policy['markov_alpha'] * xfb[39:42]

        post_rx_y_mult = float(policy.get('post_rx_y_mult', 1.0))
        post_ry_z_mult = float(policy.get('post_ry_z_mult', 1.0))
        rx_pull = None
        if guard_triggered and guard_cfg.get('rx_y_post_pull') is not None:
            rx_pull = float(guard_cfg['rx_y_post_pull'])
            post_rx_y_mult = 1.0 + (post_rx_y_mult - 1.0) * (1.0 - active_guard_strength * rx_pull)
        rx_before = float(out['rx'][1])
        ry_before = float(out['ry'][2])
        out['rx'][1] *= post_rx_y_mult
        out['ry'][2] *= post_ry_z_mult

        return out, {
            'policy_name': policy['name'],
            'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
            'selected_trust': trust_log,
            'other_scale_alpha': float(policy['other_scale_alpha']),
            'ka2_alpha': float(policy['ka2_alpha']),
            'lever_alpha': float(policy['lever_alpha']),
            'markov_alpha': float(policy['markov_alpha']),
            'state_prior_diag_mult': {str(k): float(v) for k, v in policy.get('state_prior_diag_mult', {}).items()},
            'state_q_static_mult': {str(k): float(v) for k, v in policy.get('state_q_static_mult', {}).items()},
            'state_q_dynamic_mult': {str(k): float(v) for k, v in policy.get('state_q_dynamic_mult', {}).items()},
            'state_q_late_mult': {str(k): float(v) for k, v in policy.get('state_q_late_mult', {}).items()},
            'state_alpha_mult': {str(k): float(v) for k, v in alpha_mult.items()},
            'state_alpha_add': {str(k): float(v) for k, v in alpha_add.items()},
            'post_rx_y_mult': float(post_rx_y_mult),
            'post_ry_z_mult': float(post_ry_z_mult),
            'post_rx_y_before': rx_before,
            'post_rx_y_after': float(out['rx'][1]),
            'post_ry_z_before': ry_before,
            'post_ry_z_after': float(out['ry'][2]),
            'round62_noise_scale': float(noise_scale),
            'round62_guard': copy.deepcopy(guard_cfg),
            'round62_guard_triggered': bool(guard_triggered),
            'round62_guard_strength': float(active_guard_strength),
            'round62_rx_y_post_pull': rx_pull,
        }

    method_mod._apply_trust_internalized_feedback = patched_apply_feedback
    method_mod.ROUND62_GUARD = copy.deepcopy(guard_cfg)
    method_mod.ROUND62_GUARD_STRENGTH = float(guard_strength)
    method_mod.ROUND62_NOISE_SCALE = float(noise_scale)
    method_mod.METHOD = f"42-state GM1 round62 alpha-guard probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round62_alpha_guard_{merged_candidate['name']}"
    return method_mod


def _compute_payload(source_mod, clbt, params, variant: str, method_file: str, extra=None):
    param_errors = {}
    pct_values = []
    for name, true_v, get_est in params:
        true_f = float(true_v)
        est_f = float(get_est(clbt))
        abs_err = abs(true_f - est_f)
        pct_err = abs_err / abs(true_f) * 100.0 if abs(true_f) > 1e-15 else 0.0
        param_errors[name] = {
            'true': true_f,
            'est': est_f,
            'abs_error': abs_err,
            'pct_error': pct_err,
        }
        pct_values.append(pct_err)

    pct_arr = source_mod.np.asarray(pct_values, dtype=float)
    focus_scale_pct = {
        'dKg_xx': param_errors['dKg_xx']['pct_error'],
        'dKg_xy': param_errors['dKg_xy']['pct_error'],
        'dKg_yy': param_errors['dKg_yy']['pct_error'],
        'dKg_zz': param_errors['dKg_zz']['pct_error'],
        'dKa_xx': param_errors['dKa_xx']['pct_error'],
    }
    lever_guard_pct = {
        'rx_y': param_errors['rx_y']['pct_error'],
        'ry_z': param_errors['ry_z']['pct_error'],
    }
    overall = {
        'mean_pct_error': float(source_mod.np.mean(pct_arr)),
        'median_pct_error': float(source_mod.np.median(pct_arr)),
        'max_pct_error': float(source_mod.np.max(pct_arr)),
    }
    return {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': [name for name, _, _ in params],
        'param_errors': param_errors,
        'focus_scale_pct': focus_scale_pct,
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
        'extra': extra or {},
    }


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


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


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


def _run_candidate_at_scale(candidate: dict, noise_scale: float, idx: int):
    suffix = make_suffix(noise_scale)
    source_mod = load_module(f'markov_pruned_source_r62_probe_{candidate["name"]}_{suffix}', str(SOURCE_FILE))
    compute_r61_mod = load_module(f'compute_r61_for_r62_probe_{suffix}_{idx}', str(COMPUTE_R61_FILE))
    params = compute_r61_mod._param_specs(source_mod)

    dataset = build_shared_dataset(source_mod, noise_scale)
    merged_candidate = _merge_round62_candidate(candidate)
    base_method_mod = load_module(f'markov_method_r62_probe_base_{candidate["name"]}_{suffix}', str(R53_METHOD_FILE))
    method_mod = _build_round62_method(base_method_mod, merged_candidate, noise_scale)
    result = list(_run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'42-GM1-R62-{idx}-{suffix.upper()}',
        scd_cfg=merged_candidate['scd'],
    ))
    extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra = dict(extra)
    extra.update({
        'noise_scale': float(noise_scale),
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'round62_selected_candidate': merged_candidate['name'],
        'round62_base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'round62_guard': copy.deepcopy(merged_candidate.get('round62_guard', {})),
        'round62_guard_strength': float(getattr(method_mod, 'ROUND62_GUARD_STRENGTH', 0.0)),
        'round62_note': 'Round62 keeps the full Round61 route and adds only an ultra-low-noise alpha/post-confirmation guard on dKg_yy / dKa_xx / rx_y.',
    })
    payload = _compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_round62_alpha_guard_{merged_candidate["name"]}_shared_{suffix}',
        method_file=f'probe_round62_alpha_guard::{merged_candidate["name"]}',
        extra=extra,
    )
    return merged_candidate, payload


def _score_candidate(candidate_scales: dict, r61_scales: dict):
    # Higher is better. Only ultra-low gains count positively; any 1x/0.10x/0.08x regression is penalized hard.
    score = 0.0
    penalties = []
    for scale_key in ['noise1x', 'noise0p1', 'noise0p08']:
        cand = candidate_scales[scale_key]
        ref = r61_scales[scale_key]
        protect_deltas = {
            'dKg_xx': cand['focus']['dKg_xx'] - ref['focus_scale_pct']['dKg_xx'],
            'dKg_xy': cand['focus']['dKg_xy'] - ref['focus_scale_pct']['dKg_xy'],
            'dKg_zz': cand['focus']['dKg_zz'] - ref['focus_scale_pct']['dKg_zz'],
            'mean_pct_error': cand['overall']['mean_pct_error'] - ref['overall']['mean_pct_error'],
        }
        for key, value in protect_deltas.items():
            if value > 1e-9:
                penalties.append({'scale': scale_key, 'metric': key, 'delta': float(value)})
                score -= 1000.0 * float(value)

    weights = {
        'noise0p05': {'dKg_yy': 1.6, 'dKa_xx': 1.0, 'rx_y': 1.0, 'mean_pct_error': 0.8, 'max_pct_error': 0.2},
        'noise0p03': {'dKg_yy': 2.6, 'dKa_xx': 1.6, 'rx_y': 1.4, 'mean_pct_error': 1.0, 'max_pct_error': 0.3},
    }
    for scale_key, scale_weights in weights.items():
        cand = candidate_scales[scale_key]
        ref = r61_scales[scale_key]
        score += scale_weights['dKg_yy'] * (ref['focus_scale_pct']['dKg_yy'] - cand['focus']['dKg_yy'])
        score += scale_weights['dKa_xx'] * (ref['focus_scale_pct']['dKa_xx'] - cand['focus']['dKa_xx'])
        score += scale_weights['rx_y'] * (ref['lever_guard_pct']['rx_y'] - cand['lever']['rx_y'])
        score += scale_weights['mean_pct_error'] * (ref['overall']['mean_pct_error'] - cand['overall']['mean_pct_error'])
        score += scale_weights['max_pct_error'] * (ref['overall']['max_pct_error'] - cand['overall']['max_pct_error'])
    return float(score), penalties


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round62 首批 probe：在 Round61 完整主干上，只新增 ultra-low-noise alpha/post-confirmation guard，目标是 **不动 1x / 0.08x / 0.10x 主收益**，只修 0.03x~0.05x 的 dKg_yy / dKa_xx / rx_y 过拟合。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Probe 设置')
    lines.append('')
    lines.append(f'- Round61 base candidate: `{summary["base_round61_candidate"]}`')
    lines.append('- Shared-noise probe scales: `1.00x, 0.10x, 0.08x, 0.05x, 0.03x`')
    lines.append('- Probe family: **very small deterministic ultra-low-noise alpha-guard**')
    lines.append('')
    lines.append('## 2. 候选摘要')
    lines.append('')
    lines.append('| candidate | 0.03x dKg_yy ΔvsR61 | 0.03x dKa_xx ΔvsR61 | 0.03x rx_y ΔvsR61 | 0.03x mean ΔvsR61 | 0.05x mean ΔvsR61 | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        s03 = cand['scales']['noise0p03']
        s05 = cand['scales']['noise0p05']
        lines.append(
            f"| `{name}` | {s03['delta_vs_round61']['dKg_yy']:.4f} | {s03['delta_vs_round61']['dKa_xx']:.4f} | {s03['delta_vs_round61']['rx_y']:.4f} | {s03['delta_vs_round61']['mean_pct_error']:.4f} | {s05['delta_vs_round61']['mean_pct_error']:.4f} | {cand['selection']['score']:.4f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Winner / status')
    lines.append('')
    winner = summary['winner']
    if winner:
        lines.append(f'- Early winner: `{winner["name"]}`')
        lines.append(f'- Reason: {winner["reason"]}')
    else:
        lines.append('- Early winner: **none**')
        lines.append(f'- Reason: {summary["no_winner_reason"]}')
    lines.append('')
    lines.append('## 4. Candidate notes')
    lines.append('')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        lines.append(f'### `{name}`')
        lines.append(f'- Description: {cand["description"]}')
        lines.append(f'- Rationale: {cand["rationale"]}')
        lines.append(f'- Selection note: {cand["selection"]["note"]}')
        lines.append(f'- Guard config: `{json.dumps(cand["round62_guard"], ensure_ascii=False)}`')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_dump = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'probe_scales': PROBE_SCALES,
        'round62_candidates': ROUND62_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    baseline_payloads = {}
    round61_payloads = {}
    round61_compare = {}
    for scale in PROBE_SCALES:
        suffix = make_suffix(scale)
        baseline_payloads[suffix] = json.loads(_baseline_payload_path(scale).read_text(encoding='utf-8'))
        round61_payloads[suffix] = json.loads(_round61_payload_path(scale).read_text(encoding='utf-8'))
        round61_compare[suffix] = json.loads(_compare_payload_path(scale).read_text(encoding='utf-8'))

    out = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'probe_scales': PROBE_SCALES,
        'candidate_order': [c['name'] for c in ROUND62_CANDIDATES],
        'baseline_paths': {make_suffix(s): str(_baseline_payload_path(s)) for s in PROBE_SCALES},
        'round61_paths': {make_suffix(s): str(_round61_payload_path(s)) for s in PROBE_SCALES},
        'candidate_json': str(CANDIDATE_JSON),
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
    }

    for idx, candidate in enumerate(ROUND62_CANDIDATES, start=1):
        merged_candidate = _merge_round62_candidate(candidate)
        candidate_scales = {}
        for scale in PROBE_SCALES:
            suffix = make_suffix(scale)
            merged_candidate, payload = _run_candidate_at_scale(candidate, scale, idx)
            baseline_payload = baseline_payloads[suffix]
            round61_payload = round61_payloads[suffix]
            compare_payload = round61_compare[suffix]
            focus = payload['focus_scale_pct']
            lever = payload['lever_guard_pct']
            overall = payload['overall']
            delta_vs_round61 = {
                'dKg_xx': float(focus['dKg_xx'] - round61_payload['focus_scale_pct']['dKg_xx']),
                'dKg_xy': float(focus['dKg_xy'] - round61_payload['focus_scale_pct']['dKg_xy']),
                'dKg_yy': float(focus['dKg_yy'] - round61_payload['focus_scale_pct']['dKg_yy']),
                'dKg_zz': float(focus['dKg_zz'] - round61_payload['focus_scale_pct']['dKg_zz']),
                'dKa_xx': float(focus['dKa_xx'] - round61_payload['focus_scale_pct']['dKa_xx']),
                'rx_y': float(lever['rx_y'] - round61_payload['lever_guard_pct']['rx_y']),
                'ry_z': float(lever['ry_z'] - round61_payload['lever_guard_pct']['ry_z']),
                'mean_pct_error': float(overall['mean_pct_error'] - round61_payload['overall']['mean_pct_error']),
                'median_pct_error': float(overall['median_pct_error'] - round61_payload['overall']['median_pct_error']),
                'max_pct_error': float(overall['max_pct_error'] - round61_payload['overall']['max_pct_error']),
            }
            candidate_vs_baseline = _relative_improvement_block(
                baseline_payload,
                payload,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            )
            candidate_scales[suffix] = {
                'noise_scale': float(scale),
                'focus': focus,
                'lever': lever,
                'overall': overall,
                'delta_vs_round61': delta_vs_round61,
                'candidate_vs_baseline': candidate_vs_baseline,
                'round61_vs_baseline_reference': {
                    'dKg_xx': compare_payload['all_params']['dKg_xx'],
                    'dKg_xy': compare_payload['all_params']['dKg_xy'],
                    'dKg_yy': compare_payload['all_params']['dKg_yy'],
                    'dKg_zz': compare_payload['all_params']['dKg_zz'],
                    'dKa_xx': compare_payload['all_params']['dKa_xx'],
                    'rx_y': compare_payload['all_params']['rx_y'],
                    'ry_z': compare_payload['all_params']['ry_z'],
                    'mean_pct_error': compare_payload['overall']['mean_pct_error'],
                    'median_pct_error': compare_payload['overall']['median_pct_error'],
                    'max_pct_error': compare_payload['overall']['max_pct_error'],
                },
                'extra': {
                    'guard_strength': payload['extra'].get('round62_guard_strength'),
                    'feedback_log': payload['extra'].get('feedback_log'),
                    'scd_log': payload['extra'].get('scd_log'),
                    'schedule_log': payload['extra'].get('schedule_log'),
                },
            }

        score, penalties = _score_candidate(candidate_scales, round61_payloads)
        note = 'Ultra-low improvement present without protected-scale regressions.'
        if penalties:
            note = f'Has protected-scale regression penalties: {penalties}'
        if not penalties:
            d03 = candidate_scales['noise0p03']['delta_vs_round61']
            d05 = candidate_scales['noise0p05']['delta_vs_round61']
            if d03['dKg_yy'] < 0 and d03['rx_y'] < 0 and d03['mean_pct_error'] <= 0 and d05['mean_pct_error'] <= 0:
                note = 'Clean ultra-low win signal at 0.03x/0.05x while leaving 1x/0.10x/0.08x intact.'
            elif d03['dKg_yy'] < 0 and d03['rx_y'] < 0:
                note = 'Improves the main ultra-low pain points, but overall mean/max still needs judgment.'
            else:
                note = 'Mixed ultra-low result: guard is too weak or improves only part of the target set.'

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round62_guard': copy.deepcopy(candidate.get('round62_guard', {})),
            'round62_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'scales': candidate_scales,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
        }

        print(candidate['name'], json.dumps({
            'score': score,
            'note': note,
            'noise0p03': candidate_scales['noise0p03']['delta_vs_round61'],
            'noise0p05': candidate_scales['noise0p05']['delta_vs_round61'],
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best_candidate = out['candidates'][best_name]
    best_penalties = best_candidate['selection']['penalties']
    best_d03 = best_candidate['scales']['noise0p03']['delta_vs_round61']
    best_d05 = best_candidate['scales']['noise0p05']['delta_vs_round61']

    if (not best_penalties) and best_score > 0.01 and best_d03['dKg_yy'] < 0 and best_d03['rx_y'] < 0 and best_d05['mean_pct_error'] <= 0:
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Best weighted ultra-low repair score with no protected-scale regression penalty; preserves 1x/0.10x/0.08x and improves the 0.03x pain set.',
        }
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate achieved a clean ultra-low repair win under the protected-scale no-regression rule; inspect summary JSON before formalizing Round62.'

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'report_md': str(REPORT_MD),
        'winner': out['winner'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
