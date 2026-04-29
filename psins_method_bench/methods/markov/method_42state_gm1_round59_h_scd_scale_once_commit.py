from __future__ import annotations

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
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import emit_result, load_module, summarize_result
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import (
    HYBRID_CANDIDATES,
    _merge_hybrid_candidate,
    _run_internalized_hybrid_scd,
)

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round59-h hybrid SCD scale-block once-per-phase commit'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round59_h_scd_scale_once_commit'
CANDIDATE_NAME = 'scd_scale_once_a0999_biaslink_commit'


def _pick_candidate():
    for candidate in HYBRID_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_hybrid_candidate(candidate)

    method_mod = load_module('markov_method_r59_h_scd_scale_base', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    source_mod = load_module('markov_pruned_source_for_r59_h_formal', str(SOURCE_FILE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = method_mod._build_dataset(source_mod)

    result = list(_run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label='42-GM1-R59H-SCD-SCALE-COMMIT',
        scd_cfg=merged_candidate['scd'],
    ))

    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round59_h_selected_candidate': candidate['name'],
            'round59_h_candidate_description': candidate['description'],
            'round59_h_candidate_rationale': candidate['rationale'],
            'round59_h_base_round58_candidate': 'llm_alpha12_plus_lever_plus',
            'round59_h_policy_patch': merged_candidate.get('iter_patches', {}),
            'round59_h_scd': merged_candidate.get('scd', {}),
            'round59_h_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round59_h_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round59_h_selection_rationale': 'Selected from the Round59-H hybrid probes because it was the only narrow SCD-style add-on that improved dKg_xx/max and mean while still further improving dKg_xy, dKg_yy, dKa_xx, and rx_y; it only gave back a small amount on dKg_zz and ry_z.',
            'policy': 'Round59-H keeps the full Round58 internalized alpha/lever stack unchanged and adds only a one-shot, after-transition, iter2-only SCD-style cross-covariance suppression on the dKg/dKa block (excluding Ka2/lever), with alpha=0.999 and nav/bias-to-scale coupling decay but no full SCD rewrite.',
        })
        result[4] = extra

    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
