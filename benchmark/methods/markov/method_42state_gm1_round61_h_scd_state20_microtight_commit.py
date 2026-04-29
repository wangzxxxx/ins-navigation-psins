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
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round61 conservative micro-tighten on Round60 hybrid commit'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round61_h_scd_state20_microtight_commit'
CANDIDATE_NAME = 'r61_s20_08988_ryz00116'


def _pick_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_round61_candidate(candidate)

    method_mod = load_module('markov_method_r61_h_scd_state20_microtight_base', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    source_mod = load_module('markov_pruned_source_for_r61_h_formal', str(SOURCE_FILE))
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
        label='42-GM1-R61H-SCD-S20-MICROTIGHT',
        scd_cfg=merged_candidate['scd'],
    ))

    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round61_selected_candidate': candidate['name'],
            'round61_candidate_description': candidate['description'],
            'round61_candidate_rationale': candidate['rationale'],
            'round61_base_round60_candidate': 'scale_once_a0999_s20tight_0899_ryz00116',
            'round61_base_round59_candidate': 'scd_scale_once_a0999_biaslink_commit',
            'round61_policy_patch': merged_candidate.get('iter_patches', {}),
            'round61_scd': merged_candidate.get('scd', {}),
            'round61_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round61_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round61_selection_rationale': 'Selected from the Round61 micro probes because it is the only strict no-regression candidate on the protected Round60 guard set (dKg_xx/dKg_xy/dKg_yy/dKa_xx/rx_y/max all unchanged) while still improving dKg_zz and overall mean.',
            'policy': 'Round61 keeps the full Round60 hybrid route intact (iter2-only one-shot scale-block SCD, alpha=0.999, bias-linked, post ry_z confirmation 1.00116) and makes only one extra micro change: iter2 dKg_zz state20 alpha 0.899 -> 0.8988.',
        })
        result[4] = extra

    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
