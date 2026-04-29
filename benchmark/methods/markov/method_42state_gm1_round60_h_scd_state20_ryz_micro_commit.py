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
from probe_round60_conservative import ROUND60_CONSERVATIVE_CANDIDATES, _merge_round60_candidate

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round60 conservative state20/ry_z micro-repair on Round59-H SCD commit'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round60_h_scd_state20_ryz_micro_commit'
CANDIDATE_NAME = 'scale_once_a0999_s20tight_0899_ryz00116'


def _pick_candidate():
    for candidate in ROUND60_CONSERVATIVE_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_round60_candidate(candidate)

    method_mod = load_module('markov_method_r60_h_scd_state20_ryz_micro_base', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    source_mod = load_module('markov_pruned_source_for_r60_h_formal', str(SOURCE_FILE))
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
        label='42-GM1-R60H-SCD-S20-RYZ-MICRO',
        scd_cfg=merged_candidate['scd'],
    ))

    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round60_selected_candidate': candidate['name'],
            'round60_candidate_description': candidate['description'],
            'round60_candidate_rationale': candidate['rationale'],
            'round60_base_round59_candidate': 'scd_scale_once_a0999_biaslink_commit',
            'round60_policy_patch': merged_candidate.get('iter_patches', {}),
            'round60_scd': merged_candidate.get('scd', {}),
            'round60_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round60_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round60_selection_rationale': 'Selected from the conservative Round60 probes because it kept the exact Round59-H one-shot scale-block SCD route, improved dKg_zz and ry_z together, slightly reduced mean, and held median/max in a numerical tie with Round59-H.',
            'policy': 'Round60 preserves the full Round59-H iter2-only scale-block once-per-phase SCD (alpha=0.999, bias-linked) and adds only a micro iter2 dKg_zz-state tighten (state20 alpha 0.900 -> 0.899) plus an ultra-small post ry_z confirmation (1.00116).',
        })
        result[4] = extra

    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
