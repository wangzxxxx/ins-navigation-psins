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
from probe_round59_h_scd_hybrid import HYBRID_CANDIDATES, _merge_hybrid_candidate, _run_internalized_hybrid_scd

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round59 SCD-hybrid scaleblock once-per-phase commit'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round59_scd_hybrid_scaleblock_once'
CANDIDATE_NAME = 'scd_scale_once_a0999_biaslink_commit'


def _pick_candidate():
    for candidate in HYBRID_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_hybrid_candidate(candidate)
    method_mod = load_module('markov_method_r59_scd_hybrid_scaleblock_once_base', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    source_mod = load_module('markov_pruned_source_for_r59_scd_hybrid_scaleblock_once', str(SOURCE_FILE))
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
        label='42-GM1-R59-SCD-HYBRID-SCALEBLOCK-ONCE',
        scd_cfg=merged_candidate['scd'],
    ))
    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round59_selected_candidate': candidate['name'],
            'round59_candidate_description': candidate['description'],
            'round59_candidate_rationale': candidate['rationale'],
            'round59_base_round58_candidate': 'llm_alpha12_plus_lever_plus',
            'round59_hybrid': True,
            'round59_scd': merged_candidate['scd'],
            'round59_selection_rationale': 'Selected from the Round59-H narrow hybrid probes because it improved dKg_xx, dKg_xy, dKg_yy, dKa_xx, mean, and max versus pure Round58 while only giving back a small amount on dKg_zz and ry_z, and it preserved the core R58 lever/rx_y guard.',
            'policy': 'Round59 keeps the entire Round58 internalized alpha12+lever-plus feedback stack intact and adds only a once-per-static-phase SCD-style cross-covariance suppression in iter2_commit: alpha=0.999, transition_duration=2.0s, target=dKg/dKa scale block (12:27 only), nav<->target plus bias<->target coupling decay, while leaving Ka2/lever/markov states outside the decay target.',
        })
        result[4] = extra
    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
