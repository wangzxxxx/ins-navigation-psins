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
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round62_alpha_guard import ROUND62_CANDIDATES, _build_round62_method, _merge_round62_candidate

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round62 ultra-low alpha-guard commit on Round61 hybrid route'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round62_ultralow_alpha_guard_commit'
CANDIDATE_NAME = 'r62_ultralow_guard_soft_neutral'


def _pick_candidate():
    for candidate in ROUND62_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_round62_candidate(candidate)

    base_method_mod = load_module('markov_method_r62_ultralow_guard_base', str(R53_METHOD_FILE))
    source_mod = load_module('markov_pruned_source_for_r62_ultralow_guard_formal', str(SOURCE_FILE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = base_method_mod._build_dataset(source_mod)
    method_mod = _build_round62_method(base_method_mod, merged_candidate, noise_scale=1.0)

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
        label='42-GM1-R62-ULTRALOW-GUARD-COMMIT',
        scd_cfg=merged_candidate['scd'],
    ))

    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round62_selected_candidate': candidate['name'],
            'round62_candidate_description': candidate['description'],
            'round62_candidate_rationale': candidate['rationale'],
            'round62_base_round61_candidate': 'r61_s20_08988_ryz00116',
            'round62_base_round60_candidate': 'scale_once_a0999_s20tight_0899_ryz00116',
            'round62_policy_patch': merged_candidate.get('iter_patches', {}),
            'round62_scd': merged_candidate.get('scd', {}),
            'round62_guard': merged_candidate.get('round62_guard', {}),
            'round62_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round62_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round62_selection_rationale': 'Selected from the Round62 constrained probe batch because it preserved the full Round61 1x / 0.10x / 0.08x profile exactly, improved the 0.03x ultra-low dKg_yy and rx_y pain points the most among the tested candidates, and also reduced 0.03x / 0.05x overall mean without any protected-scale regression penalty.',
            'policy': 'Round62 keeps the full Round61 route unchanged for normal/shared 1x and sweet-spot noise regimes, and only below the ultra-low-noise gate pulls iter2 dKg_yy / dKa_xx feedback toward neutral alpha while fading the tiny rx_y post-confirmation toward 1.0.',
            'round62_note': 'On the default 1x formal dataset this method is numerically identical to Round61 by design; the added benefit appears only in the ultra-low shared-noise regime (0.03x~0.05x).',
        })
        result[4] = extra

    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
