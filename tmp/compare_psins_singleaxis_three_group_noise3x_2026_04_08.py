#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
TMP_DIR = WORKSPACE / 'tmp'
BASE_COMPARE_SCRIPT = TMP_DIR / 'compare_psins_singleaxis_three_group_2026_04_08.py'
HALFTURN_A0900_SCRIPT = TMP_DIR / 'probe_singleaxis_llm_scd_halfturn_candidate_a0900_2026_04_08.py'
OUT_JSON = TMP_DIR / 'psins_singleaxis_three_group_noise3x_2026-04-08.json'


NOISE_CFG = {
    'arw_dpsh': 0.0015,
    'vrw_ugpsHz': 1.5,
    'bi_g_dph': 0.0007,
    'bi_a_ug': 5.0,
    'tau_g_s': 300.0,
    'tau_a_s': 300.0,
    'seed': 42,
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module('singleaxis_three_group_noise3x_basecmp_20260408', BASE_COMPARE_SCRIPT)
halfturn = load_module('singleaxis_three_group_noise3x_halfturn_20260408', HALFTURN_A0900_SCRIPT)

acc18 = base.acc18
glv = base.glv
base12 = base.base12
markov_noise = base.markov_noise
h24 = base.h24


REP_SCD = {
    'name': 'singleaxis_halfturn_attbias_fullscale_a0100_noise3x',
    'description': 'Representative SCD method from the previous three-method table: half-turn triggering, att+bias ↔ full-scale mask, alpha=0.1, now rerun under 3x ARW/VRW noise.',
    'alpha': 0.1,
    'row_indices': [0, 1, 2, 6, 7, 8, 9, 10, 11],
    'col_indices': [18, 19, 20, 21, 22, 23],
    'warmup_trigger_s': 5.0,
    'periodic_trigger_s': 37.5,
    'trigger_mode': 'time_halfturn_continuous_rotation',
}


GROUPS = {
    'g1_no_markov': {
        'display': 'G1 no Markov (base18)',
        'family_note': 'No Markov states, no SCD.',
    },
    'g2_with_markov': {
        'display': 'G2 with Markov (plain24)',
        'family_note': '24-state Markov/plain24, no SCD.',
    },
    'g3_with_scd': {
        'display': 'G3 with SCD (half-turn, alpha=0.1)',
        'family_note': 'Representative SCD method used in the previous three-method table: half-turn triggering + att+bias ↔ full-scale + alpha=0.1.',
    },
}


def write_payload(payload: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def build_truth_injection_imuerr() -> dict[str, np.ndarray]:
    truth_imuerr = base12.build_imuerr()
    truth_imuerr['web'] = np.zeros(3)
    truth_imuerr['wdb'] = np.zeros(3)
    return truth_imuerr


def build_filter_imuerr() -> dict[str, np.ndarray]:
    imuerr = base12.build_imuerr()
    imuerr['web'] = np.full(3, NOISE_CFG['arw_dpsh'] * glv.dpsh)
    imuerr['wdb'] = np.full(3, NOISE_CFG['vrw_ugpsHz'] * glv.ugpsHz)
    imuerr['eb'] = np.full(3, NOISE_CFG['bi_g_dph'] * glv.dph)
    imuerr['db'] = np.full(3, NOISE_CFG['bi_a_ug'] * glv.ug)
    return imuerr


def build_shared_dataset() -> dict[str, Any]:
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att0_true = base.ATT0_TRUE_DEG * glv.deg
    paras = np.array([
        [1, 0, 0, 1, base.TOTAL_ANGLE_DEG * glv.deg, base.DURATION_S, 0.0, 0.0],
    ], dtype=float)
    att_truth = acc18.attrottt(att0_true, paras, base.TS)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imu_truth = acc18.imuadderr(imu_clean, build_truth_injection_imuerr())
    imu_noisy = markov_noise.imuadderr_full(
        imu_truth,
        base.TS,
        arw=NOISE_CFG['arw_dpsh'] * glv.dpsh,
        vrw=NOISE_CFG['vrw_ugpsHz'] * glv.ugpsHz,
        bi_g=NOISE_CFG['bi_g_dph'] * glv.dph,
        tau_g=NOISE_CFG['tau_g_s'],
        bi_a=NOISE_CFG['bi_a_ug'] * glv.ug,
        tau_a=NOISE_CFG['tau_a_s'],
        seed=NOISE_CFG['seed'],
    )

    phi0 = base.PHI_GUESS_DEG * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_true), phi0))
    truth_att = att_truth[-1, 0:3].copy()

    return {
        'pos0': pos0,
        'phi0': phi0,
        'att0_true_deg': base.ATT0_TRUE_DEG.tolist(),
        'att0_guess_deg': (att0_guess / glv.deg).tolist(),
        'truth_att_deg': (truth_att / glv.deg).tolist(),
        'truth_att': truth_att,
        'imu_noisy': imu_noisy,
        'paras': paras,
        'shared_truth_semantics': {
            'deterministic_truth_from_existing_family': {
                'eb_dph': [0.01, 0.01, 0.01],
                'db_ug': [100.0, 100.0, 100.0],
                'dKg_ppm_diag': [30.0, 30.0, 30.0],
                'dKa_ppm_diag': [30.0, 30.0, 30.0],
            },
            'stochastic_noise_overridden_by_user': NOISE_CFG,
            'att0_true_deg': base.ATT0_TRUE_DEG.tolist(),
            'phi_guess_deg': base.PHI_GUESS_DEG.tolist(),
        },
    }


def make_comparison(groups: dict[str, Any]) -> dict[str, Any]:
    g1 = groups['g1_no_markov']['final_metrics']
    g2 = groups['g2_with_markov']['final_metrics']
    g3 = groups['g3_with_scd']['final_metrics']
    return {
        'delta_g2_minus_g1': {
            'yaw_arcsec': float(g2['final_yaw_abs_arcsec'] - g1['final_yaw_abs_arcsec']),
            'norm_arcsec': float(g2['final_norm_arcsec'] - g1['final_norm_arcsec']),
        },
        'delta_g3_minus_g2': {
            'yaw_arcsec': float(g3['final_yaw_abs_arcsec'] - g2['final_yaw_abs_arcsec']),
            'norm_arcsec': float(g3['final_norm_arcsec'] - g2['final_norm_arcsec']),
        },
        'delta_g3_minus_g1': {
            'yaw_arcsec': float(g3['final_yaw_abs_arcsec'] - g1['final_yaw_abs_arcsec']),
            'norm_arcsec': float(g3['final_norm_arcsec'] - g1['final_norm_arcsec']),
        },
    }


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    shared = build_shared_dataset()
    filter_imuerr = build_filter_imuerr()

    payload: dict[str, Any] = {
        'task': 'psins_singleaxis_three_group_noise3x_2026_04_08',
        'status': 'started',
        'paths': {
            'script': str(Path(__file__).resolve()),
            'json': str(OUT_JSON),
            'base_compare_script': str(BASE_COMPARE_SCRIPT),
            'halfturn_script': str(HALFTURN_A0900_SCRIPT),
        },
        'fixed_setup': {
            'path_family': 'single_axis_only',
            'duration_s': base.DURATION_S,
            'speed_dps': base.SPEED_DPS,
            'total_angle_deg': base.TOTAL_ANGLE_DEG,
            'outer_iterations': base.OUTER_ITERATIONS,
            'seed': NOISE_CFG['seed'],
            'noise': NOISE_CFG,
        },
        'shared_truth_semantics': shared['shared_truth_semantics'],
        'groups': {},
        'group_mapping': GROUPS,
        'sdc_representative': REP_SCD,
        'timestamps': {
            'started_epoch_s': time.time(),
        },
    }
    write_payload(payload)

    print('NOISE3X_RUN_STARTED', flush=True)
    print(json.dumps({'out_json': str(OUT_JSON), 'noise_cfg': NOISE_CFG}, ensure_ascii=False, indent=2), flush=True)

    payload['groups']['g1_no_markov'] = base.run_g1_base18(shared, filter_imuerr)
    payload['groups']['g1_no_markov']['group_key'] = 'g1_no_markov'
    payload['groups']['g1_no_markov']['display'] = GROUPS['g1_no_markov']['display']
    payload['status'] = 'g1_done'
    write_payload(payload)
    print(json.dumps({'group': 'g1_no_markov', 'final_metrics': payload['groups']['g1_no_markov']['final_metrics']}, ensure_ascii=False, indent=2), flush=True)

    payload['groups']['g2_with_markov'] = base.run_g24_outer(
        shared,
        filter_imuerr,
        candidate=None,
        group_key='g2_base18_plus_markov6',
    )
    payload['groups']['g2_with_markov']['group_key'] = 'g2_with_markov'
    payload['groups']['g2_with_markov']['display'] = GROUPS['g2_with_markov']['display']
    payload['status'] = 'g2_done'
    write_payload(payload)
    print(json.dumps({'group': 'g2_with_markov', 'final_metrics': payload['groups']['g2_with_markov']['final_metrics']}, ensure_ascii=False, indent=2), flush=True)

    halfturn.CANDIDATE.update(REP_SCD)
    payload['groups']['g3_with_scd'] = halfturn.run_outer(shared, filter_imuerr)
    payload['groups']['g3_with_scd']['group_key'] = 'g3_with_scd'
    payload['groups']['g3_with_scd']['display'] = GROUPS['g3_with_scd']['display']
    payload['status'] = 'done'
    payload['comparison'] = make_comparison(payload['groups'])
    payload['timestamps']['finished_epoch_s'] = time.time()
    payload['runtime_total_s'] = payload['timestamps']['finished_epoch_s'] - payload['timestamps']['started_epoch_s']
    write_payload(payload)
    print(json.dumps({'group': 'g3_with_scd', 'final_metrics': payload['groups']['g3_with_scd']['final_metrics']}, ensure_ascii=False, indent=2), flush=True)
    print('__RESULT_JSON__=' + json.dumps({'out_json': str(OUT_JSON), 'comparison': payload['comparison']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
