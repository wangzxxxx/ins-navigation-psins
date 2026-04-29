#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

WORKSPACE = Path('/root/.openclaw/workspace')
TMP_DIR = WORKSPACE / 'tmp'
SCRIPTS_DIR = WORKSPACE / 'scripts'
TMP_PSINS_DIR = WORKSPACE / 'tmp_psins_py' / 'psins_py'

BASE18_SCRIPT = TMP_DIR / 'sweep_inbound_scale18_singleaxis_integer_turns_2026_04_08.py'
BASE12_SCRIPT = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
H24_SCRIPT = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
LLM_SCD_SCRIPT = SCRIPTS_DIR / 'probe_ch4_llm_scd_only_alignment_2026-04-03.py'
MARKOV_NOISE_SCRIPT = TMP_PSINS_DIR / 'test_calibration_markov_pruned.py'

DATE_TAG = '2026-04-08'
OUT_JSON = TMP_DIR / 'psins_singleaxis_three_group_compare_2026-04-08.json'

TS = 0.01
DURATION_S = 300.0
SPEED_DPS = 4.8
TOTAL_ANGLE_DEG = SPEED_DPS * DURATION_S
OUTER_ITERATIONS = 100
SEED = 42
WVN = np.array([0.01, 0.01, 0.01], dtype=float)
ATT0_TRUE_DEG = np.array([1.0, 0.0, 10.0], dtype=float)
PHI_GUESS_DEG = np.array([0.1, 0.1, 0.5], dtype=float)
ROT_GATE_DPS = 5.0
LLM_CANDIDATE_NAME = 'llm_attbias_fullscale_early'

NOISE_CFG = {
    'arw_dpsh': 0.0005,
    'vrw_ugpsHz': 0.5,
    'bi_g_dph': 0.0007,
    'bi_a_ug': 5.0,
    'tau_g_s': 300.0,
    'tau_a_s': 300.0,
    'seed': SEED,
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base18 = load_module('singleaxis_base18_20260408_sub', BASE18_SCRIPT)
base12 = load_module('singleaxis_base12_20260408_sub', BASE12_SCRIPT)
h24 = load_module('singleaxis_h24_20260408_sub', H24_SCRIPT)
llm_mod = load_module('singleaxis_llmscd_20260408_sub', LLM_SCD_SCRIPT)
markov_noise = load_module('singleaxis_markovnoise_20260408_sub', MARKOV_NOISE_SCRIPT)
acc18 = h24.load_acc18()
glv = acc18.glv


GROUPS = {
    'g1_base18': {
        'display': 'G1 base 18-state',
        'state_layout': 'phi(3), dv(3), eb(3), db(3), kg(3), ka(3)',
        'mapped_files': [
            str(BASE18_SCRIPT),
        ],
        'mapped_functions': [
            'alignvn_dar_18state_scale_recorded',
        ],
        'family_note': 'Exact same inbound-style single-axis scale18 outer-loop family (kg/ka present, no ng/xa Markov states).',
    },
    'g2_base18_plus_markov6': {
        'display': 'G2 base 18-state + 6-state Markov',
        'state_layout': 'phi(3), dv(3), eb(3), db(3), ng(3), xa(3), kg(3), ka(3)',
        'mapped_files': [
            str(H24_SCRIPT),
            str(BASE12_SCRIPT),
            str(WORKSPACE / 'psins_method_bench' / 'scripts' / 'compare_four_group_alignment_arcsec_param_outer_fixedbaseline_2026-04-07.py'),
        ],
        'mapped_functions': [
            'avnkfinit_24',
            'apply_scale_wash',
            '24-state plain Markov/GM single-pass dynamics adapted from run_24state_round',
        ],
        'family_note': 'Conservative plain24 mapping: same 24-state single-axis semantics previously used in PSINS comparisons, transplanted onto the requested 4.8 deg/s / 300 s path.',
    },
    'g3_base18_plus_markov6_llm_scd': {
        'display': 'G3 base 18-state + 6-state Markov + LLM + SCD',
        'state_layout': 'phi(3), dv(3), eb(3), db(3), ng(3), xa(3), kg(3), ka(3)',
        'mapped_files': [
            str(H24_SCRIPT),
            str(LLM_SCD_SCRIPT),
            str(WORKSPACE / 'psins_method_bench' / 'scripts' / 'compare_four_group_alignment_arcsec_param_outer_fixedbaseline_2026-04-07.py'),
        ],
        'mapped_functions': [
            'avnkfinit_24',
            'apply_masked_scd_once',
            'Candidate llm_attbias_fullscale_early',
        ],
        'family_note': 'Conservative LLM+SCD mapping: reuse the already-ranked Chapter-4 constrained-SCD candidate llm_attbias_fullscale_early on the same plain24 backbone, without path-specific retuning.',
    },
}


_CANDIDATE = None


def pick_candidate():
    global _CANDIDATE
    if _CANDIDATE is None:
        for cand in llm_mod.CANDIDATES:
            if cand.name == LLM_CANDIDATE_NAME:
                _CANDIDATE = cand
                break
        if _CANDIDATE is None:
            raise KeyError(f'LLM candidate not found: {LLM_CANDIDATE_NAME}')
    return _CANDIDATE


def serializable_candidate(candidate) -> dict[str, Any]:
    if is_dataclass(candidate):
        return asdict(candidate)
    return dict(candidate)


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
    att0_true = ATT0_TRUE_DEG * glv.deg
    paras = np.array([
        [1, 0, 0, 1, TOTAL_ANGLE_DEG * glv.deg, DURATION_S, 0.0, 0.0],
    ], dtype=float)
    att_truth = acc18.attrottt(att0_true, paras, TS)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imu_truth = acc18.imuadderr(imu_clean, build_truth_injection_imuerr())
    imu_noisy = markov_noise.imuadderr_full(
        imu_truth,
        TS,
        arw=NOISE_CFG['arw_dpsh'] * glv.dpsh,
        vrw=NOISE_CFG['vrw_ugpsHz'] * glv.ugpsHz,
        bi_g=NOISE_CFG['bi_g_dph'] * glv.dph,
        tau_g=NOISE_CFG['tau_g_s'],
        bi_a=NOISE_CFG['bi_a_ug'] * glv.ug,
        tau_a=NOISE_CFG['tau_a_s'],
        seed=NOISE_CFG['seed'],
    )

    phi0 = PHI_GUESS_DEG * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_true), phi0))
    truth_att = att_truth[-1, 0:3].copy()

    return {
        'pos0': pos0,
        'phi0': phi0,
        'att0_true_deg': ATT0_TRUE_DEG.tolist(),
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
            'att0_true_deg': ATT0_TRUE_DEG.tolist(),
            'phi_guess_deg': PHI_GUESS_DEG.tolist(),
        },
    }


def make_metrics(final_att: np.ndarray, truth_att: np.ndarray) -> dict[str, Any]:
    phi_err = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att))
    err_arcsec = phi_err / glv.sec
    abs_err = np.abs(err_arcsec)
    return {
        'final_att_deg': (final_att / glv.deg).tolist(),
        'final_att_err_arcsec': err_arcsec.tolist(),
        'final_abs_att_err_arcsec': abs_err.tolist(),
        'final_roll_abs_arcsec': float(abs_err[0]),
        'final_pitch_abs_arcsec': float(abs_err[1]),
        'final_yaw_abs_arcsec': float(abs_err[2]),
        'final_norm_arcsec': float(np.linalg.norm(err_arcsec)),
    }


def run_plain24_single_pass(imu_input: np.ndarray, att0_guess: np.ndarray, pos0: np.ndarray,
                            phi0: np.ndarray, filter_imuerr: dict[str, np.ndarray],
                            candidate=None) -> dict[str, Any]:
    imu_corr = np.copy(imu_input)
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnbi = acc18.a2qua(att0_guess) if len(att0_guess) == 3 else np.asarray(att0_guess).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = ROT_GATE_DPS * glv.deg

    kf = h24.avnkfinit_24(
        nts,
        pos0,
        phi0,
        filter_imuerr,
        WVN.copy(),
        np.full(3, NOISE_CFG['bi_g_dph']) * glv.dph,
        np.full(3, NOISE_CFG['tau_g_s']),
        np.full(3, NOISE_CFG['bi_a_ug']) * glv.ug,
        np.full(3, NOISE_CFG['tau_a_s']),
        enable_scale_states=True,
    )
    vn = np.zeros(3)
    time_since_rot_stop = 0.0
    scd_trigger_count = 0

    for k in range(0, length, nn):
        wvm = imu_corr[k:k + nn, 0:6]
        phim, dvbm = acc18.cnscl(wvm)
        cnb = acc18.q2mat(qnbi)
        dvn = cnn @ cnb @ dvbm
        vn = vn + dvn + eth.gn * nts
        qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

        phi_k = kf['Phikk_1'].copy()
        cnbts = cnb * nts
        phi_k[3:6, 0:3] = acc18.askew(dvn)
        phi_k[3:6, 9:12] = cnbts
        phi_k[3:6, 15:18] = cnbts
        phi_k[0:3, 6:9] = -cnbts
        phi_k[0:3, 12:15] = -cnbts
        phi_k[12:15, 12:15] = np.diag(kf['fg'])
        phi_k[15:18, 15:18] = np.diag(kf['fa'])

        high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
        if high_rot:
            phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
            phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
            time_since_rot_stop = 0.0
        else:
            phi_k[0:3, 18:21] = 0.0
            phi_k[3:6, 21:24] = 0.0
            time_since_rot_stop += nts

        kf['Phikk_1'] = phi_k
        kf = acc18.kfupdate(kf, vn)

        qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
        kf['xk'][0:3] *= 0.09
        vn = vn - 0.91 * kf['xk'][3:6]
        kf['xk'][3:6] *= 0.09

        if candidate is not None and (not high_rot):
            if time_since_rot_stop >= float(candidate.transition_duration_s):
                kf = llm_mod.apply_masked_scd_once(kf, candidate.row_indices, candidate.col_indices, candidate.alpha)
                scd_trigger_count += 1
                time_since_rot_stop = -1e9  # keep once-per-release-phase behavior in this single-pass setting

    final_att = acc18.q2att(qnbi)
    return {
        'final_att': final_att,
        'est_eb': np.array(kf['xk'][6:9], dtype=float),
        'est_db': np.array(kf['xk'][9:12], dtype=float),
        'est_kg': np.array(kf['xk'][18:21], dtype=float),
        'est_ka': np.array(kf['xk'][21:24], dtype=float),
        'est_eb_dph': (kf['xk'][6:9] / glv.dph).tolist(),
        'est_db_ug': (kf['xk'][9:12] / glv.ug).tolist(),
        'est_kg_ppm': (kf['xk'][18:21] / glv.ppm).tolist(),
        'est_ka_ppm': (kf['xk'][21:24] / glv.ppm).tolist(),
        'scd_trigger_count': int(scd_trigger_count),
    }


def run_g1_base18(shared: dict[str, Any], filter_imuerr: dict[str, np.ndarray]) -> dict[str, Any]:
    t0 = time.time()
    records = base18.alignvn_dar_18state_scale_recorded(
        shared['imu_noisy'].copy(),
        np.array(shared['att0_guess_deg']) * glv.deg,
        shared['pos0'],
        shared['phi0'],
        filter_imuerr,
        WVN.copy(),
        max_iter=OUTER_ITERATIONS,
    )
    final_rec = records[-1]
    metrics = make_metrics(np.array(final_rec['final_att'], dtype=float), shared['truth_att'])
    return {
        'group_key': 'g1_base18',
        'display': GROUPS['g1_base18']['display'],
        'outer_iterations': OUTER_ITERATIONS,
        'runtime_s': time.time() - t0,
        'final_metrics': metrics,
        'final_bias_est_dph': (np.array(final_rec['cumulative_eb']) / glv.dph).tolist(),
        'final_bias_est_ug': (np.array(final_rec['cumulative_db']) / glv.ug).tolist(),
        'final_scale_est_kg_ppm': (np.array(final_rec['cumulative_kg']) / glv.ppm).tolist(),
        'final_scale_est_ka_ppm': (np.array(final_rec['cumulative_ka']) / glv.ppm).tolist(),
        'scd_trigger_total': 0,
        'outer_log_tail': [
            {
                'outer_iteration': int(rec['iter']),
                'est_eb_dph': (np.array(rec['cumulative_eb']) / glv.dph).tolist(),
                'est_db_ug': (np.array(rec['cumulative_db']) / glv.ug).tolist(),
                'est_kg_ppm': (np.array(rec['cumulative_kg']) / glv.ppm).tolist(),
                'est_ka_ppm': (np.array(rec['cumulative_ka']) / glv.ppm).tolist(),
                'final_yaw_abs_arcsec': float(abs(acc18.qq2phi(acc18.a2qua(rec['final_att']), acc18.a2qua(shared['truth_att']))[2] / glv.sec)),
            }
            for rec in records[-5:]
        ],
    }


def run_g24_outer(shared: dict[str, Any], filter_imuerr: dict[str, np.ndarray], candidate=None, group_key: str = '') -> dict[str, Any]:
    t0 = time.time()
    imu_work = shared['imu_noisy'].copy()
    outer_logs: list[dict[str, Any]] = []
    scd_trigger_total = 0

    att0_guess = np.array(shared['att0_guess_deg']) * glv.deg
    for outer_it in range(1, OUTER_ITERATIONS + 1):
        result = run_plain24_single_pass(
            imu_work,
            att0_guess,
            shared['pos0'],
            shared['phi0'],
            filter_imuerr,
            candidate=candidate,
        )
        scd_trigger_total += int(result['scd_trigger_count'])

        imu_work[:, 0:3] -= result['est_eb'] * TS
        imu_work[:, 3:6] -= result['est_db'] * TS
        imu_work = h24.apply_scale_wash(imu_work, result['est_kg'], result['est_ka'], 1.0)

        metrics = make_metrics(np.array(result['final_att'], dtype=float), shared['truth_att'])
        outer_logs.append({
            'outer_iteration': outer_it,
            'final_roll_abs_arcsec': metrics['final_roll_abs_arcsec'],
            'final_pitch_abs_arcsec': metrics['final_pitch_abs_arcsec'],
            'final_yaw_abs_arcsec': metrics['final_yaw_abs_arcsec'],
            'final_norm_arcsec': metrics['final_norm_arcsec'],
            'est_eb_dph': result['est_eb_dph'],
            'est_db_ug': result['est_db_ug'],
            'est_kg_ppm': result['est_kg_ppm'],
            'est_ka_ppm': result['est_ka_ppm'],
            'scd_trigger_count': int(result['scd_trigger_count']),
        })

    final = outer_logs[-1]
    return {
        'group_key': group_key,
        'display': GROUPS[group_key]['display'],
        'outer_iterations': OUTER_ITERATIONS,
        'runtime_s': time.time() - t0,
        'final_metrics': {
            'final_att_deg': make_metrics(np.array(result['final_att'], dtype=float), shared['truth_att'])['final_att_deg'],
            'final_att_err_arcsec': make_metrics(np.array(result['final_att'], dtype=float), shared['truth_att'])['final_att_err_arcsec'],
            'final_abs_att_err_arcsec': make_metrics(np.array(result['final_att'], dtype=float), shared['truth_att'])['final_abs_att_err_arcsec'],
            'final_roll_abs_arcsec': final['final_roll_abs_arcsec'],
            'final_pitch_abs_arcsec': final['final_pitch_abs_arcsec'],
            'final_yaw_abs_arcsec': final['final_yaw_abs_arcsec'],
            'final_norm_arcsec': final['final_norm_arcsec'],
        },
        'final_bias_est_dph': result['est_eb_dph'],
        'final_bias_est_ug': result['est_db_ug'],
        'final_scale_est_kg_ppm': result['est_kg_ppm'],
        'final_scale_est_ka_ppm': result['est_ka_ppm'],
        'scd_trigger_total': int(scd_trigger_total),
        'outer_log_tail': outer_logs[-5:],
    }


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    candidate = pick_candidate()
    shared = build_shared_dataset()
    filter_imuerr = build_filter_imuerr()

    payload: dict[str, Any] = {
        'task': 'psins_singleaxis_three_group_compare_2026_04_08',
        'status': 'started',
        'paths': {
            'script': str(Path(__file__).resolve()),
            'json': str(OUT_JSON),
        },
        'mapping': GROUPS,
        'conservative_mapping_note': (
            'No dedicated pre-existing single-axis 4.8 deg/s / 300 s LLM+SCD runner was found. '
            'So G3 conservatively reuses the already-ranked candidate llm_attbias_fullscale_early '
            'from the existing Chapter-4 plain24+constrained-SCD family, without any new retuning. '
            'Because the requested path is a continuous single-axis rotation with no explicit dwell, '
            'the release-triggered SCD may trigger rarely or not at all; the script records the actual trigger count.'
        ),
        'fixed_setup': {
            'path_family': 'single_axis_only',
            'duration_s': DURATION_S,
            'speed_dps': SPEED_DPS,
            'total_angle_deg': TOTAL_ANGLE_DEG,
            'outer_iterations': OUTER_ITERATIONS,
            'seed': SEED,
            'noise': NOISE_CFG,
        },
        'shared_truth_semantics': shared['shared_truth_semantics'],
        'groups': {},
        'selected_llm_candidate': serializable_candidate(candidate),
        'timestamps': {
            'started_epoch_s': time.time(),
        },
    }
    write_payload(payload)

    print('RUN_STARTED')
    print(json.dumps({
        'mapped_groups': {
            key: {
                'display': value['display'],
                'files': value['mapped_files'],
            }
            for key, value in GROUPS.items()
        },
        'out_json': str(OUT_JSON),
    }, ensure_ascii=False, indent=2))

    print('RUN_G1 base18 ...', flush=True)
    payload['groups']['g1_base18'] = run_g1_base18(shared, filter_imuerr)
    payload['status'] = 'g1_done'
    write_payload(payload)
    print(json.dumps({
        'group': 'g1_base18',
        'final_metrics': payload['groups']['g1_base18']['final_metrics'],
        'runtime_s': payload['groups']['g1_base18']['runtime_s'],
    }, ensure_ascii=False, indent=2), flush=True)

    print('RUN_G2 base18+markov6 ...', flush=True)
    payload['groups']['g2_base18_plus_markov6'] = run_g24_outer(
        shared,
        filter_imuerr,
        candidate=None,
        group_key='g2_base18_plus_markov6',
    )
    payload['status'] = 'g2_done'
    write_payload(payload)
    print(json.dumps({
        'group': 'g2_base18_plus_markov6',
        'final_metrics': payload['groups']['g2_base18_plus_markov6']['final_metrics'],
        'runtime_s': payload['groups']['g2_base18_plus_markov6']['runtime_s'],
    }, ensure_ascii=False, indent=2), flush=True)

    print('RUN_G3 base18+markov6+llm+scd ...', flush=True)
    payload['groups']['g3_base18_plus_markov6_llm_scd'] = run_g24_outer(
        shared,
        filter_imuerr,
        candidate=candidate,
        group_key='g3_base18_plus_markov6_llm_scd',
    )
    payload['status'] = 'done'
    payload['timestamps']['finished_epoch_s'] = time.time()
    payload['runtime_total_s'] = payload['timestamps']['finished_epoch_s'] - payload['timestamps']['started_epoch_s']
    write_payload(payload)
    print(json.dumps({
        'group': 'g3_base18_plus_markov6_llm_scd',
        'final_metrics': payload['groups']['g3_base18_plus_markov6_llm_scd']['final_metrics'],
        'runtime_s': payload['groups']['g3_base18_plus_markov6_llm_scd']['runtime_s'],
        'scd_trigger_total': payload['groups']['g3_base18_plus_markov6_llm_scd']['scd_trigger_total'],
    }, ensure_ascii=False, indent=2), flush=True)

    summary = {
        key: {
            'final_roll_abs_arcsec': value['final_metrics']['final_roll_abs_arcsec'],
            'final_pitch_abs_arcsec': value['final_metrics']['final_pitch_abs_arcsec'],
            'final_yaw_abs_arcsec': value['final_metrics']['final_yaw_abs_arcsec'],
            'final_norm_arcsec': value['final_metrics']['final_norm_arcsec'],
            'runtime_s': value['runtime_s'],
            'scd_trigger_total': value['scd_trigger_total'],
        }
        for key, value in payload['groups'].items()
    }
    print('__RESULT_JSON__=' + json.dumps({
        'out_json': str(OUT_JSON),
        'summary': summary,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
