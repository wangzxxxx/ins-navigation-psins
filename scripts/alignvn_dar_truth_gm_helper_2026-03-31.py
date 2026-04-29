#!/usr/bin/env python3
"""Reusable truth-side GM-drift helper for Chapter 4 DAR alignment probes.

This module keeps the current baseline IMU truth injection intact and adds an
optional first-order Gauss-Markov (GM) bias drift on top of it.

Interpretation:
- baseline terms (`eb/db/web/wdb/dKg/dKa`) still come from the existing
  `build_imuerr()` path in `alignvn_dar_12state_py_iterfix_2026-03-30.py`
- optional GM drift is injected only on the truth/noisy IMU side
- the filter model is *not* upgraded with explicit GM states here; the point is
  to test robustness against unmodeled slow bias drift
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

from psins_py.imu_utils import imuadderr  # noqa: E402
from psins_py.nav_utils import glv  # noqa: E402

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
_BASE12 = None

BASE_IMUERR_KEYS = ('eb', 'db', 'web', 'wdb', 'dKg', 'dKa', 'Ka2', 'rx', 'ry', 'rz')

GM_PROFILE_LIBRARY: dict[str, dict[str, Any]] = {
    'baseline': {
        'label': 'baseline',
        'enable_truth_gm': False,
        'gyro_sigma_dph': [0.0, 0.0, 0.0],
        'accel_sigma_ug': [0.0, 0.0, 0.0],
        'tau_g_s': [200.0, 200.0, 200.0],
        'tau_a_s': [200.0, 200.0, 200.0],
        'note': 'No added GM drift; exactly the current truth-side path.',
    },
    'tiny_gm': {
        'label': 'tiny_gm',
        'enable_truth_gm': True,
        'gyro_sigma_dph': [0.001, 0.001, 0.001],
        'accel_sigma_ug': [10.0, 10.0, 10.0],
        'tau_g_s': [200.0, 200.0, 200.0],
        'tau_a_s': [200.0, 200.0, 200.0],
        'note': 'Very light isotropic first-order GM drift: 10x smaller than the nominal constant eb/db magnitudes, intended for thesis-supplement sensitivity only.',
    },
    'small_gm': {
        'label': 'small_gm',
        'enable_truth_gm': True,
        'gyro_sigma_dph': [0.003, 0.003, 0.003],
        'accel_sigma_ug': [30.0, 30.0, 30.0],
        'tau_g_s': [200.0, 200.0, 200.0],
        'tau_a_s': [200.0, 200.0, 200.0],
        'note': 'Still mild GM drift, but 3x the tiny setting and still clearly below the nominal constant eb/db magnitudes.',
    },
    'mild_gm': {
        'label': 'mild_gm',
        'enable_truth_gm': True,
        'gyro_sigma_dph': [0.01, 0.01, 0.01],
        'accel_sigma_ug': [100.0, 100.0, 100.0],
        'tau_g_s': [200.0, 200.0, 200.0],
        'tau_a_s': [200.0, 200.0, 200.0],
        'note': 'Legacy heavier check: isotropic first-order GM drift with sigma equal to the current constant eb/db magnitudes.',
    },
    'stronger_gm': {
        'label': 'stronger_gm',
        'enable_truth_gm': True,
        'gyro_sigma_dph': [0.03, 0.03, 0.03],
        'accel_sigma_ug': [300.0, 300.0, 300.0],
        'tau_g_s': [200.0, 200.0, 200.0],
        'tau_a_s': [200.0, 200.0, 200.0],
        'note': 'Legacy heavier check: same GM model, but 3x the mild stationary sigma.',
    },
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def load_base12():
    global _BASE12
    if _BASE12 is None:
        _BASE12 = load_module('alignvn_base12_truth_gm_helper_20260331', BASE12_PATH)
    return _BASE12



def get_gm_profile(profile: str) -> dict[str, Any]:
    if profile not in GM_PROFILE_LIBRARY:
        raise KeyError(f'unknown GM profile: {profile}')
    return GM_PROFILE_LIBRARY[profile]



def build_truth_imuerr_variant(profile: str = 'baseline', base_imuerr: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Return the baseline IMU error dict plus optional truth-side GM metadata."""
    base12 = load_base12()
    if base_imuerr is None:
        base_imuerr = base12.build_imuerr()

    gm = get_gm_profile(profile)
    imuerr: dict[str, Any] = {key: np.array(value, copy=True) for key, value in base_imuerr.items()}
    imuerr['truth_gm_profile'] = profile
    imuerr['truth_gm_enabled'] = bool(gm['enable_truth_gm'])
    imuerr['truth_gm_gyro_sigma'] = np.array(gm['gyro_sigma_dph'], dtype=float) * glv.dph
    imuerr['truth_gm_accel_sigma'] = np.array(gm['accel_sigma_ug'], dtype=float) * glv.ug
    imuerr['truth_gm_tau_g_s'] = np.array(gm['tau_g_s'], dtype=float)
    imuerr['truth_gm_tau_a_s'] = np.array(gm['tau_a_s'], dtype=float)
    imuerr['truth_gm_note'] = gm['note']
    return imuerr



def _generate_first_order_gm_bias(n: int, ts: float, sigma: np.ndarray, tau_s: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).reshape(3)
    tau_s = np.asarray(tau_s, dtype=float).reshape(3)
    coeff = np.exp(-ts / np.maximum(tau_s, 1e-12))
    drive = sigma * np.sqrt(np.maximum(1.0 - coeff**2, 0.0))

    out = np.zeros((n, 3))
    state = np.zeros(3)
    for k in range(n):
        state = coeff * state + drive * np.random.randn(3)
        out[k] = state
    return out



def apply_truth_imu_errors(imu_clean: np.ndarray, imuerr: dict[str, Any]) -> np.ndarray:
    """Apply the current baseline error path plus optional truth-side GM drift."""
    imu_base = {key: np.array(imuerr[key], copy=True) for key in BASE_IMUERR_KEYS if key in imuerr}
    imu_noisy = imuadderr(imu_clean, imu_base)

    if not bool(imuerr.get('truth_gm_enabled', False)):
        return imu_noisy

    ts = float(imu_noisy[1, -1] - imu_noisy[0, -1])
    n = int(imu_noisy.shape[0])
    gyro_bias = _generate_first_order_gm_bias(
        n=n,
        ts=ts,
        sigma=np.asarray(imuerr['truth_gm_gyro_sigma']),
        tau_s=np.asarray(imuerr['truth_gm_tau_g_s']),
    )
    accel_bias = _generate_first_order_gm_bias(
        n=n,
        ts=ts,
        sigma=np.asarray(imuerr['truth_gm_accel_sigma']),
        tau_s=np.asarray(imuerr['truth_gm_tau_a_s']),
    )
    imu_noisy[:, 0:3] += gyro_bias * ts
    imu_noisy[:, 3:6] += accel_bias * ts
    return imu_noisy



def describe_truth_profile(profile: str) -> dict[str, Any]:
    gm = get_gm_profile(profile)
    return {
        'profile': profile,
        'enable_truth_gm': bool(gm['enable_truth_gm']),
        'gyro_sigma_dph': [float(x) for x in gm['gyro_sigma_dph']],
        'accel_sigma_ug': [float(x) for x in gm['accel_sigma_ug']],
        'tau_g_s': [float(x) for x in gm['tau_g_s']],
        'tau_a_s': [float(x) for x in gm['tau_a_s']],
        'note': gm['note'],
        'injection_formula': 'b_k = exp(-dt/tau) * b_{k-1} + sigma * sqrt(1-exp(-2dt/tau)) * N(0,1); imu += b_k * dt',
    }


if __name__ == '__main__':
    import json

    print(json.dumps({k: describe_truth_profile(k) for k in GM_PROFILE_LIBRARY}, ensure_ascii=False, indent=2))
