#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

WORKSPACE = Path('/root/.openclaw/workspace')
TMP_DIR = WORKSPACE / 'tmp'
NOISE3X_SCRIPT = TMP_DIR / 'compare_psins_singleaxis_three_group_noise3x_2026_04_08.py'
OUT_JSON = TMP_DIR / 'psins_singleaxis_three_group_noise3x_2026-04-08.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = load_module('singleaxis_three_group_noise3x_resume_20260409', NOISE3X_SCRIPT)
    payload = json.loads(OUT_JSON.read_text(encoding='utf-8'))

    shared = mod.build_shared_dataset()
    filter_imuerr = mod.build_filter_imuerr()
    mod.halfturn.CANDIDATE.update(mod.REP_SCD)

    t0 = time.time()
    result = mod.halfturn.run_outer(shared, filter_imuerr)
    payload.setdefault('groups', {})['g3_with_scd'] = result
    payload['groups']['g3_with_scd']['group_key'] = 'g3_with_scd'
    payload['groups']['g3_with_scd']['display'] = mod.GROUPS['g3_with_scd']['display']
    payload['status'] = 'done'
    payload['comparison'] = mod.make_comparison(payload['groups'])
    payload.setdefault('timestamps', {})['g3_resumed_epoch_s'] = t0
    payload['timestamps']['finished_epoch_s'] = time.time()
    payload['runtime_total_s'] = payload['timestamps']['finished_epoch_s'] - payload['timestamps'].get('started_epoch_s', t0)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'out_json': str(OUT_JSON),
        'g3_final_metrics': payload['groups']['g3_with_scd']['final_metrics'],
        'comparison': payload['comparison'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
