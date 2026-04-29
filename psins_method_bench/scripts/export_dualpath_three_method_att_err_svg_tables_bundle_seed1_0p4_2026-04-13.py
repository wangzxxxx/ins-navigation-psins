#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

WORKSPACE = Path('/root/.openclaw/workspace')
COMBO_SCRIPT = WORKSPACE / 'psins_method_bench' / 'scripts' / 'generate_dualpath_three_method_att_err_sigma_combo_0p4_2026-04-13.py'
SEED = 1
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_err_svg_tables_bundle_seed1_0p4_2026-04-13'
ZIP_PATH = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_err_svg_tables_bundle_seed1_0p4_2026-04-13.zip'
SEED1_COMBO_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_err_sigma_combo_seed1_0p4_2026-04-13'

AXIS_TO_STATE = {
    'att_err_x': 'phi_x',
    'att_err_y': 'phi_y',
    'att_err_z': 'phi_z',
}
AXIS_TITLES = {
    'att_err_x': '俯仰姿态对准误差（att_err_x）',
    'att_err_y': '横滚姿态对准误差（att_err_y）',
    'att_err_z': '航向姿态对准误差（att_err_z）',
}
CSV_COLUMNS = [
    'time_s',
    'G1baseline_err_arcsec',
    'G2MarkovMarkov_err_arcsec',
    'G3Markov+LLM+SCD_err_arcsec',
    'G1baseline_sigma_arcsec',
    'G2MarkovMarkov_sigma_arcsec',
    'G3Markov+LLM+SCD_sigma_arcsec',
]
GROUP_ORDER = ['g2_scaleonly_rotation', 'g3_markov_rotation', 'g4_scd_rotation']
GROUP_TO_ERR_COL = {
    'g2_scaleonly_rotation': 'G1baseline_err_arcsec',
    'g3_markov_rotation': 'G2MarkovMarkov_err_arcsec',
    'g4_scd_rotation': 'G3Markov+LLM+SCD_err_arcsec',
}
GROUP_TO_SIGMA_COL = {
    'g2_scaleonly_rotation': 'G1baseline_sigma_arcsec',
    'g3_markov_rotation': 'G2MarkovMarkov_sigma_arcsec',
    'g4_scd_rotation': 'G3Markov+LLM+SCD_sigma_arcsec',
}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def fmt(v: float) -> str:
    av = abs(v)
    if av >= 1e4 or (0 < av < 1e-6):
        return f'{v:.8e}'
    return f'{v:.8f}'


def ensure_seed1_combo_outputs(mod):
    SEED1_COMBO_DIR.mkdir(parents=True, exist_ok=True)
    need_generate = any(not (SEED1_COMBO_DIR / f'{axis}_combo.svg').exists() or not (SEED1_COMBO_DIR / f'{axis}_combo.png').exists() for axis in AXIS_TO_STATE)
    if not need_generate:
        return

    mod.custom.SEED = SEED
    shared = mod.custom.build_shared_dual_dataset_custom_noise_0p4()
    err_groups = [
        mod.aep.run_scale18_att_err(shared),
        mod.aep.run_plain24_att_err(shared),
        mod.aep.run_purescd_att_err(shared),
    ]
    state_groups = [
        mod.bp.trace_scale18_first3(shared),
        mod.bp.trace_plain24_first3(shared),
        mod.bp.trace_purescd24_first3(shared),
    ]
    sigma_map = {axis: mod.build_sigma_lookup(state_groups, state_label) for axis, state_label in AXIS_TO_STATE.items()}

    summary = {
        'task': 'dualpath_three_method_att_err_sigma_combo_seed1_0p4_2026_04_13',
        'note': 'Three-panel combo for 0.4x custom noise, seed=1: full error, tail zoom, and approximate covariance convergence (sigma(phi_x/y/z)).',
        'noise': '0.4x custom noise',
        'seed': SEED,
        'outer_iters': mod.OUTER_ITERS,
        'display_labels': mod.bp.GROUP_LABELS,
        'output_dir': str(SEED1_COMBO_DIR),
        'plots': [],
    }

    for axis in AXIS_TO_STATE:
        err_series = [
            {
                'group_key': g['group_key'],
                'label': g['label'],
                'x': g['x'],
                'y': g[axis],
                'iter_bounds_s': g['iter_bounds_s'],
            }
            for g in err_groups
        ]
        sigma_series = [
            {
                'group_key': gk,
                'label': sigma_map[axis][gk]['label'],
                'x': sigma_map[axis][gk]['x'],
                'y': sigma_map[axis][gk]['y'],
                'iter_bounds_s': sigma_map[axis][gk]['iter_bounds_s'],
            }
            for gk in GROUP_ORDER
        ]
        svg = SEED1_COMBO_DIR / f'{axis}_combo.svg'
        png = SEED1_COMBO_DIR / f'{axis}_combo.png'
        mod.render_combo(svg, png, axis, err_series, sigma_series)
        svg.write_text(svg.read_text(encoding='utf-8').replace('seed=0', f'seed={SEED}'), encoding='utf-8')
        os.system(f'ffmpeg -y -loglevel error -i "{svg}" -frames:v 1 "{png}"')
        summary['plots'].append({'axis': axis, 'svg': str(svg), 'png': str(png)})

    (SEED1_COMBO_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def build_series(mod):
    mod.custom.SEED = SEED
    shared = mod.custom.build_shared_dual_dataset_custom_noise_0p4()
    err_groups = [
        mod.aep.run_scale18_att_err(shared),
        mod.aep.run_plain24_att_err(shared),
        mod.aep.run_purescd_att_err(shared),
    ]
    state_groups = [
        mod.bp.trace_scale18_first3(shared),
        mod.bp.trace_plain24_first3(shared),
        mod.bp.trace_purescd24_first3(shared),
    ]
    sigma_map = {axis: mod.build_sigma_lookup(state_groups, state_label) for axis, state_label in AXIS_TO_STATE.items()}
    return err_groups, sigma_map


def build_rows(axis: str, err_groups, sigma_map):
    rows = []
    time_base = None
    for g in err_groups:
        if time_base is None:
            time_base = g['x']
    for i, t in enumerate(time_base):
        row = {col: '' for col in CSV_COLUMNS}
        row['time_s'] = t
        for g in err_groups:
            row[GROUP_TO_ERR_COL[g['group_key']]] = g[axis][i]
        for gk in GROUP_ORDER:
            row[GROUP_TO_SIGMA_COL[gk]] = sigma_map[axis][gk]['y'][i]
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) if isinstance(v, float) else v for k, v in row.items()})


def make_markdown(md_path: Path, csv_rel_paths: dict[str, str], svg_rel_paths: dict[str, str], all_rows: dict[str, list[dict]]):
    parts: list[str] = []
    parts.append('# seed=1 三个姿态对准误差滤波收敛图（SVG + 数据表）')
    parts.append('')
    parts.append('- noise: 0.4x custom noise')
    parts.append('- seed: 1')
    parts.append('- 图形布局：全程误差 / 末段放大 / 协方差 sigma')
    parts.append('- 曲线标签：G1baseline / G2MarkovMarkov / G3Markov+LLM+SCD')
    parts.append('')
    parts.append('## 文件清单')
    for axis in AXIS_TO_STATE:
        parts.append(f'- {AXIS_TITLES[axis]}：`{svg_rel_paths[axis]}`，数据 CSV：`{csv_rel_paths[axis]}`')
    parts.append('')
    parts.append('## 列说明')
    parts.append('| 列名 | 含义 |')
    parts.append('| --- | --- |')
    parts.append('| time_s | 时间 / 秒 |')
    parts.append('| G1baseline_err_arcsec | G1 baseline 的姿态对准误差（arcsec） |')
    parts.append('| G2MarkovMarkov_err_arcsec | G2 MarkovMarkov 的姿态对准误差（arcsec） |')
    parts.append('| G3Markov+LLM+SCD_err_arcsec | G3 Markov+LLM+SCD 的姿态对准误差（arcsec） |')
    parts.append('| G1baseline_sigma_arcsec | G1 对应 sigma(phi) 的等效 arcsec |')
    parts.append('| G2MarkovMarkov_sigma_arcsec | G2 对应 sigma(phi) 的等效 arcsec |')
    parts.append('| G3Markov+LLM+SCD_sigma_arcsec | G3 对应 sigma(phi) 的等效 arcsec |')
    parts.append('')

    for axis in AXIS_TO_STATE:
        rows = all_rows[axis]
        parts.append(f'## {AXIS_TITLES[axis]}')
        parts.append('')
        parts.append(f'- SVG：`{svg_rel_paths[axis]}`')
        parts.append(f'- CSV：`{csv_rel_paths[axis]}`')
        parts.append('')
        parts.append('| ' + ' | '.join(CSV_COLUMNS) + ' |')
        parts.append('|' + '|'.join([' --- '] * len(CSV_COLUMNS)) + '|')
        for row in rows:
            parts.append('| ' + ' | '.join(fmt(row[col]) if isinstance(row[col], float) else str(row[col]) for col in CSV_COLUMNS) + ' |')
        parts.append('')

    md_path.write_text('\n'.join(parts), encoding='utf-8')


def zip_dir(src_dir: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(src_dir.rglob('*')):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(src_dir)))


def main():
    mod = load_module('export_att_err_bundle_seed1_0p4_20260413', COMBO_SCRIPT)
    ensure_seed1_combo_outputs(mod)
    err_groups, sigma_map = build_series(mod)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_rel_paths = {}
    csv_rel_paths = {}
    all_rows = {}

    for axis in AXIS_TO_STATE:
        src_svg = SEED1_COMBO_DIR / f'{axis}_combo.svg'
        dst_svg = OUT_DIR / f'{axis}_combo.svg'
        shutil.copy2(src_svg, dst_svg)
        svg_rel_paths[axis] = dst_svg.name

        rows = build_rows(axis, err_groups, sigma_map)
        all_rows[axis] = rows
        csv_path = OUT_DIR / f'{axis}_data.csv'
        write_csv(csv_path, rows)
        csv_rel_paths[axis] = csv_path.name

    md_path = OUT_DIR / 'seed1_att_err_svg_and_tables.md'
    make_markdown(md_path, csv_rel_paths, svg_rel_paths, all_rows)

    summary = {
        'task': 'export_dualpath_three_method_att_err_svg_tables_bundle_seed1_0p4_2026_04_13',
        'seed': SEED,
        'noise': '0.4x custom noise',
        'bundle_dir': str(OUT_DIR),
        'markdown': str(md_path),
        'zip': str(ZIP_PATH),
        'files': sorted(p.name for p in OUT_DIR.iterdir()),
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    zip_dir(OUT_DIR, ZIP_PATH)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
