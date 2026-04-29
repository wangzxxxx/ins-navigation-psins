from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS = ROOT / 'psins_method_bench' / 'results'
REPORTS = ROOT / 'reports'
SOURCE = RESULTS / 'compare_four_group_progression_19_20_noise0p12.json'
OUTPUT = REPORTS / 'psins_g123_first_three_groups_fullparam_2026-04-05.md'

GROUPS = ['g1_kf19', 'g2_markov19', 'g3_markov20']
LABEL = {
    'g1_kf19': 'G1 普通模型@19位置',
    'g2_markov19': 'G2 Markov@19位置',
    'g3_markov20': 'G3 Markov@20位置',
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def fmt_e(x: float) -> str:
    return f'{x:.6e}'


def winner(a: float, b: float, ga: str, gb: str) -> str:
    if abs(a - b) <= 1e-12:
        return 'tie'
    return LABEL[ga] if a < b else LABEL[gb]


def build_pairwise(summary: dict):
    rows = []
    table = summary['all_params']['table']
    for ga, gb in itertools.combinations(GROUPS, 2):
        oa = next(x for x in summary['group_rows'] if x['group_key'] == ga)
        ob = next(x for x in summary['group_rows'] if x['group_key'] == gb)
        a_wins = 0
        b_wins = 0
        ties = 0
        for p, item in table.items():
            a = float(item['groups'][ga]['pct_error'])
            b = float(item['groups'][gb]['pct_error'])
            if abs(a - b) <= 1e-12:
                ties += 1
            elif a < b:
                a_wins += 1
            else:
                b_wins += 1
        rows.append({
            'ga': ga,
            'gb': gb,
            'mean_winner': winner(oa['mean_pct_error'], ob['mean_pct_error'], ga, gb),
            'median_winner': winner(oa['median_pct_error'], ob['median_pct_error'], ga, gb),
            'max_winner': winner(oa['max_pct_error'], ob['max_pct_error'], ga, gb),
            'delta_mean_a_minus_b': oa['mean_pct_error'] - ob['mean_pct_error'],
            'delta_median_a_minus_b': oa['median_pct_error'] - ob['median_pct_error'],
            'delta_max_a_minus_b': oa['max_pct_error'] - ob['max_pct_error'],
            'a_wins': a_wins,
            'b_wins': b_wins,
            'ties': ties,
        })
    return rows


def overall_row_map(summary: dict):
    return {x['group_key']: x for x in summary['group_rows']}


def render(summary: dict) -> str:
    rows = overall_row_map(summary)
    pairwise = build_pairwise(summary)
    params = summary['all_params']['param_order']
    table = summary['all_params']['table']
    cfg = summary['noise_config']
    groups = summary['groups']

    out = []
    out.append('<callout emoji="📎" background-color="light-blue">')
    out.append('这份文档只整理你最开始那四组实验里的**前三组**：`G1 普通模型@19位置 / G2 Markov@19位置 / G3 Markov@20位置`。')
    out.append('</callout>')
    out.append('')
    out.append('## 1. 实验范围')
    out.append('')
    out.append('- 本文**只包含前三组**，不包含第 4 组 `Markov + LLM + SCD @20位置`。')
    out.append('- 三组定义：')
    for g in GROUPS:
        out.append(f"  - **{LABEL[g]}**：{groups[g]['definition']}")
    out.append('')
    out.append('## 2. 统一实验口径')
    out.append('')
    out.append(f"- `noise_scale = {summary['noise_scale']}`")
    out.append(f"- `arw = {cfg['arw_dpsh']} dpsh`")
    out.append(f"- `vrw = {cfg['vrw_ugpsHz']} ugpsHz`")
    out.append(f"- `bi_g = {cfg['bi_g_dph']} dph`")
    out.append(f"- `bi_a = {cfg['bi_a_ug']} ug`")
    out.append(f"- `tau = {cfg['tau_g']}`")
    out.append(f"- `seed = {cfg['seed']}`")
    out.append('')
    out.append('## 3. Overall 指标总览（mean / median / max，相对误差%，越低越好）')
    out.append('')
    out.append('| 组别 | mean% | median% | max% |')
    out.append('|---|---:|---:|---:|')
    for g in GROUPS:
        r = rows[g]
        out.append(f"| {LABEL[g]} | {fmt_pct(r['mean_pct_error'])} | {fmt_pct(r['median_pct_error'])} | {fmt_pct(r['max_pct_error'])} |")
    out.append('')
    out.append('## 4. 三组方法两两对照')
    out.append('')
    out.append('| 对照 | mean 更优 | median 更优 | max 更优 | Δmean(左-右) | Δmedian(左-右) | Δmax(左-右) | 参数胜场(左/右/tie) |')
    out.append('|---|---|---|---|---:|---:|---:|---|')
    for r in pairwise:
        out.append(
            f"| {LABEL[r['ga']]} vs {LABEL[r['gb']]} | {r['mean_winner']} | {r['median_winner']} | {r['max_winner']} | "
            f"{r['delta_mean_a_minus_b']:+.6f} | {r['delta_median_a_minus_b']:+.6f} | {r['delta_max_a_minus_b']:+.6f} | "
            f"{r['a_wins']}/{r['b_wins']}/{r['ties']} |"
        )
    out.append('')
    out.append('## 5. 对照分析')
    out.append('')
    out.append(f"- **G1 vs G2**：Markov 在 19位置上对整体指标帮助非常有限；mean 和 max 甚至略差，只有 median 有极小改善。")
    out.append(f"- **G1 vs G3**：从 19位置普通模型切到 20位置 Markov 后，三项 overall 指标都明显下降，说明真正的大收益主要来自 **20位置路径 + Markov** 这一组合。")
    out.append(f"- **G2 vs G3**：这是最关键的一组对照。G3 相比 G2 在 mean / median / max 三项都明显更好，说明 **单纯在 19位置加 Markov 不够，换到 corrected symmetric20 路径才是主要提升来源**。")
    out.append('')
    out.append('## 6. 全参数数据表（设定值 / 各组估计值 / 相对误差%）')
    out.append('')
    out.append('| 参数 | 设定值 | G1 值 | G1 误差% | G2 值 | G2 误差% | G3 值 | G3 误差% |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for p in params:
        item = table[p]
        out.append(
            f"| {p} | {fmt_e(float(item['true']))} | "
            f"{fmt_e(float(item['groups']['g1_kf19']['est']))} | {fmt_pct(float(item['groups']['g1_kf19']['pct_error']))} | "
            f"{fmt_e(float(item['groups']['g2_markov19']['est']))} | {fmt_pct(float(item['groups']['g2_markov19']['pct_error']))} | "
            f"{fmt_e(float(item['groups']['g3_markov20']['est']))} | {fmt_pct(float(item['groups']['g3_markov20']['pct_error']))} |"
        )
    out.append('')
    out.append('## 7. 本页结论')
    out.append('')
    out.append('- 如果只看前三组，当前最优是 **G3 Markov@20位置**。')
    out.append('- `G1 -> G2` 没有形成明显提升；`G2 -> G3` 才是主要跃迁。')
    out.append('- 所以前三组的数据更支持：**改进的关键不在“19位置上加 Markov”本身，而在于切到 20位置 corrected symmetric 路径后再配合 Markov。**')
    out.append('')
    return '\n'.join(out) + '\n'


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = load_json(SOURCE)
    OUTPUT.write_text(render(summary), encoding='utf-8')
    print(str(OUTPUT))


if __name__ == '__main__':
    main()
