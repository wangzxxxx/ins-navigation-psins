from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS = ROOT / 'psins_method_bench' / 'results'
REPORTS = ROOT / 'reports'

CAUSAL_SUMMARY = RESULTS / 'causal_decompose_shared0p08_to_sym20_2026-04-05_summary.json'
OUTPUT_MD = REPORTS / 'psins_three_experiments_fullparam_pairwise_2026-04-05.md'

METHOD_ORDER = ['kf36', 'markov42', 'pure_scd_neutral', 'round61']
METHOD_LABEL = {
    'kf36': 'KF36',
    'markov42': 'Markov42',
    'pure_scd_neutral': 'Pure SCD',
    'round61': 'Round61',
}
CONDITION_LABEL = {
    'shared0p08': '实验 A：original shared path @ noise0p08',
    'sym20_0p08': '实验 B：corrected symmetric20 @ noise0p08',
    'sym20_0p12': '实验 C：corrected symmetric20 @ noise0p12',
}
CONDITION_NOTE = {
    'shared0p08': '原始 shared 数据集；方法口径与旧飞书文档一致。',
    'sym20_0p08': '只替换为 corrected symmetric20 路径，保持 noise0p08，用于隔离路径效应。',
    'sym20_0p12': '在 corrected symmetric20 路径上把 noise 从 0.08 提到 0.12，用于隔离噪声效应。',
}
PAIRWISE_FOCUS = [
    ('markov42', 'pure_scd_neutral'),
    ('markov42', 'round61'),
    ('pure_scd_neutral', 'round61'),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def fmt_e(x: float) -> str:
    return f'{x:.6e}'


def metric_winner(v1: float, v2: float, left: str, right: str) -> str:
    if abs(v1 - v2) <= 1e-12:
        return 'tie'
    return METHOD_LABEL[left] if v1 < v2 else METHOD_LABEL[right]


def overall_triplet(payload: dict) -> dict:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def pairwise_rows(payloads: dict[str, dict], param_order: list[str]):
    rows = []
    for left, right in itertools.combinations(METHOD_ORDER, 2):
        l_ov = overall_triplet(payloads[left])
        r_ov = overall_triplet(payloads[right])
        left_wins = 0
        right_wins = 0
        ties = 0
        for p in param_order:
            l = float(payloads[left]['param_errors'][p]['pct_error'])
            r = float(payloads[right]['param_errors'][p]['pct_error'])
            if abs(l - r) <= 1e-12:
                ties += 1
            elif l < r:
                left_wins += 1
            else:
                right_wins += 1
        rows.append({
            'left': left,
            'right': right,
            'mean_winner': metric_winner(l_ov['mean_pct_error'], r_ov['mean_pct_error'], left, right),
            'median_winner': metric_winner(l_ov['median_pct_error'], r_ov['median_pct_error'], left, right),
            'max_winner': metric_winner(l_ov['max_pct_error'], r_ov['max_pct_error'], left, right),
            'delta_mean_left_minus_right': l_ov['mean_pct_error'] - r_ov['mean_pct_error'],
            'delta_median_left_minus_right': l_ov['median_pct_error'] - r_ov['median_pct_error'],
            'delta_max_left_minus_right': l_ov['max_pct_error'] - r_ov['max_pct_error'],
            'left_param_wins': left_wins,
            'right_param_wins': right_wins,
            'ties': ties,
        })
    return rows


def key_pair_notes(rows: list[dict]) -> list[str]:
    out = []
    row_map = {(r['left'], r['right']): r for r in rows}
    for left, right in PAIRWISE_FOCUS:
        r = row_map[(left, right)]
        out.append(
            f"- **{METHOD_LABEL[left]} vs {METHOD_LABEL[right]}**："
            f"mean 胜者={r['mean_winner']}，median 胜者={r['median_winner']}，max 胜者={r['max_winner']}；"
            f"参数胜场 {METHOD_LABEL[left]}={r['left_param_wins']} / {METHOD_LABEL[right]}={r['right_param_wins']} / tie={r['ties']}。"
        )
    return out


def render_overall_matrix(causal: dict) -> list[str]:
    lines = []
    lines.append('| 实验 | KF36 | Markov42 | Pure SCD | Round61 |')
    lines.append('|---|---|---|---|---|')
    for cond in causal['conditions_order']:
        m = causal['matrix'][cond]
        lines.append(
            f"| {CONDITION_LABEL[cond]} | "
            f"{fmt_pct(m['kf36']['overall']['mean_pct_error'])} / {fmt_pct(m['kf36']['overall']['median_pct_error'])} / {fmt_pct(m['kf36']['overall']['max_pct_error'])} | "
            f"{fmt_pct(m['markov42']['overall']['mean_pct_error'])} / {fmt_pct(m['markov42']['overall']['median_pct_error'])} / {fmt_pct(m['markov42']['overall']['max_pct_error'])} | "
            f"{fmt_pct(m['pure_scd_neutral']['overall']['mean_pct_error'])} / {fmt_pct(m['pure_scd_neutral']['overall']['median_pct_error'])} / {fmt_pct(m['pure_scd_neutral']['overall']['max_pct_error'])} | "
            f"{fmt_pct(m['round61']['overall']['mean_pct_error'])} / {fmt_pct(m['round61']['overall']['median_pct_error'])} / {fmt_pct(m['round61']['overall']['max_pct_error'])} |"
        )
    return lines


def render_pairwise_table(rows: list[dict]) -> list[str]:
    lines = []
    lines.append('| 方法对 | mean 更优 | median 更优 | max 更优 | Δmean(左-右) | Δmedian(左-右) | Δmax(左-右) | 参数胜场(左/右/tie) |')
    lines.append('|---|---|---|---|---:|---:|---:|---|')
    for r in rows:
        lines.append(
            f"| {METHOD_LABEL[r['left']]} vs {METHOD_LABEL[r['right']]} | {r['mean_winner']} | {r['median_winner']} | {r['max_winner']} | "
            f"{r['delta_mean_left_minus_right']:+.6f} | {r['delta_median_left_minus_right']:+.6f} | {r['delta_max_left_minus_right']:+.6f} | "
            f"{r['left_param_wins']}/{r['right_param_wins']}/{r['ties']} |"
        )
    return lines


def render_param_table(payloads: dict[str, dict], param_order: list[str]) -> list[str]:
    lines = []
    lines.append('| 参数 | 设定值 | KF36 值 | KF36 误差% | Markov 值 | Markov 误差% | Pure SCD 值 | Pure SCD 误差% | Round61 值 | Round61 误差% |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for p in param_order:
        ref = float(payloads['kf36']['param_errors'][p]['true'])
        lines.append(
            f"| {p} | {fmt_e(ref)} | "
            f"{fmt_e(float(payloads['kf36']['param_errors'][p]['est']))} | {fmt_pct(float(payloads['kf36']['param_errors'][p]['pct_error']))} | "
            f"{fmt_e(float(payloads['markov42']['param_errors'][p]['est']))} | {fmt_pct(float(payloads['markov42']['param_errors'][p]['pct_error']))} | "
            f"{fmt_e(float(payloads['pure_scd_neutral']['param_errors'][p]['est']))} | {fmt_pct(float(payloads['pure_scd_neutral']['param_errors'][p]['pct_error']))} | "
            f"{fmt_e(float(payloads['round61']['param_errors'][p]['est']))} | {fmt_pct(float(payloads['round61']['param_errors'][p]['pct_error']))} |"
        )
    return lines


def best_summary(causal: dict, cond: str) -> list[str]:
    best = causal['rankings'][cond]['best_by_metric']
    r61rank = causal['rankings'][cond]['round61_rank']
    return [
        f"- **best mean**：{METHOD_LABEL[best['mean_pct_error']['method']]}（{fmt_pct(best['mean_pct_error']['value'])}）",
        f"- **best median**：{METHOD_LABEL[best['median_pct_error']['method']]}（{fmt_pct(best['median_pct_error']['value'])}）",
        f"- **best max**：{METHOD_LABEL[best['max_pct_error']['method']]}（{fmt_pct(best['max_pct_error']['value'])}）",
        f"- **Round61 排名**：mean 第 {r61rank['mean_pct_error']}，median 第 {r61rank['median_pct_error']}，max 第 {r61rank['max_pct_error']}。",
    ]


def build_markdown() -> str:
    causal = load_json(CAUSAL_SUMMARY)
    condition_payloads = {}
    for cond, paths in causal['condition_result_jsons'].items():
        condition_payloads[cond] = {m: load_json(Path(path)) for m, path in paths.items()}

    param_order = condition_payloads['shared0p08']['kf36']['param_order']
    lines: list[str] = []
    lines.append('<callout emoji="📘" background-color="light-blue">')
    lines.append('按你的要求，把之前三组实验统一整理为：**全参数数据表 + 不同标定方法两两对照分析**。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. 文档说明')
    lines.append('')
    lines.append('- 三组实验分别是：')
    lines.append('  1. `shared noise0p08`')
    lines.append('  2. `corrected symmetric20 + noise0p08`')
    lines.append('  3. `corrected symmetric20 + noise0p12`')
    lines.append('- 每组实验的标定方法固定为：`KF36 / Markov42 / Pure SCD neutral / Round61`。')
    lines.append('- 下面每组都包含：')
    lines.append('  - 全参数设定值 / 估计值 / 相对误差%')
    lines.append('  - 方法两两对照结果')
    lines.append('  - 该组实验的简短分析')
    lines.append('')
    lines.append('## 2. 三组实验 overall 总览（mean / median / max %）')
    lines.append('')
    lines.extend(render_overall_matrix(causal))
    lines.append('')
    lines.append('## 3. 总体拆因结论')
    lines.append('')
    lines.append(f"- 这三组实验的主结论：**{causal['causal_verdict']['label']}** 是主要因素。")
    lines.append(f"- 原因：{causal['causal_verdict']['reason']}")
    lines.append('')

    for cond in causal['conditions_order']:
        payloads = condition_payloads[cond]
        pair_rows = pairwise_rows(payloads, param_order)
        lines.append(f"## {CONDITION_LABEL[cond]}")
        lines.append('')
        lines.append(f"- {CONDITION_NOTE[cond]}")
        lines.extend(best_summary(causal, cond))
        lines.append('')
        lines.append('### 3.1 方法两两对照')
        lines.append('')
        lines.extend(render_pairwise_table(pair_rows))
        lines.append('')
        lines.append('### 3.2 本组实验分析')
        lines.append('')
        lines.extend(key_pair_notes(pair_rows))
        lines.append('')
        lines.append('### 3.3 全参数数据表')
        lines.append('')
        lines.extend(render_param_table(payloads, param_order))
        lines.append('')
        lines.append('---')
        lines.append('')

    lines.append('## 4. 汇总判断')
    lines.append('')
    lines.append('- 在 **shared0p08** 上，Round61 仍然是最强的整体方法（mean / median 最优），说明旧实验结果本身是可复现的。')
    lines.append('- 一旦切换到 **corrected symmetric20** 路径，Round61 的角色发生变化：更像是压 `max` 的方法，而不再是压 `mean / median` 的方法。')
    lines.append('- 当路径固定在 `corrected symmetric20` 再把噪声从 `0.08` 提到 `0.12` 时，Round61 的 mean 优势并没有回来，因此旧效果的消失主要不是由噪声单独造成。')
    lines.append('- 所以，后续如果要为 `corrected symmetric20` 做原生策略，应该优先从 **sym20-native base** 重建，而不是继续直接迁移 shared 路径上的 old Round61。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(build_markdown(), encoding='utf-8')
    print(str(OUTPUT_MD))


if __name__ == '__main__':
    main()
