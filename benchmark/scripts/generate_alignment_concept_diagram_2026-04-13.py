#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path('/root/.openclaw/workspace/tmp/psins_repeatability')
OUT_PNG = OUT_DIR / 'fig_alignment_concept_diagram_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_alignment_concept_diagram_2026-04-13.svg'

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def add_box(ax, xy, w, h, title, body, fc='#F8FBFF', ec='#6D8FB3', title_color='#17324D', body_color='#22313F'):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                           linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + 0.02*w, y + h - 0.09*h, title, fontsize=13, fontweight='bold', color=title_color,
            va='top', ha='left')
    ax.text(x + 0.02*w, y + h - 0.24*h, body, fontsize=11, color=body_color,
            va='top', ha='left', linespacing=1.45)
    return patch


def arrow(ax, x1, y1, x2, y2, color='#6D8FB3', text=None, text_offset=(0, 0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.8, color=color, shrinkA=6, shrinkB=6))
    if text:
        mx = (x1 + x2) / 2 + text_offset[0]
        my = (y1 + y2) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=10.5, color=color, ha='center', va='center')


def main() -> None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(FONT)
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.4, 7.6), dpi=180)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.03, 0.965, '双轴对准里几个容易混淆的量：q_true、初始失准角、滤波 phi 与最终姿态误差',
            fontsize=17, fontweight='bold', ha='left', va='top', color='#102A43')
    ax.text(0.03, 0.928,
            '核心结论：初始失准角 φ0 是人为设置的起点误差；滤波状态 xk[0:3]=phi(t) 是随时间变化的残余失准角估计；'
            '最终统计的 roll / pitch / yaw 则是 q_est 与 q_true 重新做差得到的真实终值姿态误差。',
            fontsize=11.5, ha='left', va='top', color='#334E68')

    add_box(
        ax, (0.04, 0.68), 0.26, 0.18,
        '① 真值姿态轨迹 q_true(t)',
        '仿真中由 build_rot_paras() 明确生成\n初始真值姿态设为 att0 = [0, 0, 0]\n对所有 seed 相同，不带随机性',
        fc='#F1F8FF', ec='#5B8DB8'
    )
    add_box(
        ax, (0.37, 0.68), 0.26, 0.18,
        '② 初始失准角 φ0',
        '人为设定：φ0 = [0.1, 0.1, 0.5] deg\n即一开始估计姿态相对真值就有这组误差\n对准的目标是把它逐步压到接近 0',
        fc='#FFF7E8', ec='#D6A756', title_color='#6B4E16'
    )
    add_box(
        ax, (0.70, 0.68), 0.26, 0.18,
        '③ 初始估计姿态 q_est(0)',
        '由 q_est(0) = qaddphi(q_true(0), φ0) 得到\n所以“估计姿态”和“真值姿态”一开始就不相等\n这就是对准问题的起点',
        fc='#F7F3FF', ec='#8A6BBE', title_color='#38245C'
    )

    add_box(
        ax, (0.20, 0.38), 0.60, 0.20,
        '④ 滤波器内部状态：xk[0:3] = phi(t)',
        '它表示“当前还剩多少姿态失准没有被消掉”，是一个随时间变化的残余量，而不是固定常数。\n'
        '滤波更新后会用 0.91·phi 去修正姿态，再把剩余 0.09·phi 保留在状态里，因此 phi(t) 是内部残余失准角估计。\n'
        '对准收敛时，phi(t) 应该逐渐趋近于 0，而不是趋近于初始失准角 φ0。',
        fc='#EDFDF5', ec='#4E9F76', title_color='#0F5132'
    )

    add_box(
        ax, (0.04, 0.08), 0.28, 0.20,
        '⑤ 最终估计姿态 q_est(T)',
        '经过多轮旋转激励 + 滤波反馈后得到\n这是算法最终给出的姿态估计结果',
        fc='#F1F8FF', ec='#5B8DB8'
    )
    add_box(
        ax, (0.36, 0.08), 0.28, 0.20,
        '⑥ 最终姿态误差 φ_err(T)',
        '按 φ_err(T)=qq2phi(q_est(T), q_true(T)) 重新计算\n这才是正式结果口径里的“真实终值姿态误差”\n当前代码输出分量顺序是 [pitch, roll, yaw]',
        fc='#FFF7E8', ec='#D6A756', title_color='#6B4E16'
    )
    add_box(
        ax, (0.68, 0.08), 0.28, 0.20,
        '⑦ 图表里的 roll / pitch / yaw / norm',
        '它们来自 φ_err(T) 的统计：\nroll / pitch / yaw = 终值姿态误差分量\nnorm = ||φ_err(T)||\n1σ = 不同 seed 下终值真实误差的统计离散度',
        fc='#F7F3FF', ec='#8A6BBE', title_color='#38245C'
    )

    arrow(ax, 0.30, 0.77, 0.37, 0.77, text='给定参考姿态')
    arrow(ax, 0.63, 0.77, 0.70, 0.77, text='叠加初始失准角')
    arrow(ax, 0.83, 0.68, 0.64, 0.58, text='作为滤波初值', text_offset=(0.01, 0.02))
    arrow(ax, 0.50, 0.68, 0.50, 0.58, text='目标：逐步压小', text_offset=(0.08, 0.0))
    arrow(ax, 0.50, 0.38, 0.18, 0.28, text='得到最终姿态估计', text_offset=(-0.01, 0.02))
    arrow(ax, 0.18, 0.18, 0.36, 0.18, text='与真值姿态重新做差')
    arrow(ax, 0.64, 0.18, 0.68, 0.18, text='做统计')
    arrow(ax, 0.18, 0.68, 0.36, 0.18, text='提供 q_true(T)', text_offset=(-0.01, -0.01))

    note = FancyBboxPatch((0.03, 0.30), 0.94, 0.055, boxstyle='round,pad=0.01,rounding_size=0.015',
                          linewidth=1.3, edgecolor='#D9485F', facecolor='#FFF1F3')
    ax.add_patch(note)
    ax.text(0.05, 0.327,
            '真实实验里如果没有真值姿态 q_true(t)，就不能严格计算 φ_err(T)、roll/pitch/yaw 真值误差和 norm 真值误差；\n'
            '此时通常只能拿到滤波器内部的 phi(t)、协方差 1σ 或重复实验的估计值离散度，而不是严格真值误差。',
            fontsize=10.6, color='#7A1E2C', va='center', ha='left')

    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == '__main__':
    main()
