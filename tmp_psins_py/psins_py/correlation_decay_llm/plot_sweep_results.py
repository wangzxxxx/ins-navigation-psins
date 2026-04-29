"""
plot_sweep_results.py
---------------------
Plot the alpha and transition_duration sweep results.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
import numpy as np
import os

# Data from sweep
alphas     = [0.93, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
alpha_wins = [20,   20,   20,   20,   20,   20,    20,   20,    21,   20]
alpha_errs = [7.03, 6.99, 6.95, 6.88, 6.83, 6.80, 6.78, 6.76,  6.77, 6.85]

tds       = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
td_wins   = [20,  20,  20,  21,  21,  21,  21,  20]
td_errs   = [6.96,6.90,6.82,6.77,6.75,6.75,6.75,6.72]

out_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Figure 1: Alpha Sweep ───
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('SCD Parameter Sweep: Alpha (Decay Rate)', fontsize=16, fontweight='bold')

# Wins
ax1.bar(range(len(alphas)), alpha_wins, color=['#2ecc71' if w==21 else '#3498db' for w in alpha_wins],
        edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Wins / 30', fontsize=13)
ax1.set_ylim(15, 22)
ax1.axhline(y=15, color='red', ls='--', lw=1.5, alpha=0.7, label='Standard KF (15/30 = tie)')
ax1.axhline(y=20, color='gray', ls=':', lw=1, alpha=0.5)
ax1.legend(fontsize=10)
ax1.set_title('Win Count vs Alpha', fontsize=13)
for i, (a, w) in enumerate(zip(alphas, alpha_wins)):
    ax1.text(i, w + 0.15, str(w), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Average Error
ax2.plot(range(len(alphas)), alpha_errs, 'o-', color='#e74c3c', lw=2.5, markersize=8)
ax2.fill_between(range(len(alphas)), alpha_errs, alpha=0.15, color='#e74c3c')
ax2.set_ylabel('Average Error %', fontsize=13)
ax2.set_xlabel('Alpha (Decay Rate)', fontsize=13)
ax2.set_xticks(range(len(alphas)))
ax2.set_xticklabels([f'{a:.3f}' for a in alphas], fontsize=10)
ax2.set_title('Average Parameter Error vs Alpha', fontsize=13)

# Annotate minimum
min_idx = np.argmin(alpha_errs)
ax2.annotate(f'{alpha_errs[min_idx]:.2f}%', xy=(min_idx, alpha_errs[min_idx]),
             xytext=(min_idx-1.5, alpha_errs[min_idx]-0.15),
             arrowprops=dict(arrowstyle='->', color='#e74c3c'),
             fontsize=12, fontweight='bold', color='#e74c3c')

for ax in [ax1, ax2]:
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sweep_alpha.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'sweep_alpha.png'), format='png', dpi=150, bbox_inches='tight')
print("Saved sweep_alpha.svg/png")

# ─── Figure 2: Transition Duration Sweep ───
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('SCD Parameter Sweep: Transition Duration (alpha=0.99)', fontsize=16, fontweight='bold')

ax1.bar(range(len(tds)), td_wins, color=['#2ecc71' if w==21 else '#3498db' for w in td_wins],
        edgecolor='white', linewidth=1.5)
ax1.set_ylabel('Wins / 30', fontsize=13)
ax1.set_ylim(15, 22)
ax1.axhline(y=15, color='red', ls='--', lw=1.5, alpha=0.7, label='Standard KF (15/30 = tie)')
ax1.legend(fontsize=10)
ax1.set_title('Win Count vs Transition Duration', fontsize=13)
for i, (t, w) in enumerate(zip(tds, td_wins)):
    ax1.text(i, w + 0.15, str(w), ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.plot(range(len(tds)), td_errs, 's-', color='#9b59b6', lw=2.5, markersize=8)
ax2.fill_between(range(len(tds)), td_errs, alpha=0.15, color='#9b59b6')
ax2.set_ylabel('Average Error %', fontsize=13)
ax2.set_xlabel('Transition Duration (seconds)', fontsize=13)
ax2.set_xticks(range(len(tds)))
ax2.set_xticklabels([f'{t:.1f}' for t in tds], fontsize=10)
ax2.set_title('Average Parameter Error vs Transition Duration', fontsize=13)

for ax in [ax1, ax2]:
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sweep_trans_dur.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'sweep_trans_dur.png'), format='png', dpi=150, bbox_inches='tight')
print("Saved sweep_trans_dur.svg/png")

# ─── Figure 3: Combined Heatmap-style Summary ───
fig, ax = plt.subplots(figsize=(14, 4))
fig.suptitle('SCD Method: Complete Parameter Landscape', fontsize=16, fontweight='bold')

# Create a combined bar chart
labels = [f'a={a}' for a in alphas] + [''] + [f'td={t}s' for t in tds]
values = alpha_wins + [0] + td_wins
colors = []
for v in alpha_wins:
    colors.append('#2ecc71' if v == 21 else '#3498db')
colors.append('white')
for v in td_wins:
    colors.append('#2ecc71' if v == 21 else '#f39c12')

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1)
ax.set_ylim(0, 24)
ax.set_ylabel('Wins / 30', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.axhline(y=15, color='red', ls='--', lw=1.5, alpha=0.7, label='Tie line (15/30)')
ax.axhline(y=20, color='gray', ls=':', lw=1, alpha=0.5, label='Previous best (20/30)')
ax.legend(fontsize=10)

# Add vertical separator
sep_x = len(alphas)
ax.axvline(x=sep_x, color='black', ls='-', lw=2, alpha=0.3)
ax.text(len(alphas)/2, 23, 'Alpha Sweep\n(td=2.0s)', ha='center', fontsize=11, style='italic')
ax.text(len(alphas) + 1 + len(tds)/2, 23, 'TD Sweep\n(a=0.99)', ha='center', fontsize=11, style='italic')

for i, v in enumerate(values):
    if v > 0:
        ax.text(i, v + 0.3, str(v), ha='center', fontsize=10, fontweight='bold')

ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'sweep_combined.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'sweep_combined.png'), format='png', dpi=150, bbox_inches='tight')
print("Saved sweep_combined.svg/png")

plt.close('all')
print("\nAll plots saved!")
