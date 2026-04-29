import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_align_rotation_2axis.glv_utils import glv
from py_align_rotation_2axis.nav_utils import posset
from py_align_rotation_2axis.math_utils import a2qua, qaddphi, q2att, qq2phi
from py_align_rotation_2axis.imu_utils import attrottt, avp2imu, imuerrset, imuadderr, setdiag
from py_align_rotation_2axis.align_utils import alignvn_dar_12state


def compute_phase_times(paras, ts):
    """
    Compute cumulative time boundaries for each stage in paras.
    Returns a list of (stage_id, t_start, t_end) tuples.
    """
    boundaries = []
    t_cur = 0.0
    prev_id = paras[0, 0]
    t_stage_start = 0.0

    for row in paras:
        stage_id = row[0]
        angle = row[4]   # Already in radians
        T_rot = row[5]
        T0 = row[6]
        T1 = row[7]
        duration = T0 + T_rot + T1
        if stage_id != prev_id:
            boundaries.append((prev_id, t_stage_start, t_cur))
            t_stage_start = t_cur
            prev_id = stage_id
        t_cur += duration
    boundaries.append((prev_id, t_stage_start, t_cur))
    return boundaries


def plot_alignment_results(xkpk_all_iters, phase_boundaries, out_dir, truth_att0):
    """
    Generate detailed SVG convergence plots for dual-axis rotation alignment.
    xkpk_all_iters: list of xkpk arrays, one per iteration.
    Each xkpk row: [xk(12) | diag(Pxk)(12) | t]  -> 25 columns
    """
    os.makedirs(out_dir, exist_ok=True)

    # Concatenate all iterations for plotting, offset time per iteration
    colors_iters = ['blue', 'green', 'orange', 'red', 'purple']
    iter_labels = [f'Iter {i+1}' for i in range(len(xkpk_all_iters))]

    def draw_phase_lines(ax, boundaries, t_offset=0.0):
        """Draw vertical dashed lines at phase transitions."""
        prev_id = None
        label_done = set()
        for (sid, ts_start, ts_end) in boundaries:
            t = ts_start + t_offset
            if t > 0:
                ax.axvline(x=t, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
            if sid not in label_done:
                mid_t = (ts_start + ts_end) / 2 + t_offset
                ax.text(mid_t, ax.get_ylim()[1], f'S{int(sid)}',
                        fontsize=6, color='gray', ha='center', va='top')
                label_done.add(sid)

    def plot_panel(fig_ax, time_arr, vals, std_arr, unit_scale, unit_str,
                   ylabel, title, boundaries, color, iter_label,
                   truth_line=None):
        ax = fig_ax
        ax.plot(time_arr, vals * unit_scale, color=color, label=f'{iter_label} est', linewidth=1.2)
        ax.fill_between(time_arr,
                        (vals - std_arr) * unit_scale,
                        (vals + std_arr) * unit_scale,
                        alpha=0.2, color=color)
        if truth_line is not None:
            ax.axhline(y=truth_line * unit_scale, color='black', linestyle=':', linewidth=1.0, label='Truth')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)

    # ----------------------------------------------------------------
    # Figure layout: 6 rows x 2 cols
    #   Row 0: phi_E (misalignment E)
    #   Row 1: phi_N (misalignment N)
    #   Row 2: phi_U (misalignment U / heading)
    #   Row 3: eb_x, eb_y, eb_z (gyro bias, combined)
    #   Row 4: db_x, db_y, db_z (acc bias, combined)
    # ----------------------------------------------------------------
    state_labels = [
        r'$\phi_E$', r'$\phi_N$', r'$\phi_U$',
        r'$\delta v_E$', r'$\delta v_N$', r'$\delta v_U$',
        r'$eb_x$', r'$eb_y$', r'$eb_z$',
        r'$db_x$', r'$db_y$', r'$db_z$'
    ]
    unit_scales = [1/glv.min]*3 + [1.0]*3 + [1/glv.dph]*3 + [1/glv.ug]*3
    unit_labels = ["arcmin"]*3 + ["m/s"]*3 + ["dph"]*3 + ["ug"]*3

    phase_names = {1: 'Z↑ rot', 2: 'Flip', 3: 'X↑ rot', 4: 'Flip', 5: 'Z↓ rot', 6: 'Flip', 7: 'Reset'}
    phase_colors = ['#4472C4','#ED7D31','#A9D18E','#FF0000','#7030A0','#00B0F0','#C00000']

    # ---- Per-state dual-panel plots (estimate + uncertainty) ----
    for si in range(12):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
        fig.suptitle(f"State [{si}]: {state_labels[si]}", fontsize=13)

        t_global_offset = 0.0
        for it_idx, xkpk in enumerate(xkpk_all_iters):
            t = xkpk[:, 24] + t_global_offset
            val = xkpk[:, si]              # State estimate
            std = np.sqrt(np.abs(xkpk[:, 12 + si]))  # Std dev from diag(P)
            color = colors_iters[it_idx % len(colors_iters)]

            # Top panel: state estimate
            axes[0].plot(t, val * unit_scales[si], color=color,
                         label=iter_labels[it_idx], linewidth=1.2)
            # Bottom panel: uncertainty (std dev)
            axes[1].plot(t, std * unit_scales[si], color=color,
                         label=iter_labels[it_idx], linewidth=1.2)

            # Phase boundary lines
            for (sid, ts_s, ts_e) in phase_boundaries:
                vt = ts_s + t_global_offset
                if vt > t_global_offset:
                    for ax in axes:
                        ax.axvline(x=vt, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            # Iteration boundary
            if it_idx > 0:
                for ax in axes:
                    ax.axvline(x=t_global_offset, color='red', linestyle='-', alpha=0.4, linewidth=1.0)

            t_global_offset = t[-1]

        axes[0].set_ylabel(f"Estimate ({unit_labels[si]})")
        axes[0].set_title("State Estimate (per Iteration)")
        axes[0].legend(fontsize=8, ncol=5)
        axes[0].grid(True, alpha=0.4)

        axes[1].set_ylabel(f"Std Dev ({unit_labels[si]})")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Uncertainty (±1σ)")
        axes[1].legend(fontsize=8, ncol=5)
        axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        safe_label = state_labels[si].replace('$', '').replace('\\', '').replace('{', '').replace('}', '').replace(' ', '_')
        plt.savefig(os.path.join(out_dir, f"{si:02d}_{safe_label}.svg"), format='svg')
        plt.close()

    # ---- Combined misalignment plot ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    phi_names = [r'$\phi_E$ (arcmin)', r'$\phi_N$ (arcmin)', r'$\phi_U$ (arcmin)']
    t_global_offset = 0.0
    for it_idx, xkpk in enumerate(xkpk_all_iters):
        t = xkpk[:, 24] + t_global_offset
        color = colors_iters[it_idx % len(colors_iters)]
        for pi in range(3):
            val = xkpk[:, pi] / glv.min
            std = np.sqrt(np.abs(xkpk[:, 12 + pi])) / glv.min
            axes[pi].plot(t, val, color=color, label=iter_labels[it_idx], linewidth=1.2)
            axes[pi].fill_between(t, val - std, val + std, alpha=0.15, color=color)
        for (sid, ts_s, ts_e) in phase_boundaries:
            vt = ts_s + t_global_offset
            if vt > t_global_offset:
                for ax in axes:
                    ax.axvline(x=vt, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if it_idx > 0:
            for ax in axes:
                ax.axvline(x=t_global_offset, color='red', linestyle='-', alpha=0.4, linewidth=1.0)
        t_global_offset = t[-1]
    for pi in range(3):
        axes[pi].set_ylabel(phi_names[pi])
        axes[pi].legend(fontsize=8, ncol=5)
        axes[pi].grid(True, alpha=0.4)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Misalignment Angles Convergence (All Iterations)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "00_misalignment_overview.svg"), format='svg')
    plt.close()

    # ---- Combined bias plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    bias_names = [r'Gyro Bias $eb$ (dph)', r'Acc Bias $db$ (ug)']
    bias_scales = [1/glv.dph, 1/glv.ug]
    bias_offsets = [[6, 7, 8], [9, 10, 11]]
    bias_colors_xyz = ['red', 'green', 'blue']
    t_global_offset = 0.0
    for it_idx, xkpk in enumerate(xkpk_all_iters):
        t = xkpk[:, 24] + t_global_offset
        color = colors_iters[it_idx % len(colors_iters)]
        for bi, (b_idxs, bscale) in enumerate(zip(bias_offsets, bias_scales)):
            for ci, sidx in enumerate(b_idxs):
                label = f'{iter_labels[it_idx]} {"xyz"[ci]}' if bi == 0 else None
                axes[bi].plot(t, xkpk[:, sidx] * bscale,
                              color=bias_colors_xyz[ci],
                              linestyle=['-', '--', ':'][it_idx % 3],
                              label=label, linewidth=1.0)
        for (sid, ts_s, ts_e) in phase_boundaries:
            vt = ts_s + t_global_offset
            if vt > t_global_offset:
                for ax in axes:
                    ax.axvline(x=vt, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if it_idx > 0:
            for ax in axes:
                ax.axvline(x=t_global_offset, color='red', linestyle='-', alpha=0.4, linewidth=1.0)
        t_global_offset = t[-1]
    for bi in range(2):
        axes[bi].set_ylabel(bias_names[bi])
        axes[bi].legend(fontsize=7, ncol=6)
        axes[bi].grid(True, alpha=0.4)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Gyro & Acc Bias Estimation (All Iterations)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_bias_overview.svg"), format='svg')
    plt.close()

    print(f"Saved all plots to '{out_dir}/'")


def main():
    ts = 0.01

    att0 = np.array([0, 0, 0]) * glv.deg
    pos0 = posset(34, 116, 480)

    paras = np.array([
        # Z轴竖直
        [1, 0,0,1,  720*glv.deg, 30, 0, 0],
        [1, 0,0,1, -720*glv.deg, 30, 0, 0],
        [1, 0,0,1,    0*glv.deg, 10, 0, 0],
        # 翻转90度
        [2, 0,1,0,   90*glv.deg, 10, 0, 0],
        # X轴竖直 (原Z轴水平)
        [3, 1,0,0,  720*glv.deg, 30, 0, 0],
        [3, 1,0,0, -720*glv.deg, 30, 0, 0],
        [3, 1,0,0,    0*glv.deg, 10, 0, 0],
        # 翻转90度
        [4, 0,1,0,   90*glv.deg, 10, 0, 0],
        # Z轴倒立
        [5, 0,0,-1, 720*glv.deg, 30, 0, 0],
        [5, 0,0,-1,-720*glv.deg, 30, 0, 0],
        [5, 0,0,-1,    0*glv.deg, 10, 0, 0],
        # # 翻转90度
        # [6, 0,1,0,   90*glv.deg, 10, 0, 0],
        # # 复位
        # [7, 0,1,0,   90*glv.deg, 10, 0, 0],
        # [7, 0,0,1,    0*glv.deg, 100, 0, 0]
    ], dtype=float)

    print("Generating attitude data...")
    att = attrottt(att0, paras, ts)
    length = att.shape[0]
    avp = np.zeros((length, 10))
    avp[:, 0:3] = att[:, 0:3]
    avp[:, 6:9] = np.tile(pos0, (length, 1))
    avp[:, 9] = att[:, 3]

    print("Generating pure IMU sequence...")
    imu, _ = avp2imu(avp)

    print("Setting and injecting IMU errors...")
    imuerr = imuerrset(0.01, 100.0, 0.0001, 1.0)
    imuerr['dKg'] = setdiag(imuerr['dKg'], 30 * glv.ppm)
    imuerr['dKa'] = setdiag(imuerr['dKa'], 30 * glv.ppm)
    imu2 = imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = q2att(qaddphi(a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    # Run alignment, collecting xkpk per iteration
    from py_align_rotation_2axis.align_utils import alignvn_dar_12state_multiiter
    max_iter = 3
    print("Running dual-axis rotation alignment (capturing all iterations)...")
    att0_aligned, xkpk_all_iters = alignvn_dar_12state_multiiter(
        imu2, att0_guess, pos0, phi, imuerr, wvn, max_iter=max_iter
    )

    # Compute phase boundaries
    phase_boundaries = compute_phase_times(paras, ts)
    print("Phase boundaries computed:")
    for (sid, ts_s, ts_e) in phase_boundaries:
        print(f"  Stage {int(sid)}: {ts_s:.1f}s ~ {ts_e:.1f}s")

    # Generate plots
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'plots_align_svg')
    plot_alignment_results(xkpk_all_iters, phase_boundaries, out_dir, att0)

    # Result evaluation
    phi_err = qq2phi(a2qua(att0_aligned), a2qua(avp[-1, 0:3]))
    print(f"\nFinal Attitude Alignment Error (arcsec):")
    print(f"  phi_E = {phi_err[0]/glv.sec:.2f}\"")
    print(f"  phi_N = {phi_err[1]/glv.sec:.2f}\"")
    print(f"  phi_U = {phi_err[2]/glv.sec:.2f}\"")


if __name__ == "__main__":
    main()
