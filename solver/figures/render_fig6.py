"""
Render the manuscript's Fig. 6 (Section III-A, "Spatially-Resolved Pressure
and Flow Along the Reconstructed Geometry"): a single representative
cardiac-cycle phase, both umbilical arteries coloured by the spatially
resolved 1D solution, UV shown at a single uniform colour from the lumped
Windkessel relation, cardiac-cycle insets marking the plotted phase.

Field values are mapped onto the rendered synthetic UA paths by normalized
arc-length fraction (fraction of the solver's L_cm and fraction of each
path's own total length), not absolute cm, so a small mismatch between the
solver's calibrated segment length (72.2 cm, Section III-B) and any one
synthetic path's actual arc length does not truncate or misalign the field.
"""
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

GEOM = os.environ.get("UMBILICAL_GEOMETRY_DIR", str(Path(__file__).resolve().parents[2] / "umbilical-geometry" / "geometry"))
HERE = str(Path(__file__).resolve().parent)

ff = np.load(f"{HERE}\\full_field_GA25.npz")
uv_pos = np.load(f"{GEOM}\\uv_centerline.npy")
ua1_pos = np.load(f"{HERE}\\ga25_ua1_path.npy")
ua2_pos = np.load(f"{HERE}\\ga25_ua2_path.npy")

x_cm = ff["x_cm"]; L_cm = float(ff["L_cm"])
A = ff["A"]; q = ff["q"]; Pwk = ff["Pwk"]
A0 = float(ff["A0"]); Eh_r0 = float(ff["Eh_r0"]); p0 = float(ff["p0"])
R2 = float(ff["R2"]); Pv = float(ff["Pv"])
phase_t = ff["phase_t_s"]; T = float(ff["T_s"])

MMHG = 1.0 / 1333.22
p_field = (p0 + (4.0 / 3.0) * Eh_r0 * (1.0 - np.sqrt(A0 / A))) * MMHG  # (n_phases, M) mmHg
qven = (Pwk - Pv) / R2  # mL/s
Pwk_mmHg = Pwk * MMHG

spread = p_field.max(axis=1) - p_field.min(axis=1)
idx = int(np.argmax(spread))
pct = 100.0 * phase_t[idx] / T
t_ms = phase_t[idx] * 1000

x_frac = x_cm / L_cm


def arc_frac(pos):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    return s / s[-1]


ua1_frac = arc_frac(ua1_pos)
ua2_frac = arc_frac(ua2_pos)
p_on_ua1 = np.interp(ua1_frac, x_frac, p_field[idx]); q_on_ua1 = np.interp(ua1_frac, x_frac, q[idx])
p_on_ua2 = np.interp(ua2_frac, x_frac, p_field[idx]); q_on_ua2 = np.interp(ua2_frac, x_frac, q[idx])

cmap_p = plt.get_cmap("turbo")
cmap_q = plt.get_cmap("turbo")
norm_p = plt.Normalize(min(p_on_ua1.min(), p_on_ua2.min()), max(Pwk_mmHg[idx], p_on_ua1.max(), p_on_ua2.max()))
norm_q = plt.Normalize(min(q_on_ua1.min(), q_on_ua2.min(), qven[idx]), max(q_on_ua1.max(), q_on_ua2.max()))

VIEW = dict(elev=12, azim=50)


def style_axis3d(ax, pts):
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
    ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.grid(False)
    ax.view_init(**VIEW)


def draw_panel(fig, rect3d, rect_inset, cmap, norm, ua1_field, ua2_field, uv_scalar,
               cycle_curve, title, cbar_label, inset_ylabel):
    all_pts = np.vstack([uv_pos, ua1_pos, ua2_pos])
    ax = fig.add_axes(rect3d, projection="3d")

    uv_color = cmap(norm(uv_scalar))
    ax.plot(uv_pos[:, 0], uv_pos[:, 1], uv_pos[:, 2], color=uv_color,
            linewidth=5.5, solid_capstyle="round", zorder=3)

    for pos, field in ((ua1_pos, ua1_field), (ua2_pos, ua2_field)):
        segs = np.stack([pos[:-1], pos[1:]], axis=1)
        seg_val = 0.5 * (field[:-1] + field[1:])
        lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=4.0, zorder=4)
        lc.set_array(seg_val)
        ax.add_collection3d(lc)

    style_axis3d(ax, all_pts)
    ax.set_title(title, fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=18)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    axi = fig.add_axes(rect_inset)
    axi.plot(phase_t * 1000, cycle_curve, color="#444444", linewidth=1.3)
    axi.plot(t_ms, cycle_curve[idx], "o", color="#d1495b", markersize=6, zorder=5)
    axi.set_xlabel("t (ms)", fontsize=8, labelpad=1)
    axi.set_ylabel(inset_ylabel, fontsize=8, labelpad=1)
    axi.tick_params(labelsize=7, pad=1)
    axi.set_title("cardiac cycle (inlet)", fontsize=8, pad=2)
    for spine in ("top", "right"):
        axi.spines[spine].set_visible(False)


fig = plt.figure(figsize=(14, 7.6))
draw_panel(
    fig, [0.02, 0.10, 0.46, 0.78], [0.065, 0.14, 0.15, 0.19],
    cmap_p, norm_p, p_on_ua1, p_on_ua2, Pwk_mmHg[idx],
    p_field[:, 0], "Pressure", "Pressure (mmHg)", "P inlet (mmHg)",
)
draw_panel(
    fig, [0.50, 0.10, 0.46, 0.78], [0.545, 0.14, 0.15, 0.19],
    cmap_q, norm_q, q_on_ua1, q_on_ua2, qven[idx],
    q[:, 0], "Flow rate", "Flow rate (mL/s)", "Q inlet (mL/s)",
)

fig.suptitle(
    f"Simulated pressure and flow rate at t={t_ms:.0f} ms ({pct:.0f}% of cardiac cycle, "
    f"near peak systole), GA=25wk, calibrated real reconstructed UA length L={L_cm:.1f} cm",
    fontsize=12,
)
fig.text(
    0.5, 0.02,
    "Both umbilical arteries (UA1/UA2) coloured by the spatially resolved 1D solution; UV is a single uniform "
    r"colour from the lumped Windkessel relation $P_{wk}(t)$ / $q_{ven}(t)=(P_{wk}(t)-P_v)/R_2$ (Section II-F). "
    "Insets show the corresponding inlet waveform over one cardiac cycle, with the plotted phase marked. "
    "The full cycle can be explored interactively at the companion web viewer (see caption).",
    ha="center", fontsize=9, style="italic", wrap=True,
)

out_path = f"{HERE}\\fig6.png"
plt.savefig(out_path, dpi=250)
print("saved", out_path)
