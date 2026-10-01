"""Render the manuscript's path-length ablation figure (two panels).

Panel A: PI vs UA trunk length; Panel B: S/D vs UA trunk length.
Two series per panel:
  - fixed-parameter arm (GA-25 Table-II operating point held, only L varies)
  - held-out 24-week predictions after full recalibration at each length
Shaded bands: cohort PI range across 19-29 weeks (0.96-1.30, Table II
targets) and the normal (3.02) / IUGR (5.70) S/D waveform values of
Wen et al. 2020.

Data: solver/results/anatomy_ablation.json (produced by
calibration/anatomy_ablation.py). Output: TMI figures/fig_ablation.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fig_ablation.png"

d = json.loads((ROOT / "results" / "anatomy_ablation.json").read_text(encoding="utf-8"))

scen = d["scenarios"]
fixed = {s["name"]: s["metrics"] for s in d["fixed_parameter_arm"]["scenarios"]}
refit = {s["name"]: s["held_out_24wk"] for s in d["refit_arm"]["scenarios"]}

rows = sorted(scen, key=lambda s: s["L_cm"])
L = [s["L_cm"] for s in rows]
pi_fixed = [fixed[s["name"]]["PI"] for s in rows]
sd_fixed = [fixed[s["name"]]["S_D"] for s in rows]
pi_held = [refit[s["name"]]["PI"] for s in rows]
sd_held = [refit[s["name"]]["S_D"] for s in rows]

L_REF = 72.23
PI_BAND = (0.960, 1.295)   # cohort targets, 19-29 wk (Table II)
SD_BAND = (3.02, 5.70)     # normal / IUGR waveforms, Wen et al. 2020

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

C_FIXED = "#b2182b"
C_HELD = "#2166ac"

ax = axes[0]
ax.axhspan(*PI_BAND, color="#cccccc", alpha=0.35, linewidth=0)
ax.axvline(L_REF, color="#666666", linestyle=":", linewidth=1.0)
ax.plot(L, pi_fixed, "o--", color=C_FIXED, markerfacecolor="white",
        markersize=7, linewidth=1.4, label="fixed physiology")
ax.plot(L, pi_held, "s-", color=C_HELD, markersize=6,
        linewidth=1.6, label="after recalibration")
ax.set_xlabel("UA trunk length (cm)")
ax.set_ylabel("PI")
ax.set_ylim(0.5, 1.85)
ax.set_title("A  Pulsatility index")
ax.text(0.03, 0.90, "cohort PI range, 19–29 wk", transform=ax.transAxes,
        fontsize=10.5, color="#444444")

ax = axes[1]
ax.axhspan(*SD_BAND, color="#cccccc", alpha=0.35, linewidth=0)
ax.axvline(L_REF, color="#666666", linestyle=":", linewidth=1.0)
ax.plot(L, sd_fixed, "o--", color=C_FIXED, markerfacecolor="white",
        markersize=7, linewidth=1.4)
ax.plot(L, sd_held, "s-", color=C_HELD, markersize=6, linewidth=1.6)
ax.set_xlabel("UA trunk length (cm)")
ax.set_ylabel("S/D")
ax.set_ylim(1.6, 6.6)
ax.set_title("B  Systolic/diastolic ratio")
ax.text(0.03, 0.90, "normal – IUGR waveform S/D", transform=ax.transAxes,
        fontsize=10.5, color="#444444")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
