"""Render the identifiability profile figure (reviewer major comment 4)
from the stored sweep artifacts in results/pi_filter_matched.json.

No re-simulation: everything plotted was produced by
fit_pi_filter_matched.py's stage 4 (1D no-refit sweeps, Cm x Cuv grid,
Rs x Ls grid) and saved in the results JSON.

Outputs: figures/identifiability_profiles.png (+ copy to the outputs dir).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    data = json.loads((ROOT / "results" / "pi_filter_matched.json")
                      .read_text(encoding="utf-8"))
    ident = data["identifiability"]
    sweeps = ident["sweeps_1d"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8),
                        gridspec_kw={"width_ratios": [1, 1.7, 1]})

    # (a) 1D profile sweeps: cost vs multiplicative factor of each parameter
    ax = axes[0]
    colors = plt.get_cmap("tab10")
    for k, (name, sweep) in enumerate(sweeps.items()):
        factors = np.array(sweep["factors"])
        costs = np.array([c if c is not None and np.isfinite(c) else np.nan
                          for c in sweep["costs"]])
        ax.plot(factors, costs, "o-", ms=3.5, color=colors(k), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("parameter / fitted value")
    ax.set_ylabel("cost  J = 0.5 Σ r²")
    ax.set_title("(a) 1D profile sweeps (no refit)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # (b) Cm x Cuv admissible region
    grid = ident["grid_Cm_Cuv"]
    log_c = np.array(grid["log10_Cm"])
    cost = np.array([[np.nan if c is None else c for c in row]
                     for row in grid["cost"]])
    ax = axes[1]
    pc = ax.pcolormesh(log_c, log_c, np.log10(cost + 1e-12), shading="auto",
                       cmap="viridis")
    j_min = np.nanmin(cost)
    ax.contour(log_c, log_c, cost - j_min, levels=[1.0, 4.0, 9.0],
               colors="w", linewidths=2.0)
    ax.plot([np.log10(4e-5)], [np.log10(5.65e-5)], "r*", ms=12,
            label="fitted / pinned values")
    ax.set_xlabel("log10 Cm (cm5/dyn)")
    ax.set_ylabel("log10 Cuv (cm5/dyn)")
    ax.set_title("(b) Cm × Cuv: the compliance plateau\n"
                 "(broad plateau — compliance NOT identifiable "
                 "from UA Doppler)")
    ax.legend(fontsize=10, loc="lower left")
    fig.colorbar(pc, ax=ax, label="log10 J")
    ax.annotate("PLATEAU", xy=(0.5, 0.5), xycoords="axes fraction",
                fontsize=13, color="w", ha="center", fontweight="bold")

    # (c) Rs x Ls ridge grid (Rout DC-coupled)
    grid2 = ident["grid_Rs_Ls"]
    fs = np.array(grid2["Rs_factors"])
    cost2 = np.array([[np.nan if c is None else c for c in row]
                      for row in grid2["cost"]])
    ax = axes[2]
    pc2 = ax.pcolormesh(fs, fs, np.log10(cost2 + 1e-12), shading="auto",
                        cmap="viridis")
    j_min2 = np.nanmin(cost2)
    ax.contour(fs, fs, cost2 - j_min2, levels=[1.0, 4.0, 9.0],
               colors="w", linewidths=2.0)
    ax.plot([1.0], [1.0], "r*", ms=12, label="fitted values")
    ax.set_xlabel("Rs / fitted value")
    ax.set_ylabel("Ls / fitted value")
    ax.set_title("(c) Rs × Ls (Rout DC-coupled): axes well identified")
    ax.legend(fontsize=10, loc="lower left")
    fig.colorbar(pc2, ax=ax, label="log10 J")

    fig.suptitle("Identifiability of the placental pi-filter at the fitted "
                 "operating point (reviewer major comment 4)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "figures" / "identifiability_profiles.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)
    try:
        shutil.copyfile(out, DEFAULT_RESULTS_DIR / "identifiability_profiles.png")
        print("copied to", DEFAULT_RESULTS_DIR / "identifiability_profiles.png")
    except OSError as exc:
        print("copy failed:", exc)


if __name__ == "__main__":
    main()
