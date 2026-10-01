"""Waveform-level UV output: the fitted pi-filter driving the reduced 1D UV.

Implements the handover task "add an explicit UV outlet pressure or reduced
1D UV return model" at waveform level: the calibrated placental pi-filter
determines the storage-consistent placental efflux Q_out(t) (the flow that
has left the placental compliance -- the pi-filter analogue of the 1D
solver's efflux 2*(Pwk-Pv)/R2 vs the arterial inflow q_b, per the solver
README's UV bookkeeping note), and ``solve_venous_flow_UV`` converts that
inflow into cord-vein velocity and pressure-drop waveforms at the GA-derived
UV calibre.

The coupling is one-way and quasi-steady: the UV's Poiseuille resistance is
a fraction of a percent of the total series resistance and its drop is
already inside the pi-filter's Rout lump, so feeding back would
double-count it.

Honest scale caveat (computed, not hidden): the model's ABSOLUTE flow scale
follows the matched 1D benchmark's DC convention (35.29 mL/min two UAs at
the fitted point), which the 1D project itself documents as undershooting
literature umbilical flow by 54-92% at GA 19-29 (solver README, Table II).
Model UV mean velocities are therefore correspondingly low vs typical
literature values; the waveform SHAPE (pulsatility, timing) is the
scale-free output of interest here, and the pi-filter's placental+venous
compliances are shown to smooth the efflux to near-steady UV flow.

Usage:
    python calibration/uv_coupled_waveforms.py
    python calibration/uv_coupled_waveforms.py --allow-geometry-extrapolation
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import placental_pi_filter as pf
from gestational_geometry import uv_radius_cm

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BPM = 140.0
FALLBACK_F1_INFLATION = 1.0654


def fitted_matched_params():
    """Rebuild the fitted matched-case parameter set from the results JSON."""
    data = json.loads((ROOT / "results" / "pi_filter_matched.json")
                      .read_text(encoding="utf-8"))
    fitted = data["fitted_params"]
    base = pf.default_pi_filter_params()
    p = pf.with_overrides(base, Rs=fitted["Rs"], Rout=fitted["Rout"])
    f1 = data.get("f1_diagnostic") or {}
    return p, float(f1.get("area_pulsatility_tav_inflation",
                           FALLBACK_F1_INFLATION))


def pulsatility(signal):
    """(max-min)/mean of a wrap-inclusive waveform (exclude the duplicate)."""
    core = np.asarray(signal)[:-1]
    return float((core.max() - core.min()) / max(abs(core.mean()), 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-geometry-extrapolation", action="store_true",
                        help="include the provisional 34/38-week bands")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--results-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger()
    log.addHandler(logging.FileHandler(
        ROOT / "logs" / "uv_coupled_waveforms.log"))

    import fit_pi_filter_ga_bands as gb

    matched = pf.default_pi_filter_params(ga=26.0)
    r_series_matched = matched.Rs + matched.R_UV + matched.Rout

    params, f1_inflation = fitted_matched_params()
    log.info("fitted matched-case params rebuilt (Rs=%.1f Rout=%.1f); "
             "F1 inflation %.4f", params.Rs, params.Rout, f1_inflation)

    cases = []
    # --- matched case -----------------------------------------------------
    run = pf.simulate_pi_filter(BPM, params)
    coupling = pf.couple_uv_return(run, params, ga=26.0)
    cases.append({"case": "matched_1D_fit_GA26", "ga": 26.0,
                  "params_source": "fitted (fit_pi_filter_matched.json)",
                  "run": run, "coupling": coupling})
    log.info("matched GA26: UV mean velocity %.3f cm/s, mean flow %.2f "
             "mL/min, efflux pulsatility %.2f%% (arterial inflow %.2f%%)",
             coupling["uv"]["mean_velocity_cm_s"],
             coupling["uv"]["mean_flow_mL_min"],
             100.0 * pulsatility(coupling["efflux_total_mL_s"]),
             100.0 * pulsatility(run["Q_UA"]))

    # --- GA bands (re-fit identically to fit_pi_filter_ga_bands) ----------
    for band in gb.GA_BANDS:
        extrapolated = band["ga"] > 30.0
        if extrapolated and not args.allow_geometry_extrapolation:
            continue
        row = gb.fit_band(band, r_series_matched, matched.Rs,
                          args.allow_geometry_extrapolation)
        if not row.get("status_ok"):
            log.info("band %s: %s", band["band"], row["status"])
            continue
        base, _r_series = gb.band_params(band, r_series_matched, matched.Rs,
                                         args.allow_geometry_extrapolation)
        p_band = pf.with_overrides(base, Rs=row["Rs"], Rout=row["Rout"])
        run_b = pf.simulate_pi_filter(BPM, p_band)
        coupling_b = pf.couple_uv_return(run_b, p_band, ga=band["ga"],
                                         allow_geometry_extrapolation=True)
        cases.append({"case": f"band_{band['band']}", "ga": band["ga"],
                      "params_source": "GA-band RI-matched fit",
                      "run": run_b, "coupling": coupling_b,
                      "provisional": extrapolated or None})
        log.info("band %s (GA %.0f): UV mean velocity %.3f cm/s, mean flow "
                 "%.2f mL/min, efflux pulsatility %.2f%%%s",
                 band["band"], band["ga"],
                 coupling_b["uv"]["mean_velocity_cm_s"],
                 coupling_b["uv"]["mean_flow_mL_min"],
                 100.0 * pulsatility(coupling_b["efflux_total_mL_s"]),
                 " [PROVISIONAL: extrapolated geometry]" if extrapolated
                 else "")

    # --- serialization -----------------------------------------------------
    payload = {
        "bpm": BPM,
        "coupling": "one-way quasi-steady: pi-filter efflux Q_out(t) -> "
                    "solve_venous_flow_UV (storage-consistent inflow; the UV "
                    "does not feed back, its drop is inside Rout)",
        "f1_inflation": f1_inflation,
        "scale_caveat":
            "absolute flow/velocity scale follows the matched 1D benchmark's "
            "DC convention, which the 1D project documents as undershooting "
            "literature umbilical flow by 54-92% at GA 19-29; waveform shape "
            "and pulsatility are the scale-free outputs",
        "cases": [
            {
                "case": c["case"], "ga": c["ga"],
                "params_source": c["params_source"],
                "provisional": c.get("provisional"),
                "converged": c["run"]["converged"],
                "uv_radius_cm": c["coupling"]["uv"]["radius_cm"],
                "uv_resistance_dyn_s_cm5":
                    c["coupling"]["uv"]["resistance_dyn_s_cm5"],
                "uv_mean_flow_mL_min":
                    c["coupling"]["uv"]["mean_flow_mL_min"],
                "uv_mean_velocity_cm_s":
                    c["coupling"]["uv"]["mean_velocity_cm_s"],
                "uv_velocity_range_cm_s": [
                    float(c["coupling"]["uv"]["velocity_cm_s"].min()),
                    float(c["coupling"]["uv"]["velocity_cm_s"].max())],
                "uv_mean_pressure_drop_mmHg": float(
                    np.mean(c["coupling"]["uv"]["pressure_drop_dyn_cm2"]))
                    / pf.MMHG_TO_DYN_CM2,
                "mean_efflux_total_mL_min":
                    c["coupling"]["mean_efflux_total_mL_min"],
                "mean_arterial_inflow_total_mL_min":
                    c["coupling"]["mean_arterial_inflow_total_mL_min"],
                "efflux_pulsatility": pulsatility(c["coupling"]["efflux_total_mL_s"]),
                "arterial_inflow_pulsatility": pulsatility(c["run"]["Q_UA"]),
                "model_PI": c["run"]["metrics"]["PI"],
                "model_RI": c["run"]["metrics"]["RI"],
            }
            for c in cases
        ],
    }
    out_json = ROOT / "results" / "uv_coupled_waveforms.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    log.info("wrote %s", out_json)

    if not args.no_figure:
        out_png = ROOT / "figures" / "uv_coupled_waveforms.png"
        make_figure(cases, out_png)
        log.info("wrote %s", out_png)
        if args.results_dir is not None:
            try:
                copy_to = Path(args.results_dir) / "uv_coupled_waveforms.png"
                shutil.copyfile(out_png, copy_to)
                log.info("copied figure to %s", copy_to)
            except OSError as exc:
                log.warning("could not copy figure to results dir: %s", exc)


def make_figure(cases, path):
    matched = cases[0]
    run, coupling = matched["run"], matched["coupling"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))

    ax = axes[0]
    ax.plot(run["t"], run["Q_UA"] * 2.0, lw=2.0,
            label="arterial inflow $2\\,Q_{UA}$")
    ax.plot(run["t"], coupling["efflux_total_mL_s"], lw=2.0, ls="--",
            label="placental efflux $2\\,Q_{out}$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("flow (mL/s)")
    ax.set_title("Placental storage filters the efflux\n(fitted matched case)")
    ax.legend(fontsize=11)

    ax = axes[1]
    uv = coupling["uv"]
    ax.plot(uv["t"], uv["velocity_cm_s"], lw=2.0, color="C2")
    ax2 = ax.twinx()
    ax2.plot(uv["t"], uv["pressure_drop_dyn_cm2"] / pf.MMHG_TO_DYN_CM2,
             lw=1.0, color="0.6")
    ax2.set_ylabel("UV pressure drop (mmHg)", color="0.4")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("UV velocity (cm/s)", color="C2")
    ax.set_title(f"Cord-vein waveform at GA 26\n(mean "
                 f"{uv['mean_velocity_cm_s']:.2f} cm/s, pulsatility "
                 f"{100.0 * pulsatility(uv['velocity_cm_s']):.1f}%)")

    ax = axes[2]
    ok = [c for c in cases[1:] if not c.get("provisional")]
    prov = [c for c in cases[1:] if c.get("provisional")]
    gas_ok = [c["ga"] for c in ok]
    vel_ok = [c["coupling"]["uv"]["mean_velocity_cm_s"] for c in ok]
    q_ok = [c["coupling"]["uv"]["mean_flow_mL_min"] for c in ok]
    ax.plot(gas_ok, q_ok, "o-", label="UV mean flow (mL/min)")
    ax.set_xlabel("gestational age (weeks)")
    ax.set_ylabel("UV mean flow (mL/min)", color="C0")
    ax3 = ax.twinx()
    ax3.plot(gas_ok, vel_ok, "s--", color="C1",
             label="UV mean velocity (cm/s)")
    if prov:
        ax3.plot([c["ga"] for c in prov],
                 [c["coupling"]["uv"]["mean_velocity_cm_s"] for c in prov],
                 "s", color="0.6", label="velocity (extrapolated, provisional)")
        ax.plot([c["ga"] for c in prov],
                [c["coupling"]["uv"]["mean_flow_mL_min"] for c in prov],
                "o", color="0.6")
    ax3.set_ylabel("UV mean velocity (cm/s)", color="C1")
    ax.set_title("UV mean flow / velocity vs GA\n(flagged R_series scaling "
                 "assumption)")
    ax.legend(fontsize=8, loc="upper left")
    ax3.legend(fontsize=8, loc="lower right")

    fig.suptitle("Pi-filter efflux driving the reduced 1D UV "
                 "(absolute scale anchored to the matched 1D benchmark)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
