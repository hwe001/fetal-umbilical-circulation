"""Fit the placental pi-filter across GA bands to the empirical Doppler targets.

Second step of the teammate-recommended sequence
(`outputs/solver_teammate_update.md`): after the matched-1D calibration
(`fit_pi_filter_matched.py`), fit GA-band PI/RI/S/D medians from the
deidentified cohort (5,770 examinations / 1,648 patients; 1,192 usable UA
PI observations after filtering).

Per band (median GA 22/26/30/34/38 weeks):

* Geometry from `gestational_geometry` (UA diameter PCHIP curve; UV
  diameter; Ls, Rs_prior, R_UV, Cuv re-derived at the band GA).
* Cm frozen at the matched-case prior and Cuv at its GA-band geometric
  prior: planning-time identifiability analysis showed the compliances are
  NOT identifiable from UA Doppler (1/(omega*C) << R), so they are pinned,
  never fitted (handover identifiability requirement).
* The total series resistance has NO per-band flow target (the cohort has
  no quantitative flow measurements), and Doppler indices are invariant to
  a pure flow scaling -- so a convention is required.  We scale the
  matched-case total resistance by the conduit prior ratio,
  R_series(GA) = R_series(26) * Rs_prior(GA)/Rs_prior(26), i.e. the
  FLAGGED assumption that the unresolved microcirculatory resistance
  scales like the geometric conduit prior.  Mean flow is then emergent and
  reported, never fitted.
* Free parameter per band: ONE resistance-split fraction
  f = Rs/R_series(GA) (Rout = (1-f)*R_series), rootsolved (brentq) to the
  band's RI target.  RI is the only convention-clean independent index:
  S/D = 1/(1-RI) is redundant, and PI contains the time-averaged velocity
  in its denominator, which the rigid-A0 model cannot reproduce from a
  pulsating-area reference (F1, see fit_pi_filter_matched.py).
* Model PI is reported against BOTH the raw empirical PI and the
  F1-adjusted empirical PI (x the matched-case area-pulsatility
  inflation), never fitted to either.

Gestational-age caveat: the UA diameter curve is anchored 12-30 weeks
(same limitation that restricted the manuscript's Table I to GA 19-29).
The 32-36 and 36-40 bands therefore run ONLY with
--allow-geometry-extrapolation and are labelled provisional.

The cohort's 185 longitudinal patients are validation context only
(handover instruction: repeated records are not independent samples); they
constrain nothing here.

Usage:
    python calibration/fit_pi_filter_ga_bands.py
    python calibration/fit_pi_filter_ga_bands.py --allow-geometry-extrapolation
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import placental_pi_filter as pf
from gestational_geometry import ua_radius_cm

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BPM = 140.0   # matched-benchmark heart rate held fixed across bands (documented)

# Teammate update, "Approximate median GA-band targets" (cross-sectional
# medians of the deidentified cohort; PI slope -0.0256/week, R2 = 0.35)
GA_BANDS = [
    {"band": "20-24", "ga": 22.0, "PI": 1.19, "RI": 0.72, "S_D": 3.51},
    {"band": "24-28", "ga": 26.0, "PI": 1.14, "RI": 0.70, "S_D": 3.31},
    {"band": "28-32", "ga": 30.0, "PI": 0.98, "RI": 0.63, "S_D": 2.76},
    {"band": "32-36", "ga": 34.0, "PI": 0.93, "RI": 0.61, "S_D": 2.53},
    {"band": "36-40", "ga": 38.0, "PI": 0.79, "RI": 0.55, "S_D": 2.22},
]
TEAMMATE_PI_SLOPE_PER_WEEK = -0.0256

# one verified manually-annotated image (30-week intra-abdominal UA,
# diameter 0.33 cm) -- external sanity note only, never a fit input
IMAGE_RADIUS_SANITY = 0.165


def band_params(band, r_series_matched, rs_prior_26, allow_extrapolation):
    """Geometry-derived parameters for one GA band + the R_series convention."""
    ga = band["ga"]
    r0 = float(ua_radius_cm(ga, allow_extrapolation))
    rs_prior = pf.conduit_resistance(r0, 72.2, pf.CONDUIT_COEFFICIENT_1D)
    r_series = r_series_matched * rs_prior / rs_prior_26
    base = pf.default_pi_filter_params(
        ga=ga, r0_cm=r0, R_series_total=r_series,
        allow_geometry_extrapolation=allow_extrapolation)
    return base, r_series


def ri_vs_fraction(f, base, r_series):
    """Model RI at resistance-split fraction f (Rout DC-closed)."""
    p = pf.with_overrides(base, Rs=f * r_series,
                          Rout=(1.0 - f) * r_series)
    r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
    return r["metrics"]["RI"], r


def fit_band(band, r_series_matched, rs_prior_26, allow_extrapolation):
    base, r_series = band_params(band, r_series_matched, rs_prior_26,
                                 allow_extrapolation)
    target_ri = band["RI"]

    def g(f):
        ri, _ = ri_vs_fraction(f, base, r_series)
        return ri - target_ri

    lo, hi = 0.02, 0.97
    try:
        f = brentq(g, lo, hi, xtol=1e-6)
    except ValueError:
        return {"band": band["band"], "ga": band["ga"],
                "status": "RI target unreachable in the split fraction range "
                          f"[{lo}, {hi}] (model RI at bounds: "
                          f"{g(lo) + target_ri:.3f} / {g(hi) + target_ri:.3f})",
                "status_ok": False}
    ri, run = ri_vs_fraction(f, base, r_series)
    m = run["metrics"]
    return {
        "band": band["band"], "ga": band["ga"], "status": "fitted",
        "status_ok": True,
        "targets": {"PI": band["PI"], "RI": band["RI"], "S_D": band["S_D"]},
        "model": {"RI": m["RI"], "PI": m["PI"], "S_D": m["S_D"],
                  "PSV_cm_s": m["PSV_cm_s"], "EDV_cm_s": m["EDV_cm_s"],
                  "total_two_UA_flow_mL_min": m["total_two_UA_flow_mL_min"]},
        "split_fraction_Rs": float(f),
        "Rs": f * r_series, "Rout": (1.0 - f) * r_series,
        "R_series_total": r_series,
        "Ls": base.Ls, "Cm": base.Cm, "R_UV": base.R_UV, "Cuv": base.Cuv,
        "converged": run["converged"],
        "flags": run["flags"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-geometry-extrapolation", action="store_true",
                        help="run the 32-36 / 36-40 bands outside the UA "
                             "diameter curve's anchored 12-30 week range "
                             "(provisional, flagged in all outputs)")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--results-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger()
    log.addHandler(logging.FileHandler(ROOT / "logs" / "fit_pi_filter_ga_bands.log"))

    # matched-case anchors
    matched = pf.default_pi_filter_params(ga=26.0)
    r_series_matched = matched.Rs + matched.R_UV + matched.Rout
    rs_prior_26 = matched.Rs
    # F1 inflation from the matched 1D reference, if available
    f1_inflation = 1.066
    f1_source = "planning-time matched-case value (run fit_pi_filter_matched.py without --no-1d for the measured factor)"
    f1_json = ROOT / "results" / "pi_filter_matched.json"
    if f1_json.exists():
        data = json.loads(f1_json.read_text(encoding="utf-8"))
        diag = data.get("f1_diagnostic")
        if diag and diag.get("area_pulsatility_tav_inflation"):
            f1_inflation = float(diag["area_pulsatility_tav_inflation"])
            f1_source = "measured matched-case 1D reference (pi_filter_matched.json)"

    log.info("R_series(26) = %.1f dyn*s/cm^5; Rs_prior(26) = %.1f; "
             "F1 inflation = %.4f (%s)",
             r_series_matched, rs_prior_26, f1_inflation, f1_source)

    results = []
    for band in GA_BANDS:
        extrapolated = band["ga"] > 30.0
        if extrapolated and not args.allow_geometry_extrapolation:
            results.append({"band": band["band"], "ga": band["ga"],
                            "status": "skipped: GA beyond the UA diameter "
                                      "curve's anchored 12-30 week range; "
                                      "re-run with "
                                      "--allow-geometry-extrapolation",
                            "status_ok": False})
            log.info("band %s (GA %.0f): skipped (needs extrapolation)",
                     band["band"], band["ga"])
            continue
        row = fit_band(band, r_series_matched, rs_prior_26,
                       args.allow_geometry_extrapolation)
        if extrapolated:
            row["provisional_extrapolated_geometry"] = True
        if row.get("status_ok"):
            row["PI_target_F1_adjusted"] = band["PI"] * f1_inflation
            log.info("band %s (GA %.0f): f=%.4f -> RI=%.3f (target %.2f), "
                     "PI=%.3f (empirical %.2f, F1-adj %.2f), Q2UA=%.1f mL/min%s",
                     row["band"], row["ga"], row["split_fraction_Rs"],
                     row["model"]["RI"], band["RI"], row["model"]["PI"],
                     band["PI"], row["PI_target_F1_adjusted"],
                     row["model"]["total_two_UA_flow_mL_min"],
                     " [PROVISIONAL: extrapolated geometry]"
                     if extrapolated else "")
        else:
            log.info("band %s: %s", row["band"], row["status"])
        results.append(row)

    # model PI slope over the fitted, anchored bands vs the cohort's
    # cross-sectional slope
    ok = [r for r in results if r.get("status_ok")
          and not r.get("provisional_extrapolated_geometry")]
    slope_summary = None
    if len(ok) >= 2:
        gas = np.array([r["ga"] for r in ok])
        pis = np.array([r["model"]["PI"] for r in ok])
        slope_model = float(np.polyfit(gas, pis, 1)[0])
        slope_summary = {
            "model_PI_slope_per_week": slope_model,
            "cohort_cross_sectional_PI_slope_per_week":
                TEAMMATE_PI_SLOPE_PER_WEEK,
            "bands_used": [r["band"] for r in ok],
        }
        log.info("model PI slope (anchored bands): %+.4f/week vs cohort "
                 "cross-sectional %+.4f/week", slope_model,
                 TEAMMATE_PI_SLOPE_PER_WEEK)

    payload = {
        "bpm": BPM,
        "R_series_convention":
            "R_series(GA) = R_series(26) * Rs_prior(GA)/Rs_prior(26) -- "
            "FLAGGED assumption: unresolved microcirculatory resistance "
            "scales like the geometric conduit prior (no per-band flow "
            "target exists; Doppler indices are flow-scale invariant)",
        "R_series_matched_dyn_s_cm5": r_series_matched,
        "frozen": {"Cm": matched.Cm,
                   "note": "Cm/Cuv not identifiable from UA Doppler; Cm "
                           "frozen at the matched prior, Cuv re-derived "
                           "from GA geometry"},
        "f1_inflation": f1_inflation,
        "f1_inflation_source": f1_source,
        "image_radius_sanity_note": {
            "ga_weeks": 30, "site": "intra-abdominal UA",
            "diameter_cm": 0.33, "radius_cm": IMAGE_RADIUS_SANITY,
            "curve_radius_cm_30wk": float(ua_radius_cm(30.0)),
            "role": "external consistency note only; not a fit input"},
        "bands": results,
        "slope_summary": slope_summary,
        "longitudinal_note":
            "185 patients with repeated GA-separated observations are "
            "validation context only (handover: repeated records are not "
            "independent samples); they constrain nothing in this fit",
    }

    out_json = ROOT / "results" / "fit_pi_filter_ga_bands.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    log.info("wrote %s", out_json)

    if not args.no_figure:
        out_png = ROOT / "figures" / "fit_pi_filter_ga_bands.png"
        make_figure(payload, out_png)
        log.info("wrote %s", out_png)
        if args.results_dir is not None:
            try:
                copy_to = Path(args.results_dir) / "pi_filter_ga_band_plots.png"
                shutil.copyfile(out_png, copy_to)
                log.info("copied figure to %s", copy_to)
            except OSError as exc:
                log.warning("could not copy figure to results dir: %s", exc)


def make_figure(payload, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    ok = [r for r in payload["bands"] if r.get("status_ok")]
    gas = np.array([r["ga"] for r in ok])
    prov = np.array([r.get("provisional_extrapolated_geometry", False)
                     for r in ok])

    ax = axes[0]
    emp = [b for b in GA_BANDS]
    ax.plot([b["ga"] for b in emp], [b["PI"] for b in emp], "o", color="0.2",
            label="cohort median PI")
    ax.plot(gas[~prov], [r["model"]["PI"] for r, _p in
                        zip(ok, prov) if not _p], "s-", label="model PI")
    if prov.any():
        ax.plot(gas[prov], [r["model"]["PI"] for r, _p in
                            zip(ok, prov) if _p], "s--", color="0.6",
                label="model PI (extrapolated geometry)")
    adj = [r["PI_target_F1_adjusted"] for r in ok]
    ax.plot(gas, adj, "^", color="0.5", label="cohort PI (F1-adjusted)")
    ax.set_xlabel("gestational age (weeks)")
    ax.set_ylabel("PI")
    ax.set_title("Pulsatility index vs GA")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot([b["ga"] for b in emp], [b["RI"] for b in emp], "o", color="0.2",
            label="cohort median RI")
    ax.plot(gas[~prov], [r["model"]["RI"] for r, _p in
                         zip(ok, prov) if not _p], "s-",
            label="model RI (fitted)")
    if prov.any():
        ax.plot(gas[prov], [r["model"]["RI"] for r, _p in
                            zip(ok, prov) if _p], "s--", color="0.6",
                label="model RI (extrapolated geometry)")
    ax.set_xlabel("gestational age (weeks)")
    ax.set_ylabel("RI")
    ax.set_title("Resistive index vs GA (fit target)")
    ax.legend(fontsize=10)

    fig.suptitle("Placental pi-filter across GA bands "
                 "(resistance split fitted to RI per band)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
