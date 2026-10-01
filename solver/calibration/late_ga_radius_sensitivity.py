"""Radius-sensitivity of the provisional 32-40-week GA bands.

Context: the image cohort contains NO third-trimester UA-diameter
acquisitions beyond one patient's 30-week visit (queue audit 2026-09-16:
33 named candidates at 11-14 weeks, 2 at 30 weeks, zero at 32-40 weeks;
1,288 DICOM-UID images in the 24+/-1-week window remain unidentified).
The 34/38-week bands therefore run on the PCHIP-EXTRAPOLATED UA diameter
curve, and the extrapolated radius is unverifiable from project data.

This script quantifies how much the provisional band results depend on
that unverifiable radius: each band is re-fit at the curve radius scaled
by 0.8 / 1.0 / 1.2 (a +/-20% diameter error band, comparable to the
spread between the two verified 30-week measurement sites: intra-
abdominal 0.33 cm vs free loop 0.39 cm, ~17%), with the R_series
convention propagating the radius change exactly as the band fit does
(R_series scales with the conduit prior r^-4).  The split fraction f is
re-rootsolved to the band's RI target each time; PI and the emergent flow
are reported as functions of the assumed radius.

Read-only with respect to gestational_geometry.py -- the anchored curve
is never modified; extrapolation stays flagged.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import placental_pi_filter as pf
from gestational_geometry import ua_radius_cm

BPM = 140.0
PROVISIONAL_BANDS = [
    {"band": "32-36", "ga": 34.0, "RI": 0.61},
    {"band": "36-40", "ga": 38.0, "RI": 0.55},
]
RADII_SCALES = (0.8, 1.0, 1.2)


def band_params_at_r0(ga, r0_cm, r_series_matched, rs_prior_26):
    """The GA-band circuit at an explicit radius (variant of
    fit_pi_filter_ga_bands.band_params with r0 as an argument)."""
    rs_prior = pf.conduit_resistance(r0_cm, 72.2, pf.CONDUIT_COEFFICIENT_1D)
    r_series = r_series_matched * rs_prior / rs_prior_26
    base = pf.default_pi_filter_params(ga=ga, r0_cm=r0_cm,
                                       R_series_total=r_series,
                                       allow_geometry_extrapolation=True)
    return base, r_series


def fit_band_at_r0(band, r0_cm, r_series_matched, rs_prior_26):
    base, r_series = band_params_at_r0(band["ga"], r0_cm, r_series_matched,
                                       rs_prior_26)
    target_ri = band["RI"]

    def g(f):
        p = pf.with_overrides(base, Rs=f * r_series,
                              Rout=(1.0 - f) * r_series)
        return pf.simulate_pi_filter(BPM, p, init="newton_periodic")[
            "metrics"]["RI"] - target_ri

    f = None
    try:
        f = brentq(g, 0.02, 0.97, xtol=1e-6)
    except ValueError:
        # no bracket: the RI target is unreachable at ANY split fraction
        # for this radius -- itself a sensitivity finding (the extrapolated
        # regime can qualitatively break the fit, not merely shift it)
        ri_lo = g(0.02) + target_ri
        ri_hi = g(0.97) + target_ri
        return {
            "band": band["band"], "ga": band["ga"],
            "radius_scale": None, "r0_cm": float(r0_cm),
            "split_fraction_Rs": None,
            "model_RI": None, "model_PI": None, "model_S_D": None,
            "total_two_UA_flow_mL_min": None,
            "R_series_total": float(r_series),
            "converged": None,
            "status": (f"RI target {target_ri} unreachable: model RI spans "
                       f"[{ri_lo:.3f}, {ri_hi:.3f}] over the full split "
                       "range"),
        }
    p = pf.with_overrides(base, Rs=f * r_series, Rout=(1.0 - f) * r_series)
    run = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
    m = run["metrics"]
    return {
        "band": band["band"], "ga": band["ga"],
        "radius_scale": None, "r0_cm": float(r0_cm),
        "split_fraction_Rs": float(f),
        "model_RI": m["RI"], "model_PI": m["PI"],
        "model_S_D": m["S_D"],
        "total_two_UA_flow_mL_min": m["total_two_UA_flow_mL_min"],
        "R_series_total": float(r_series),
        "converged": run["converged"],
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger()
    log.addHandler(logging.FileHandler(
        ROOT / "logs" / "late_ga_radius_sensitivity.log"))

    matched = pf.default_pi_filter_params(ga=26.0)
    r_series_matched = matched.Rs + matched.R_UV + matched.Rout
    rs_prior_26 = matched.Rs

    results = []
    for band in PROVISIONAL_BANDS:
        curve_r0 = float(ua_radius_cm(band["ga"], allow_extrapolation=True))
        for scale in RADII_SCALES:
            row = fit_band_at_r0(band, curve_r0 * scale, r_series_matched,
                                 rs_prior_26)
            row["radius_scale"] = scale
            results.append(row)
            if row["model_PI"] is None:
                log.info("band %s (GA %.0f) x%.1f radius: %s",
                         band["band"], band["ga"], scale, row["status"])
            else:
                log.info("band %s (GA %.0f) x%.1f radius: r0=%.4f cm -> "
                         "f=%.3f PI=%.3f RI=%.3f Q2UA=%.1f mL/min",
                         band["band"], band["ga"], scale, row["r0_cm"],
                         row["split_fraction_Rs"], row["model_PI"],
                         row["model_RI"], row["total_two_UA_flow_mL_min"])

    # sensitivity summary: PI and flow spread across the +/-20% radius band
    summary = {}
    for band in PROVISIONAL_BANDS:
        rows = [r for r in results if r["band"] == band["band"]]
        fitted = [r for r in rows if r["model_PI"] is not None]
        unreachable = [r for r in rows if r["model_PI"] is None]
        summary[band["band"]] = {
            "PI_range": [min(r["model_PI"] for r in fitted),
                         max(r["model_PI"] for r in fitted)],
            "flow_range_mL_min": [
                min(r["total_two_UA_flow_mL_min"] for r in fitted),
                max(r["total_two_UA_flow_mL_min"] for r in fitted)],
            "f_range": [min(r["split_fraction_Rs"] for r in fitted),
                        max(r["split_fraction_Rs"] for r in fitted)],
            "n_radius_scales_unreachable": len(unreachable),
            "unreachable_status": [r["status"] for r in unreachable],
        }
        log.info("band %s sensitivity: PI %.3f-%.3f, flow %.1f-%.1f mL/min, "
                 "f %.3f-%.3f, unreachable scales: %d",
                 band["band"],
                 *summary[band["band"]]["PI_range"],
                 *summary[band["band"]]["flow_range_mL_min"],
                 *summary[band["band"]]["f_range"],
                 summary[band["band"]]["n_radius_scales_unreachable"])

    payload = {
        "purpose": "sensitivity of the provisional 32-40-week bands to the "
                   "unverifiable extrapolated UA radius (no third-trimester "
                   "acquisitions exist in the image cohort beyond one 30-"
                   "week patient; queue audit 2026-09-16)",
        "radii_scales": list(RADII_SCALES),
        "bands": results,
        "summary": summary,
    }
    out = ROOT / "results" / "late_ga_radius_sensitivity.json"
    out.write_text(json.dumps(payload, indent=2, default=str),
                   encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
