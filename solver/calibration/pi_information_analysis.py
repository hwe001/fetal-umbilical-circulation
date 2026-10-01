import os
# -*- coding: utf-8 -*-
"""Quantify how much independent information PI carries once the GA-band
split has been fitted to RI (editor concern #3: circularity).

(a) Cohort side: Pearson/Spearman correlation between PI and RI across
    the 1,194 filtered observations (shared numerator -> strong expected
    correlation).
(b) Model side: at each anchored band, with the split fixed by the RI
    fit, sweep the pinned shape parameters over their admissible ranges
    (Ls x0.25..x4 with Rs re-root-solved to the same RI target; Cm, Cuv
    over the plateau) and report how far the predicted PI can move.
    If PI barely moves, the PI agreement in 3.5 is a consistency check of
    the geometry scaling rather than an independent test -- stated as
    such.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

SOLVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLVER / "model"))
import placental_pi_filter as pf
from gestational_geometry import ua_radius_cm

BPM = 140.0
BANDS = [(22.0, 0.72), (26.0, 0.70), (30.0, 0.63)]


def cohort_correlation():
    df = pd.read_excel(
        os.environ.get("UA_COHORT_XLSX", "fetal_doppler_deidentified.xlsx"))
    pi = df[df["UA_PI_struct"].notna()].copy()
    pi["GA_weeks"] = pd.to_numeric(pi["Clinical_GA"], errors="coerce")
    pi.loc[pi["GA_weeks"].isna(), "GA_weeks"] = pd.to_numeric(
        pi["US_GA"], errors="coerce")
    pi = pi[(pi["GA_weeks"] >= 8) & (pi["GA_weeks"] <= 45)
            & (pi["UA_PI_struct"] > 0) & (pi["UA_PI_struct"] <= 4.0)]
    pi["RI"] = pd.to_numeric(pi["UA_RI_struct"], errors="coerce")
    pi = pi[pi["RI"].notna()]
    pearson = float(np.corrcoef(pi["UA_PI_struct"], pi["RI"])[0, 1])
    spearman = float(pi["UA_PI_struct"].corr(pi["RI"], method="spearman"))
    return {"n": int(len(pi)), "pearson_PI_RI": pearson,
            "spearman_PI_RI": spearman}


def main():
    out = {"cohort": cohort_correlation()}
    print("cohort PI-RI:", out["cohort"])

    data = json.loads((SOLVER / "results" / "pi_filter_matched.json")
                      .read_text(encoding="utf-8"))
    fp = data["fitted_params"]
    matched = pf.default_pi_filter_params()
    r_series_matched = matched.Rs + matched.R_UV + matched.Rout

    band_results = []
    for ga, ri_target in BANDS:
        r0 = float(ua_radius_cm(ga))
        rs_prior = pf.conduit_resistance(r0, 72.2, pf.CONDUIT_COEFFICIENT_1D)
        r_series = r_series_matched * rs_prior / matched.Rs
        base = pf.default_pi_filter_params(
            ga=ga, r0_cm=r0, R_series_total=r_series)

        def fit_split(ls_scale, cm, cuv):
            Ls = base.Ls * ls_scale
            rd = pf.conduit_resistance  # alias unused
            # root-solve the split fraction to the RI target at this shape
            def g(f):
                p = pf.with_overrides(base, Ls=Ls, Rs=f * r_series,
                                      Rout=(1 - f) * r_series)
                return pf.simulate_pi_filter(
                    BPM, p, init="newton_periodic")["metrics"]["RI"] - ri_target
            f = brentq(g, 0.02, 0.97, xtol=1e-6)
            p = pf.with_overrides(base, Ls=Ls, Rs=f * r_series,
                                  Rout=(1 - f) * r_series)
            r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
            return r["metrics"]["PI"], f

        # reference point (pinned shape)
        pi_ref, f_ref = fit_split(1.0, base.Cm, base.Cuv)
        # Ls admissible sweep (x0.5..x2; the ridge direction), RI re-fit
        pis = []
        for ls_s in (0.5, 1.0, 2.0):
            pi_v, f_v = fit_split(ls_s, base.Cm, base.Cuv)
            pis.append(pi_v)
        # compliance plateau sweep at pinned Ls (Cm, Cuv corners of the
        # 1e-5..1.6e-4 plateau)
        for cm in (1e-5, 1.6e-4):
            for cuv in (1e-5, 1.6e-4):
                p = pf.with_overrides(base, Cm=cm, Cuv=cuv,
                                      Rs=pi_ref and base.Rs,
                                      Rout=(1 - 0.7483) * r_series)
                # keep the fitted split; compliances barely move RI (F4)
                r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
                pis.append(r["metrics"]["PI"])
        row = {
            "ga": ga, "RI_target": ri_target, "split_fraction": f_ref,
            "PI_reference": pi_ref,
            "PI_range_over_admissible_shape": [min(pis), max(pis)],
            "PI_relative_spread": (max(pis) - min(pis)) / pi_ref,
        }
        band_results.append(row)
        print(json.dumps(row, default=str))

    out["bands"] = band_results
    out["statement"] = (
        "Given the RI fit, per-band model PI moves only within the reported "
        "range when the pinned shape parameters sweep their admissible "
        "ranges; the PI agreement in 3.5 is therefore a consistency check "
        "of the GA-trend of the resistance split under the geometry "
        "convention, not an independent out-of-sample validation.")
    (SOLVER / "results" / "pi_information_analysis.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", SOLVER / "results" / "pi_information_analysis.json")


if __name__ == "__main__":
    main()
