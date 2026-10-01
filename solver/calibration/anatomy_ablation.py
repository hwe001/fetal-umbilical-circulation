"""Anatomy ablation: what does the MR-derived UA trunk length contribute?

Two arms, per external-review protocol (2026-09-13):

  (i)  FIXED-PARAMETER ARM -- hold the GA=25 Table-I production fit
       (R_total, dp_inlet, p0) fixed and vary only the UA trunk length.
       Measures the model's DIRECT sensitivity to anatomy: how much a
       plausible length error changes PI/RI/S/D and mean flow when the
       physiology is not re-adjusted.

  (ii) REFIT ARM -- recalibrate (R_total, dp_inlet) at GA 19/23/25/29 for
       each length scenario (same cohort-median PI/RI targets, same
       coarse-mesh procedure as Table I), interpolate to 24 weeks, and
       compare held-out PI/RI/S/D and absolute flow against the pilot
       cohort and the Ozawa reference. Tests whether calibration ABSORBS
       the anatomical difference: a null result here is only meaningful
       alongside arm (i).

Length scenario construction (documented per review):

  L_UA = L_cord * sqrt(1 + (2*pi*r*CI)^2)

    r      = 0.482 cm radial offset (specimen configuration, manuscript II-C)
    CI     = coiling index in turns/cm. van Dijk 2002 population: mean 0.17,
             10th-90th percentile 0.07-0.30 (as quoted in manuscript II-C).
    L_cord = umbilical cord arc length. Specimen: 61.78 cm at ~34 weeks.
             Short/long scenarios (45 / 75 cm) bracket the population range
             of cord lengths reported in cord-length studies (e.g. Naeye
             1985); exact percentile anchoring to be confirmed against the
             source before publication.

  Scenarios vary ONE factor at a time, holding the other at the specimen
  reference (L_cord = 61.78 cm, CI = 0.20 -> 72.2 cm). Cord length and
  coiling index are correlated in some reported populations; JOINT
  variation is not modeled here and is a stated limitation.

  PROVENANCE CAVEAT (per review): the 72.2 cm reference path is derived
  from ONE approximately 34-week specimen. It is NOT a patient-specific
  length known at the 19-29-week calibration points -- no length-vs-GA
  curve is available (manuscript II-E states this).

Outputs: solver/results/anatomy_ablation.json (written incrementally) and
a run log on stdout. Fit procedure matches Table I: fit_ga_targets.calibrate
at dx=0.73 cm, max_nfev=18, index_definition="minimum", warm-started from
the 72.2 cm Table-I fits; held-out/production checks at dx=0.25 cm.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "calibration"))
import solve_arterial_flow_UA as ua
from fit_ga_targets import calibrate
from gestational_geometry import ua_radius_cm

OUT_PATH = ROOT / "results" / "anatomy_ablation.json"

R1_FRAC = 0.15
COMPLIANCE = 4.0e-5
BPM = 140.0
P0_FIXED = 30000.0          # Table-I p0 (P0_DEFAULT)
PROD_DX_CM = 0.25
COARSE_DX_CM = 0.73
MAX_NFEV = 18               # same budget as the Table-I calibration

R_OFFSET_CM = 0.482
SPEC_CORD_CM = 61.78
SPEC_CI = 0.20

GAS = [19, 23, 25, 29]
TARGETS = {19: (1.295, 0.740), 23: (1.190, 0.710),
           25: (1.140, 0.700), 29: (0.960, 0.630)}

# Held-out references at 24 weeks:
PILOT_24 = {"PI": 1.04, "RI": 0.66, "S_D": 2.96}   # pilot cohort, 24+-1 wk
# Ozawa 50th percentile at 23 wk (71.7) and 25 wk (92.7), linearly
# interpolated to 24 wk (manuscript Table II targets at the flanking GAs):
OZAWA_24_ML_MIN = 0.5 * (71.7 + 92.7)


def helix_len(cord_cm, ci):
    return cord_cm * np.sqrt(1.0 + (2.0 * np.pi * R_OFFSET_CM * ci) ** 2)


SCENARIOS = [
    {"name": "reference_72",  "cord_cm": SPEC_CORD_CM, "ci": SPEC_CI,
     "note": "MR-derived specimen length (one ~34-week specimen)"},
    {"name": "ci_low_010",    "cord_cm": SPEC_CORD_CM, "ci": 0.10,
     "note": "coiling near 10th percentile (van Dijk), same cord"},
    {"name": "ci_high_030",   "cord_cm": SPEC_CORD_CM, "ci": 0.30,
     "note": "coiling near 90th percentile (van Dijk), same cord"},
    {"name": "cord_short_45", "cord_cm": 45.0, "ci": SPEC_CI,
     "note": "short-cord scenario, same coiling"},
    {"name": "cord_long_75",  "cord_cm": 75.0, "ci": SPEC_CI,
     "note": "long-cord scenario, same coiling"},
]
for s in SCENARIOS:
    s["L_cm"] = round(float(helix_len(s["cord_cm"], s["ci"])), 2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(results):
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def prod_sim(ga, L, R_total, dp_inlet, p0=P0_FIXED, n_cycles=15):
    r0 = ua_radius_cm(ga, allow_extrapolation=False)
    M = int(round(L / PROD_DX_CM)) + 1
    return ua.simulate(bpm=BPM, r0=r0, R_total=R_total, C_total=COMPLIANCE,
                       p0=p0, dp_inlet=dp_inlet, L=L, M=M,
                       n_cycles=n_cycles, max_cycles=80,
                       index_definition="minimum")


def summarize(res):
    m = res["metrics"]
    return {
        "PI": round(m["PI"], 4), "RI": round(m["RI"], 4),
        "S_D": round(m["S_D"], 4),
        "total_two_UA_flow_mL_min": round(m["total_two_UA_flow_mL_min"], 2),
        "converged": bool(res["converged"]),
        "cycles": int(res["n_cycles_run"]),
        "fallback": round(res["outlet_fallback_frac"], 5),
    }


def fixed_arm(results):
    log("=== ARM (i): fixed-parameter sensitivity at GA=25 ===")
    ref = next(r for r in results["table_I_reference"]
               if r["ga_weeks"] == 25)
    R_fixed = ref["R_total_dyn_s_cm5"]
    dp_fixed = ref["dp_inlet_dyn_cm2"]
    log(f"fixed operating point: R={R_fixed:.1f}, dp={dp_fixed:.1f}, p0={P0_FIXED:.0f}")
    results["fixed_parameter_arm"] = {
        "description": "GA=25 Table-I production fit held fixed; only L varies",
        "R_total": R_fixed, "dp_inlet": dp_fixed, "p0": P0_FIXED,
        "scenarios": [],
    }
    for s in SCENARIOS:
        t0 = time.time()
        res = prod_sim(25, s["L_cm"], R_fixed, dp_fixed)
        row = {"name": s["name"], "L_cm": s["L_cm"], "note": s["note"],
               "metrics": summarize(res)}
        results["fixed_parameter_arm"]["scenarios"].append(row)
        log(f"  {s['name']}: L={s['L_cm']}cm PI={row['metrics']['PI']:.3f} "
            f"RI={row['metrics']['RI']:.3f} SD={row['metrics']['S_D']:.2f} "
            f"flow={row['metrics']['total_two_UA_flow_mL_min']:.1f} "
            f"conv={row['metrics']['converged']} ({time.time()-t0:.0f}s)")
        save(results)


def refit_arm(results):
    log("=== ARM (ii): refit per length scenario, held-out 24 wk ===")
    results["refit_arm"] = {"description": "recalibrated per length scenario "
                            "(Table-I procedure), held-out comparison at 24 wk",
                            "scenarios": []}
    ref_fits = {r["ga_weeks"]: r for r in results["table_I_reference"]}
    for s in SCENARIOS:
        t0 = time.time()
        log(f"--- scenario {s['name']} (L={s['L_cm']} cm) ---")
        entry = {"name": s["name"], "L_cm": s["L_cm"], "note": s["note"],
                 "fits": []}
        for ga in GAS:
            warm = ref_fits[ga]
            initial = [warm["R_total_dyn_s_cm5"], warm["dp_inlet_dyn_cm2"]]
            tgt_pi, tgt_ri = TARGETS[ga]
            log(f"  GA={ga}: fitting (targets PI={tgt_pi}, RI={tgt_ri})",
                )
            fit = calibrate(ga, s["L_cm"], tgt_pi, tgt_ri, initial,
                            dx_cm=COARSE_DX_CM, max_nfev=MAX_NFEV,
                            index_definition="minimum")
            entry["fits"].append({
                "ga_weeks": ga,
                "R_total": round(fit["R_total_dyn_s_cm5"], 1),
                "dp_inlet": round(fit["dp_inlet_dyn_cm2"], 1),
                "PI_sim": round(fit["metrics"]["PI"], 4),
                "RI_sim": round(fit["metrics"]["RI"], 4),
                "PI_tgt": tgt_pi, "RI_tgt": tgt_ri,
                "optimizer_success": fit["optimizer_success"],
                "cost": round(fit["cost"], 4),
            })
            log(f"    -> R={fit['R_total_dyn_s_cm5']:.0f} "
                f"dp={fit['dp_inlet_dyn_cm2']:.0f} "
                f"PI={fit['metrics']['PI']:.3f} RI={fit['metrics']['RI']:.3f} "
                f"cost={fit['cost']:.3f}")
            save(results)

        # held-out 24 wk: linear interpolation between the 23 and 25 wk fits
        f23 = next(f for f in entry["fits"] if f["ga_weeks"] == 23)
        f25 = next(f for f in entry["fits"] if f["ga_weeks"] == 25)
        R24 = 0.5 * (f23["R_total"] + f25["R_total"])
        dp24 = 0.5 * (f23["dp_inlet"] + f25["dp_inlet"])
        log(f"  held-out 24 wk: R={R24:.0f} dp={dp24:.0f}")
        res24 = prod_sim(24, s["L_cm"], R24, dp24)
        m24 = res24["metrics"]
        entry["held_out_24wk"] = {
            "R_total_interp": round(R24, 1),
            "dp_inlet_interp": round(dp24, 1),
            "PI": round(m24["PI"], 4), "RI": round(m24["RI"], 4),
            "S_D": round(m24["S_D"], 4),
            "total_two_UA_flow_mL_min": round(m24["total_two_UA_flow_mL_min"], 2),
            "pct_error_PI": round(100 * (m24["PI"] - PILOT_24["PI"]) / PILOT_24["PI"], 1),
            "pct_error_RI": round(100 * (m24["RI"] - PILOT_24["RI"]) / PILOT_24["RI"], 1),
            "pct_error_SD": round(100 * (m24["S_D"] - PILOT_24["S_D"]) / PILOT_24["S_D"], 1),
            "flow_pct_error_vs_Ozawa24": round(
                100 * (m24["total_two_UA_flow_mL_min"] - OZAWA_24_ML_MIN) / OZAWA_24_ML_MIN, 1),
            "converged": bool(res24["converged"]),
            "cycles": int(res24["n_cycles_run"]),
        }
        h = entry["held_out_24wk"]
        log(f"  held-out: PI={h['PI']:.3f} ({h['pct_error_PI']:+.1f}%) "
            f"RI={h['RI']:.3f} ({h['pct_error_RI']:+.1f}%) "
            f"SD={h['S_D']:.2f} ({h['pct_error_SD']:+.1f}%) "
            f"flow={h['total_two_UA_flow_mL_min']:.1f} "
            f"({h['flow_pct_error_vs_Ozawa24']:+.1f}%)")
        results["refit_arm"]["scenarios"].append(entry)
        save(results)
        log(f"  scenario done ({time.time()-t0:.0f}s)")


def main():
    ref_data = json.loads((ROOT / "results" / "full_recalibration.json")
                          .read_text(encoding="utf-8"))
    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d"),
            "purpose": "Anatomy ablation: does MR-derived UA length change "
                       "calibrated/held-out predictions?",
            "provenance_caveat": "The 72.2 cm reference path is derived from "
                                 "ONE approximately 34-week specimen; it is not "
                                 "a patient-specific length known at the "
                                 "19-29-week calibration points.",
            "length_construction": "L_UA = L_cord * sqrt(1+(2*pi*r*CI)^2), "
                                   "r=0.482 cm (specimen offset); CI from van "
                                   "Dijk 2002 (mean 0.17, 10-90th 0.07-0.30); "
                                   "single-axis variations only -- cord-length "
                                   "and coiling-index correlation not modeled",
            "procedure": "fit_ga_targets.calibrate, dx=0.73 cm, max_nfev=18, "
                         "index_definition=minimum, warm-started from the "
                         "72.2 cm Table-I fits; held-out/production at "
                         "dx=0.25 cm",
            "held_out_refs": {"pilot_24wk": PILOT_24,
                              "ozawa_24wk_mL_min_interp": OZAWA_24_ML_MIN},
        },
        "scenarios": SCENARIOS,
        "table_I_reference": ref_data["table_I"],
    }
    save(results)
    fixed_arm(results)
    refit_arm(results)
    log("ABLATION COMPLETE -> " + str(OUT_PATH))


if __name__ == "__main__":
    main()
