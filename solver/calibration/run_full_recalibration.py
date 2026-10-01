"""Run coarse-to-production UA recalibration and the coupled UV return model."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "calibration"))
import solve_arterial_flow_UA as ua
from fit_ga_targets import calibrate
from gestational_geometry import ua_radius_cm
from solve_venous_flow_UV import simulate_uv_return

SOURCE = json.loads((ROOT / "data" / "targets_and_original_fit.json").read_text(encoding="utf-8"))
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def production_check(fit, compliance=4e-5, include_uv=False):
    ga = fit["ga_weeks"]
    result = ua.simulate(
        140.0, ua_radius_cm(ga, allow_extrapolation=True),
        fit["R_total_dyn_s_cm5"], compliance,
        p0=fit["p0_dyn_cm2"], dp_inlet=fit["dp_inlet_dyn_cm2"],
        L=72.2, M=290, n_cycles=15, max_cycles=80,
        index_definition="minimum",
    )
    checked = {
        "metrics": result["metrics"], "converged": result["converged"],
        "fallback_fraction": result["outlet_fallback_frac"],
        "cycles": result["n_cycles_run"],
    }
    if include_uv:
        # Storage-consistent UV inflow: placental efflux through the distal
        # resistor of BOTH arteries (symmetric pair), 2*(Pwk-Pv)/R2 -- NOT the
        # arterial inflow q_b, which bypasses placental storage at waveform
        # level (cycle means coincide; see README "UV bookkeeping note").
        # R1_frac = 0.15 and Pv = 0 are simulate()'s defaults, used here.
        r2_eff = (1.0 - 0.15) * fit["R_total_dyn_s_cm5"]
        q_uv_total = 2.0 * result["Pwk"] / r2_eff
        uv = simulate_uv_return(
            result["t"], q_uv_total, ga,
            allow_geometry_extrapolation=False,
        )
        checked["uv"] = {k: v for k, v in uv.items() if np.isscalar(v)}
    return checked


def main():
    result_path = OUT / "full_recalibration.json"
    previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    results = {"table_I": previous.get("table_I", []), "table_II": [], "geometry": {
        "UA_length_cm": 72.2, "UV_length_cm": 61.78,
        "index_definition": "minimum/pre-systolic",
        "coarse_dx_cm": 0.73, "production_dx_cm": 72.2 / 289,
    }, "excluded": {
        "table_I_GA": [11, 33, 37],
        "reason": "UA diameter curve is directly anchored only over 12-30 weeks; extrapolated production calibration is not accepted as validated.",
    }}

    for row in SOURCE["table_I_original_L20cm"]["rows"]:
        ga, R, dp, pi, ri = row
        if not 12 <= ga <= 30:
            continue
        if any(item["ga_weeks"] == ga for item in results["table_I"]):
            continue
        print(f"Table I GA {ga}: fitting", flush=True)
        fit = calibrate(ga, 72.2, pi, ri, [R, dp], dx_cm=0.73,
                        max_nfev=18, index_definition="minimum",
                        allow_geometry_extrapolation=False)
        fit["targets"] = {"PI": pi, "RI": ri}
        print(f"Table I GA {ga}: production check", flush=True)
        fit["production"] = production_check(fit, include_uv=False)
        results["table_I"].append(fit)
        (OUT / "full_recalibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for row in SOURCE["table_II_original_L20cm"]["rows"]:
        ga, R, dp, p0, pi, ri, flow = row
        print(f"Table II GA {ga}: fitting", flush=True)
        fit = calibrate(ga, 72.2, pi, ri, [R, dp, p0], target_flow=flow,
                        dx_cm=0.73, max_nfev=32, index_definition="minimum",
                        allow_geometry_extrapolation=True)
        fit["targets"] = {"PI": pi, "RI": ri, "UV_flow_mL_min": flow}
        print(f"Table II GA {ga}: production check and UV coupling", flush=True)
        fit["production"] = production_check(fit, include_uv=True)
        results["table_II"].append(fit)
        (OUT / "full_recalibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(OUT / "full_recalibration.json", flush=True)


if __name__ == "__main__":
    main()
