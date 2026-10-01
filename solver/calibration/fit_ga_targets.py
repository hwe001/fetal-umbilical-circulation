"""Convergence-aware calibration of the 1D UA model in log-parameter space."""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import solve_arterial_flow_UA as ua
from gestational_geometry import ua_radius_cm


def calibrate(ga, length_cm, target_pi, target_ri, initial, target_flow=None,
              bpm=140.0, compliance=4.0e-5, dx_cm=0.25,
              index_definition="minimum", max_nfev=60,
              allow_geometry_extrapolation=False):
    """Fit positive parameters using log transforms and honest failure rejection."""
    n_params = 3 if target_flow is not None else 2
    if len(initial) != n_params:
        raise ValueError(f"expected {n_params} initial parameters")
    M = int(np.ceil(length_cm / dx_cm)) + 1

    @lru_cache(maxsize=512)
    def evaluate(log_key):
        pars = np.exp(np.asarray(log_key))
        R, dp = pars[:2]
        p0 = pars[2] if n_params == 3 else ua.P0_DEFAULT
        result = ua.simulate(
            bpm=bpm, r0=ua_radius_cm(ga, allow_extrapolation=allow_geometry_extrapolation),
            R_total=R, C_total=compliance, p0=p0, dp_inlet=dp,
            L=length_cm, M=M, n_cycles=15, max_cycles=80,
            index_definition=index_definition,
        )
        metrics = result["metrics"]
        return metrics, result["converged"], result["outlet_fallback_frac"]

    def residual(log_pars):
        # Rounding only stabilizes cache keys; it does not quantize model inputs
        # at a meaningful physiological scale.
        key = tuple(np.round(log_pars, 10))
        metrics, converged, fallback = evaluate(key)
        values = [(metrics["PI"] - target_pi) / 0.02,
                  (metrics["RI"] - target_ri) / 0.02]
        if target_flow is not None:
            values.append((metrics["total_two_UA_flow_mL_min"] - target_flow) /
                          max(0.05 * target_flow, 1.0))
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            return np.full(n_params, 1.0e3)
        # Penalize failed solves without returning a parameter-independent
        # constant (which gives a zero numerical Jacobian and false optimizer
        # success at the starting point).
        if not converged:
            values += np.where(values >= 0, 20.0, -20.0)
        if fallback > 1e-4:
            values += np.where(values >= 0, 50.0 * fallback, -50.0 * fallback)
        return values

    initial = np.asarray(initial, dtype=float)
    global_lower = np.asarray([100.0, 100.0] + ([1.0e3] if n_params == 3 else []))
    global_upper = np.asarray([2.0e6, 2.0e5] + ([2.0e5] if n_params == 3 else []))
    # Flow-constrained fits often require a much lower equivalent placental
    # resistance after the UA trunk is lengthened. Do not let a legacy local
    # bound masquerade as physiological incompatibility.
    lower = np.log(global_lower if target_flow is not None
                   else np.maximum(global_lower, initial / 4.0))
    upper = np.log(np.minimum(global_upper, initial * 4.0))
    fit = least_squares(
        residual, np.log(initial), bounds=(lower, upper), method="trf",
        x_scale="jac", diff_step=0.03, max_nfev=max_nfev,
    )
    pars = np.exp(fit.x)
    metrics, converged, fallback = evaluate(tuple(np.round(fit.x, 10)))
    return {
        "ga_weeks": ga, "length_cm": length_cm,
        "R_total_dyn_s_cm5": float(pars[0]),
        "dp_inlet_dyn_cm2": float(pars[1]),
        "p0_dyn_cm2": float(pars[2]) if n_params == 3 else ua.P0_DEFAULT,
        "metrics": metrics, "periodic_convergence": bool(converged),
        "outlet_fallback_fraction": float(fallback),
        "optimizer_success": bool(fit.success), "optimizer_message": fit.message,
        "cost": float(fit.cost), "nfev": int(fit.nfev),
        "index_definition": index_definition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga", type=float, required=True)
    parser.add_argument("--length", type=float, default=72.2)
    parser.add_argument("--pi", type=float, required=True)
    parser.add_argument("--ri", type=float, required=True)
    parser.add_argument("--initial", type=float, nargs="+", required=True)
    parser.add_argument("--flow", type=float)
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--allow-geometry-extrapolation", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index-definition", choices=("minimum", "end_cycle", "legacy_floor"),
                        default="minimum")
    args = parser.parse_args()
    result = calibrate(args.ga, args.length, args.pi, args.ri, args.initial,
                       target_flow=args.flow,
                       dx_cm=args.dx, max_nfev=args.max_nfev,
                       allow_geometry_extrapolation=args.allow_geometry_extrapolation,
                       index_definition=args.index_definition)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
