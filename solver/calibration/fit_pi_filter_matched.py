"""Fit the placental pi-filter to the matched 1D UA benchmark.

Stage 3 of the handover's immediate tasks ("fit the pi-filter to the
matched 1D UA case before fitting patient data").  The 1D reference is the
project benchmark (outputs/initial_1d_0d_results.md): 140 bpm, r0=0.13 cm,
R_total=3.0e4, C_total=4.0e-5, p0=3.0e4, dp_inlet=1.2e4, L=72.2 cm ->
PI 1.577, RI 0.840, S/D 6.255, PSV 11.08, EDV 1.77, 35.29 mL/min (two UAs).

Calibration staging (module docstring of placental_pi_filter.py has the
math): cycle-averaging the periodic ODEs pins exactly ONE linear
constraint -- the sum Rs + R_UV + Rout through the mean-flow target
(stage 1) -- so with Ls/R_UV pinned from geometry, the remaining shape
information in (PI, RI) constrains essentially one resistance split
(stage 3, mode "dc").  The compliances Cm/Cuv are NOT identifiable from UA
Doppler indices at this operating point (1/(omega*C) << Rs shunts the
distal load); stage 4 quantifies that with a sensitivity SVD and sweep
grids instead of pretending the fit determines them.

Fit conventions follow calibration/fit_ga_targets.py: positive parameters
in log space, least_squares(method="trf", x_scale="jac", diff_step=0.03),
and a sign-preserving penalty for non-converged/non-physiologic solves so
the numerical Jacobian never silently zeroes.

Known, deliberate outcome (planning-time verification, "F1"): the 1D
reference reports velocity as q/A(t) with a PULSATING outlet area, while
the pi-filter has rigid A0.  Area pulsatility inflates the 1D TAV by
~6.6%, so a rigid-area model matching PSV/EDV/RI/mean flow lands at
PI ~ 1.69, not 1.577.  This is computed explicitly (PI_expected_rigid),
not hand-waved, and must NOT be "fixed" by tuning Cm.  Consequence for
the default target mode: PI is the ONLY Doppler index containing TAV in
its denominator, hence the only one contaminated by the area convention;
the default therefore fits (PSV, EDV) ("--target-mode absolute") and
reports PI against the rigid-area expectation.  Fitting (PI, RI) with
equal weights ("--target-mode indices") is still available; its
least-squares optimum is a compromise (measured: Rs ~ 5.48e4,
PI 1.61 / RI 0.820 -- both targets missed) because the two targets pull
the single resistance split in opposite directions.

Identifiability implementation note: stage 4 uses a sensitivity SVD plus
no-refit sweep grids (1D per parameter; 2D Cm x Cuv plateau; 2D Rs x Ls
ridge with the DC closure).  Refit-based profiling was scoped out: along a
flat direction a refit cannot change the cost by construction, and the
grids make the flat directions visible directly at a fraction of the
runtime.

Usage:
    python calibration/fit_pi_filter_matched.py            # full pipeline
    python calibration/fit_pi_filter_matched.py --mode dc_shape
    python calibration/fit_pi_filter_matched.py --no-1d --no-sweep
"""

import argparse
import json
import logging
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import placental_pi_filter as pf
import solve_arterial_flow_UA as ua

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BPM = pf.MATCHED_1D_CASE["bpm"]
TARGETS = {
    "PI": pf.MATCHED_1D_CASE["PI"],
    "RI": pf.MATCHED_1D_CASE["RI"],
    "S_D": pf.MATCHED_1D_CASE["S_D"],
    "PSV_cm_s": pf.MATCHED_1D_CASE["PSV_cm_s"],
    "EDV_cm_s": pf.MATCHED_1D_CASE["EDV_cm_s"],
    "total_two_UA_flow_mL_min": pf.MATCHED_1D_CASE["total_two_UA_flow_mL_min"],
}
# residual scales ~ the precision worth fitting to
SCALES = {"PI": 0.05, "RI": 0.02, "S_D": 0.25, "PSV_cm_s": 0.10,
          "EDV_cm_s": 0.10, "Q_rel": 0.05}

FREE_NAMES = {"dc": ("Rs",),
              "dc_shape": ("Rs", "Cm", "Cuv"),
              "free": ("Rs", "Ls", "Cm", "Cuv", "Rout")}
# 0 = S/D exactly redundant with RI (S_D = 1/(1-RI)); 0 = mean flow exact
# by the DC closure in dc/dc_shape modes
DEFAULT_WEIGHTS = {"PI": 1.0, "RI": 1.0, "S_D": 0.0, "Q": 0.0}

PARAM_SOURCES = {
    "Rs": "UA conduit prior, coefficient 22 (1D Olufsen friction, "
          "solve_arterial_flow_UA.friction_source)",
    "Ls": "UA inertance rho*L/A at r0=0.13 cm, L=72.2 cm",
    "Cm": "prior = matched 1D Windkessel C_total (NOT identifiable from "
          "UA Doppler; pins the admissible family)",
    "R_UV": "UV Poiseuille (coefficient 8) at GA-26 diameter 5.8 mm, "
            "consistent with solve_venous_flow_UV",
    "Cuv": "tube-law compliance of the UV (FLAGGED: arterial Eh/r0 law "
           "applied to a vein; not independently measured)",
    "Rout": "DC closure: R_series_total - Rs - R_UV (unresolved placental "
            "microcirculation + venous runoff)",
}


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------
def residual_values(metrics, weights, target_mode):
    """Normalized residual vector for one evaluation."""
    if target_mode == "absolute":
        values = [
            weights["PI"] * (metrics["PSV_cm_s"] - TARGETS["PSV_cm_s"])
            / SCALES["PSV_cm_s"],
            weights["RI"] * (metrics["EDV_cm_s"] - TARGETS["EDV_cm_s"])
            / SCALES["EDV_cm_s"],
        ]
    else:
        values = [
            weights["PI"] * (metrics["PI"] - TARGETS["PI"]) / SCALES["PI"],
            weights["RI"] * (metrics["RI"] - TARGETS["RI"]) / SCALES["RI"],
        ]
    if weights["S_D"]:
        values.append(weights["S_D"] * (metrics["S_D"] - TARGETS["S_D"])
                      / SCALES["S_D"])
    if weights["Q"]:
        values.append(
            weights["Q"]
            * (metrics["total_two_UA_flow_mL_min"]
               - TARGETS["total_two_UA_flow_mL_min"])
            / (SCALES["Q_rel"] * TARGETS["total_two_UA_flow_mL_min"]))
    return np.asarray(values, dtype=float)


class Evaluator:
    """Cached pi-filter evaluation in log-parameter space."""

    def __init__(self, base, r_series_total, mode, weights, target_mode,
                 n_res):
        self.base = base
        self.r_series_total = r_series_total
        self.mode = mode
        self.weights = weights
        self.target_mode = target_mode
        self.n_res = n_res
        self.free_names = FREE_NAMES[mode]

    def build_params(self, log_vals):
        overrides = dict(zip(self.free_names, np.exp(np.asarray(log_vals))))
        p = pf.with_overrides(self.base, **overrides)
        if self.mode != "free":
            # DC closure keeps the mean flow exact at every evaluation
            p = pf.with_overrides(
                p, Rout=self.r_series_total - p.Rs - p.R_UV)
        return p

    @lru_cache(maxsize=4096)
    def evaluate(self, log_key):
        """log_key: tuple of log-parameters.  Returns (residuals, info)."""
        p = self.build_params(np.asarray(log_key))
        try:
            r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
        except ValueError:
            return np.full(self.n_res, 1.0e3), {"diverged": True}
        metrics = r["metrics"]
        values = residual_values(metrics, self.weights, self.target_mode)
        bad = (not r["converged"]
               or not r["flags"]["positive_edv"]
               or not r["flags"]["positive_pressure"])
        if bad:
            # sign-preserving: a parameter-independent constant would give a
            # zero numerical Jacobian and false optimizer success
            values = values + np.where(values >= 0.0, 20.0, -20.0)
        info = {
            "PI": metrics["PI"], "RI": metrics["RI"], "S_D": metrics["S_D"],
            "PSV_cm_s": metrics["PSV_cm_s"], "EDV_cm_s": metrics["EDV_cm_s"],
            "total_two_UA_flow_mL_min": metrics["total_two_UA_flow_mL_min"],
            "converged": r["converged"],
        }
        return values, info

    def cost_at(self, p):
        """Residual cost for an explicit parameter set (sweeps/grids)."""
        try:
            r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
        except ValueError:
            return np.inf, None
        values = residual_values(r["metrics"], self.weights, self.target_mode)
        bad = (not r["converged"]
               or not r["flags"]["positive_edv"]
               or not r["flags"]["positive_pressure"])
        if bad:
            values = values + np.where(values >= 0.0, 20.0, -20.0)
        return float(0.5 * np.sum(values ** 2)), r


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage1_dc(log):
    q_target = TARGETS["total_two_UA_flow_mL_min"] * pf.ML_MIN_TO_ML_S / 2.0
    r_series = (pf.MATCHED_1D_CASE["p0"] - 0.0) / q_target
    log.info("stage 1 (DC mean-flow constraint): R_series_total = "
             "(p0 - Pout)/Q_per_artery = %.1f dyn*s/cm^5 "
             "(Q_per_artery = %.6f mL/s)", r_series, q_target)
    return r_series


def stage2_priors(log):
    base = pf.default_pi_filter_params()
    rows = []
    q_target = TARGETS["total_two_UA_flow_mL_min"] * pf.ML_MIN_TO_ML_S / 2.0
    r_series = base.Rs + base.R_UV + base.Rout
    for name, value, unit in (
            ("Rs", base.Rs, "dyn*s/cm^5"), ("Ls", base.Ls, "dyn*s^2/cm^5"),
            ("Cm", base.Cm, "cm^5/dyn"), ("R_UV", base.R_UV, "dyn*s/cm^5"),
            ("Cuv", base.Cuv, "cm^5/dyn"), ("Rout", base.Rout, "dyn*s/cm^5"),
            ("A0", base.A0, "cm^2")):
        rows.append({"param": name, "value": float(value), "unit": unit,
                     "source": PARAM_SOURCES.get(name, "derived"),
                     "role": "prior (pinned)" if name not in FREE_NAMES["dc"]
                             else "prior (fitted in mode dc)"})
    log.info("stage 2 (priors):")
    for row in rows:
        log.info("  %-5s = %12.5g %-14s %s", row["param"], row["value"],
                 row["unit"], row["source"])
    log.info("  R_UV / R_series_total = %.5f (the cord vein is hydraulically "
             "transparent; the handover's R_UV path is dominated by the "
             "villous circulation)", base.R_UV / r_series)
    return base, r_series, rows


def stage3_fit(log, base, r_series_total, mode, weights, target_mode,
               max_nfev):
    n_res = 2 + (1 if weights["S_D"] else 0) + (1 if weights["Q"] else 0)
    evaluator = Evaluator(base, r_series_total, mode, weights, target_mode,
                          n_res)

    p0_values = np.array([getattr(base, name) for name in
                          evaluator.free_names])
    bounds = {}
    bounds["Rs"] = (base.Rs / 4.0, base.Rs * 4.0)
    bounds["Ls"] = (base.Ls / 8.0, base.Ls * 8.0)
    bounds["Cm"] = (1.0e-8, 1.0e-2)
    bounds["Cuv"] = (1.0e-8, 1.0e-2)
    bounds["Rout"] = (1.0e2, 1.0e6)
    lower = np.array([np.log(bounds[name][0]) for name in
                      evaluator.free_names])
    upper = np.array([np.log(bounds[name][1]) for name in
                      evaluator.free_names])

    def residual(log_pars):
        values, _info = evaluator.evaluate(tuple(np.round(log_pars, 10)))
        return values

    fit = least_squares(residual, np.log(p0_values), bounds=(lower, upper),
                        method="trf", x_scale="jac", diff_step=1e-3,
                        max_nfev=max_nfev)
    fitted = evaluator.build_params(fit.x)
    values, info = evaluator.evaluate(tuple(np.round(fit.x, 10)))
    log.info("stage 3 (fit, mode=%s): success=%s nfev=%d cost=%.4g",
             mode, fit.success, fit.nfev, fit.cost)
    for name, value in zip(evaluator.free_names, np.exp(fit.x)):
        log.info("  fitted %-5s = %.6g", name, value)
    log.info("  achieved: PI=%.4f RI=%.4f S/D=%.3f PSV=%.3f EDV=%.3f "
             "Q2UA=%.3f", info["PI"], info["RI"], info["S_D"],
             info["PSV_cm_s"], info["EDV_cm_s"],
             info["total_two_UA_flow_mL_min"])
    return {
        "fitted": fitted, "evaluator": evaluator, "fit": fit,
        "info": info, "mode": mode, "free_names": evaluator.free_names,
        "residuals": values,
    }


def stage4_identifiability(log, base, fitted, r_series_total, weights,
                           target_mode, do_sweep):
    """Sensitivity SVD + no-refit sweeps (see module docstring for why)."""
    # SVD over the five raw parameters at the fitted point; the flow target
    # is always active in the raw space (Rout is NOT DC-closed here)
    raw_weights = dict(weights, Q=1.0)
    theta = ["Rs", "Ls", "Cm", "Cuv", "Rout"]
    theta0 = np.array([getattr(fitted, name) for name in theta])

    def raw_residuals(theta_vals):
        p = pf.with_overrides(fitted, **dict(zip(theta, theta_vals)))
        try:
            r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
        except ValueError:
            return np.full(3, 1.0e3)
        values = residual_values(r["metrics"], raw_weights, target_mode)
        bad = (not r["converged"] or not r["flags"]["positive_edv"]
               or not r["flags"]["positive_pressure"])
        if bad:
            values = values + np.where(values >= 0.0, 20.0, -20.0)
        return values

    r0 = raw_residuals(theta0)
    jac = np.empty((len(r0), len(theta)))
    for i in range(len(theta)):
        h = 1e-4 * theta0[i]
        step = np.zeros(len(theta))
        step[i] = h
        jac[:, i] = (raw_residuals(theta0 + step)
                     - raw_residuals(theta0 - step)) / (2.0 * h)
    # full_matrices=True: for an (n_res x n_param) Jacobian with
    # n_param > n_res, the rows of vt beyond rank n_res span the EXACT null
    # space -- the flat directions the fit cannot see at all
    _u, sv, vt = np.linalg.svd(jac, full_matrices=True)
    sv = sv[:min(jac.shape)]
    cond = float(sv.max() / max(sv.min(), 1e-30))
    flat = []
    rank = int(np.sum(sv > sv.max() * 1e-8))
    for k in range(rank, vt.shape[0]):
        combo = ", ".join(f"{c:+.2f} log {n}" for c, n in zip(vt[k], theta)
                          if abs(c) > 0.15)
        flat.append({"singular_value": 0.0, "direction": combo,
                     "kind": "exact null direction"})
    if sv.min() < sv.max() * 1e-2:
        combo = ", ".join(f"{c:+.2f} log {n}" for c, n in zip(vt[len(sv) - 1],
                                                              theta)
                          if abs(c) > 0.15)
        flat.append({"singular_value": float(sv.min()), "direction": combo,
                     "kind": "weakly identified (sigma/sigma_max < 1e-2)"})
    log.info("stage 4 (identifiability): singular values = %s (rank %d)",
             np.array2string(sv, precision=3), rank)
    for entry in flat:
        log.info("  %s (sigma=%.3g): %s", entry["kind"],
                 entry["singular_value"], entry["direction"])
    if not flat:
        log.info("  no flat or weak directions found")

    artifacts = {"singular_values": sv.tolist(), "condition_number": cond,
                 "flat_directions": flat, "theta_names": theta,
                 "theta0": theta0.tolist()}

    if not do_sweep:
        return artifacts

    # ---- 1D sweeps (no refit).  Flow target active: PI/RI are invariant to
    # a pure flow rescaling, so sweeping Rout with the flow residual off
    # would look spuriously flat.
    sweep_factors = np.geomspace(0.25, 4.0, 13)
    sweeps = {}
    for i, name in enumerate(theta):
        costs, values = [], []
        for f in sweep_factors:
            p = pf.with_overrides(fitted, **{name: theta0[i] * f})
            if name != "Rout" and fitted.n_arteries == 1:
                p = pf.with_overrides(p, Rout=r_series_total - p.Rs - p.R_UV)
            cost, _r = _cost_of(p, raw_weights, target_mode)
            costs.append(cost)
            values.append(float(theta0[i] * f))
        sweeps[name] = {"factors": sweep_factors.tolist(),
                        "values": values, "costs": costs}
        finite = np.isfinite(costs)
        if finite.any():
            j_min = np.min(np.array(costs)[finite])
            admissible = {f"delta_{d}": [
                float(v) for v, c in zip(values, costs)
                if np.isfinite(c) and c <= j_min + d]
                for d in (1, 4, 9)}
            sweeps[name]["j_min"] = float(j_min)
            sweeps[name]["admissible"] = admissible
            log.info("  sweep %-5s: j_min=%.4g admissible@delta1=[%.4g .. %.4g]",
                     name, j_min, min(admissible["delta_1"]),
                     max(admissible["delta_1"]))
    artifacts["sweeps_1d"] = sweeps

    # ---- 2D grids -------------------------------------------------------
    log_c = np.linspace(-8.0, -2.0, 11)
    grid_cm_cuv = {"log10_Cm": log_c.tolist(), "log10_Cuv": log_c.tolist(),
                   "cost": []}
    for cm in 10.0 ** log_c:
        row = []
        for cuv in 10.0 ** log_c:
            p = pf.with_overrides(fitted, Cm=cm, Cuv=cuv)
            cost, _r = _cost_of(p, raw_weights, target_mode)
            row.append(cost if np.isfinite(cost) else None)
        grid_cm_cuv["cost"].append(row)
    artifacts["grid_Cm_Cuv"] = grid_cm_cuv
    flat_count = sum(
        1 for row in grid_cm_cuv["cost"] for c in row
        if c is not None and c <= np.nanmin([c2 for c2 in grid_cm_cuv["cost"]
                                             if c2 is not None]) + 1.0)
    log.info("  grid Cm x Cuv (11x11 over 1e-8..1e-2): %d/%d cells within "
             "delta_J<=1 of the minimum -- the compliance plateau",
             flat_count, 121)

    rs_grid = np.geomspace(0.25, 4.0, 11)
    grid_rs_ls = {"Rs_factors": rs_grid.tolist(), "Ls_factors": rs_grid.tolist(),
                  "cost": []}
    for rs_f in rs_grid:
        row = []
        for ls_f in rs_grid:
            p = pf.with_overrides(fitted, Rs=fitted.Rs * rs_f,
                                  Ls=fitted.Ls * ls_f)
            p = pf.with_overrides(p, Rout=r_series_total - p.Rs - p.R_UV)
            cost, _r = _cost_of(p, raw_weights, target_mode)
            row.append(cost if np.isfinite(cost) else None)
        grid_rs_ls["cost"].append(row)
    artifacts["grid_Rs_Ls"] = grid_rs_ls
    costs = np.array([[c if c is not None else np.nan
                       for c in row] for row in grid_rs_ls["cost"]])
    ridge = np.nanmin(costs)
    ridge_cells = int(np.sum(costs <= ridge + 1.0))
    log.info("  grid Rs x Ls (11x11, Rout DC-coupled): %d/%d cells within "
             "delta_J<=1 of the minimum -- axes are well identified; the "
             "flat directions are JOINT Rs-Ls-Rout moves (see SVD null "
             "directions), invisible on single-parameter axes",
             ridge_cells, 121)
    return artifacts


def _cost_of(p, weights, target_mode):
    try:
        r = pf.simulate_pi_filter(BPM, p, init="newton_periodic")
    except ValueError:
        return np.inf, None
    values = residual_values(r["metrics"], weights, target_mode)
    bad = (not r["converged"] or not r["flags"]["positive_edv"]
           or not r["flags"]["positive_pressure"])
    if bad:
        values = values + np.where(values >= 0.0, 20.0, -20.0)
    return float(0.5 * np.sum(values ** 2)), r


def run_1d_reference(log):
    """1D benchmark + the area-pulsatility (F1) diagnostic."""
    log.info("running the 1D reference (ua.simulate, ~20-40 s)...")
    result = ua.simulate(
        bpm=BPM, r0=pf.MATCHED_1D_CASE["r0_cm"],
        R_total=pf.MATCHED_1D_CASE["R_total"],
        C_total=pf.MATCHED_1D_CASE["C_total"],
        p0=pf.MATCHED_1D_CASE["p0"], dp_inlet=pf.MATCHED_1D_CASE["dp_inlet"],
        L=pf.MATCHED_1D_CASE["L_cm"], index_definition="minimum",
    )
    m1 = result["metrics"]
    a0 = np.pi * pf.MATCHED_1D_CASE["r0_cm"] ** 2
    tav_rigid = m1["mean_flow_mL_s_per_artery"] / a0
    inflation = m1["TAV_cm_s"] / tav_rigid
    pi_expected_rigid = ((m1["PSV_cm_s"] - m1["EDV_cm_s"]) / tav_rigid)
    diagnostic = {
        "TAV_1D_cm_s": m1["TAV_cm_s"],
        "TAV_rigid_cm_s": tav_rigid,
        "area_pulsatility_tav_inflation": inflation,
        "PI_1D": m1["PI"],
        "PI_expected_rigid": pi_expected_rigid,
        "converged": result["converged"],
    }
    log.info("1D reference: PI=%.4f RI=%.4f S/D=%.3f converged=%s",
             m1["PI"], m1["RI"], m1["S_D"], result["converged"])
    log.info("F1 diagnostic: 1D TAV=%.4f vs rigid-area TAV=%.4f "
             "(inflation x%.4f) -> PI_expected_rigid=%.4f "
             "(1D reported PI=%.4f)", m1["TAV_cm_s"], tav_rigid, inflation,
             pi_expected_rigid, m1["PI"])
    return result, diagnostic


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(res_1d, res_pf, path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    m_pf = res_pf["metrics"]

    ax = axes[0, 0]
    if res_1d is not None:
        ax.plot(res_1d["t"], res_1d["u_outlet"], lw=2.0,
                label="1D (pulsating area)")
    ax.plot(res_pf["t"], res_pf["Q_UA"] / res_pf["A0"], lw=2.0, ls="--",
            label="pi-filter (rigid $A_0$)")
    ax2 = ax.twinx()
    ax2.plot(res_pf["t"], res_pf["p_in"] / pf.MMHG_TO_DYN_CM2, lw=0.8,
             alpha=0.4, color="grey")
    ax2.set_ylabel("$p_{in}$ (mmHg)", alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("velocity (cm/s)")
    ax.set_title("Outlet velocity waveform (one cycle)")
    ax.legend(loc="upper right", fontsize=11)
    ax.annotate(f"pi-filter PSV {m_pf['PSV_cm_s']:.2f}  "
                f"EDV {m_pf['EDV_cm_s']:.2f} cm/s",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=10)
    ax.annotate("the pi-filter leads the 1D by ~0.1 s: the 1D outlet sees the\n"
                "pulse after a ~72 cm wave-transit delay, which no lumped\n"
                "model can reproduce; Doppler indices are phase-invariant",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right",
                fontsize=7, alpha=0.85)

    ax = axes[0, 1]
    labels = ["PSV", "EDV", "TAV", "PI", "RI", "S/D", "Q 2UA"]
    one_d = [TARGETS["PSV_cm_s"], TARGETS["EDV_cm_s"], None, TARGETS["PI"],
             TARGETS["RI"], TARGETS["S_D"],
             TARGETS["total_two_UA_flow_mL_min"]]
    if res_1d is not None:
        m1 = res_1d["metrics"]
        one_d = [m1["PSV_cm_s"], m1["EDV_cm_s"], m1["TAV_cm_s"], m1["PI"],
                 m1["RI"], m1["S_D"], m1["total_two_UA_flow_mL_min"]]
    pfv = [m_pf["PSV_cm_s"], m_pf["EDV_cm_s"], m_pf["TAV_cm_s"], m_pf["PI"],
           m_pf["RI"], m_pf["S_D"], m_pf["total_two_UA_flow_mL_min"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, one_d, 0.36, label="1D")
    ax.bar(x + 0.18, pfv, 0.36, label="pi-filter")
    ax.set_xticks(x, labels)
    ax.set_title("Doppler indices")
    ax.legend(fontsize=11)
    pi_ratio = m_pf["PI"] / TARGETS["PI"]
    ax.annotate(
        f"PI: {pi_ratio - 1.0:+.1%} vs 1D — rigid-$A_0$ TAV vs the 1D's "
        f"pulsating-area TAV (F1), not a compliance effect",
        xy=(0.5, -0.22), xycoords="axes fraction", ha="center", fontsize=7)

    ax = axes[1, 0]
    ax.plot(res_pf["t"], res_pf["p_in"] / pf.MMHG_TO_DYN_CM2, lw=1.0,
            color="grey", label="$p_{in}$")
    ax.plot(res_pf["t"], res_pf["Pm"] / pf.MMHG_TO_DYN_CM2, lw=2.0,
            label="$P_m$ (placental)")
    ax.plot(res_pf["t"], res_pf["P_UV"] / pf.MMHG_TO_DYN_CM2, lw=2.0, ls="--",
            label="$P_{UV}$ (venous)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("pressure (mmHg)")
    ax.set_title("Pi-filter pressures")
    ax.legend(fontsize=11)

    ax = axes[1, 1]
    # dPm_dt / dP_UV_dt are stored as dP/dt in dyn/cm^2/s; the storage rate
    # in flow units is C*dP/dt (mL/s)
    cm = res_pf["params"]["Cm"]
    cuv = res_pf["params"]["Cuv"]
    ax.plot(res_pf["t"], cm * res_pf["dPm_dt"], lw=1.5,
            label="$C_m\\,dP_m/dt$")
    ax.plot(res_pf["t"], cuv * res_pf["dP_UV_dt"], lw=1.5, ls="--",
            label="$C_{UV}\\,dP_{UV}/dt$")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("storage rate (mL/s)")
    ax.set_title("Compliance storage (zero cycle-mean in the periodic state)")
    ax.legend(fontsize=11)

    fig.suptitle("Placental pi-filter fitted to the matched 1D UA benchmark "
                 "(140 bpm, r0 = 0.13 cm)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# serialization / main
# ---------------------------------------------------------------------------
def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("dc", "dc_shape", "free"),
                        default="dc")
    parser.add_argument("--target-mode", choices=("indices", "absolute"),
                        default="absolute",
                        help="'absolute' (default) fits (PSV, EDV) -- the raw "
                             "waveform observables, immune to the 1D's "
                             "pulsating-area convention (F1). 'indices' fits "
                             "(PI, RI); because the rigid-area model cannot "
                             "reproduce the 1D's PI (F1), the least-squares "
                             "optimum there is a compromise matching neither "
                             "systole nor diastole exactly.")
    parser.add_argument("--weight-sd", type=float, default=DEFAULT_WEIGHTS["S_D"])
    parser.add_argument("--weight-q", type=float, default=DEFAULT_WEIGHTS["Q"])
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--no-1d", action="store_true",
                        help="skip the 1D reference run (figure panel 1 and "
                             "the F1 diagnostic need it)")
    parser.add_argument("--no-sweep", action="store_true",
                        help="skip the identifiability sweeps (SVD always runs)")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="extra directory to receive a copy of the figure")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger()
    log.addHandler(logging.FileHandler(ROOT / "logs" / "pi_filter_matched.log"))

    weights = dict(DEFAULT_WEIGHTS, S_D=args.weight_sd, Q=args.weight_q)
    if args.mode == "free" and weights["Q"] == 0:
        log.info("mode 'free' leaves the mean flow unconstrained; forcing "
                 "--weight-q 1")
        weights["Q"] = 1.0

    r_series_total = stage1_dc(log)
    base, r_series_check, prior_rows = stage2_priors(log)
    fit_result = stage3_fit(log, base, r_series_total, args.mode, weights,
                            args.target_mode, args.max_nfev)
    fitted = fit_result["fitted"]

    res_1d, f1_diagnostic = (None, None)
    if not args.no_1d:
        res_1d, f1_diagnostic = run_1d_reference(log)

    artifacts = stage4_identifiability(log, base, fitted, r_series_total,
                                       weights, args.target_mode,
                                       do_sweep=not args.no_sweep)

    # final production-quality run of the fitted parameter set (standard
    # warm start, tau-driven cycle budget -- not the Newton fast path)
    final = pf.simulate_pi_filter(BPM, fitted)
    m = final["metrics"]
    log.info("final run: converged=%s n_cycles=%d PI=%.4f RI=%.4f S/D=%.3f "
             "PSV=%.3f EDV=%.3f Q2UA=%.3f", final["converged"],
             final["n_cycles_run"], m["PI"], m["RI"], m["S_D"],
             m["PSV_cm_s"], m["EDV_cm_s"], m["total_two_UA_flow_mL_min"])

    payload = {
        "mode": args.mode,
        "target_mode": args.target_mode,
        "weights": weights,
        "targets": TARGETS,
        "scales": SCALES,
        "R_series_total_dyn_s_cm5": r_series_total,
        "priors": prior_rows,
        "fitted_params": {
            "Rs": fitted.Rs, "Ls": fitted.Ls, "Cm": fitted.Cm,
            "R_UV": fitted.R_UV, "Cuv": fitted.Cuv, "Rout": fitted.Rout,
            "Rp": fitted.Rp, "Pout": fitted.Pout, "A0": fitted.A0,
            "n_arteries": fitted.n_arteries},
        "free_names": list(fit_result["free_names"]),
        "optimizer": {"success": bool(fit_result["fit"].success),
                      "message": fit_result["fit"].message,
                      "cost": float(fit_result["fit"].cost),
                      "nfev": int(fit_result["fit"].nfev)},
        "achieved": {k: v for k, v in m.items()},
        "flags": final["flags"],
        "diagnostics": {k: v for k, v in final["diagnostics"].items()},
        "flow_conservation": final["flow_conservation"],
        "f1_diagnostic": f1_diagnostic,
        "identifiability": artifacts,
    }

    out_json = ROOT / "results" / "pi_filter_matched.json"
    out_json.write_text(json.dumps(payload, indent=2, default=json_default),
                        encoding="utf-8")
    log.info("wrote %s", out_json)

    if not args.no_figure:
        out_png = ROOT / "figures" / "pi_filter_matched.png"
        make_figure(res_1d, final, out_png)
        log.info("wrote %s", out_png)
        if args.results_dir is not None:
            try:
                copy_to = Path(args.results_dir) / "pi_filter_matched_plots.png"
                shutil.copyfile(out_png, copy_to)
                log.info("copied figure to %s", copy_to)
            except OSError as exc:
                log.warning("could not copy figure to results dir: %s", exc)


if __name__ == "__main__":
    main()
