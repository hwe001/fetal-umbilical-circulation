"""Placental pi-filter: 3-state lumped model of the fetoplacental circulation.

Specified in the project handover
(`placenta_1d_0d_pi_filter_handover.md`, 2026-09-15) as the replacement for
the 2-state UA surrogate in ``solve_0d_model.py``, whose matched-1D failure
mode is excessive pulsatility and absent diastolic runoff (PI 2.54 vs 1.58,
S/D capped at 20 vs 6.25 -- see ``outputs/matched_1d_0d_results.md``).

Circuit and governing equations (handover, verbatim)::

    P_UA,in -> [Rs + Ls] -> Pm -> [R_UV] -> P_UV -> [Rout] -> Pout

    Ls  dQ_UA/dt  = P_UA,in(t) - Rs*Q_UA - Pm
    Cm  dPm/dt    = Q_UA - (Pm - P_UV)/R_UV
    Cuv dP_UV/dt  = (Pm - P_UV)/R_UV - (P_UV - Pout)/Rout

State vector ``y = [Q_UA, Pm, P_UV]``.  ``Pm`` is the placental
(microcirculatory) pressure, ``P_UV`` the umbilical-vein-side pressure.  An
optional weak shunt leak ``Rp`` from ``Pm`` to ``Pout`` (the hepatic
pi-filter's characteristic shunt leg, ``pi_filter_healthy_infant_model.py``)
is supported but DEFAULTS TO OFF (``Rp = inf``) because the handover
equations omit it; the hepatic parameters themselves are never reused here.

Conventions
-----------
* **Per-artery parameters.**  All parameter values describe ONE umbilical
  artery.  Under the handover's option A (two identical UAs, 50:50 split)
  the two arteries are exact copies, so the split never enters the ODEs --
  only the reporting does.  ``to_shared_bed()``/``from_shared_bed()``
  convert to/from the exactly-equivalent merged circuit.
* **Units.**  CGS-hemodynamics throughout, matching the 1D core
  (``solve_arterial_flow_UA.py``): pressure dyn/cm^2, flow cm^3/s (= mL/s),
  resistance dyn*s/cm^5, inertance dyn*s^2/cm^5, compliance cm^5/dyn.
  1 mmHg = 1333.22 dyn/cm^2; only reporting converts units.
* **Metrics.**  All Doppler indices come from
  ``solve_arterial_flow_UA.waveform_metrics`` with the "minimum" EDV
  definition.  The deprecated 5%-of-PSV floor in ``solve_0d_model.
  indices_from_result`` (RI-plateau artifact; solver README) is deliberately
  NOT reproduced here.

Two properties of the periodic state are used by the calibration and tested
in ``tests/test_pi_filter.py``:

* **DC staging** -- cycle-averaging the ODEs kills every state derivative
  (compliances carry zero mean current, the inertance carries the mean
  flow), so the mean flow obeys Ohm's law on the series resistance::

      Q_mean = (p_in_mean - Pout)/(Rs + R_UV + Rout)      [Rp = inf]

* **Convergence** -- both waveform shape AND cycle-mean levels must be
  periodic.  The slowest circuit time constant (tau_max, reported in
  ``diagnostics``) governs the mean-level transient; a shape-only check
  converges visibly too early, so ``n_cycles_min`` defaults to
  ``ceil(10*tau_max/T)`` (the 1D solver's own margin convention) and the
  periodicity test includes the cycle means.

Implementation notes carried over from planning-time verification:

* The per-cycle warm start advances the cursor with ``sol.sol(t+T)`` from a
  ``dense_output`` solve.  ``solve_0d_model.py`` instead reads
  ``sol.y[:, -1]`` from an endpoint-exclusive ``t_eval``, which advances
  only (1 - 1/n_phase)*T per cycle -- a 0.5%-per-cycle phase-precessing warm
  start (superseded-file bug, documented not patched).
* The cycle-mean inlet pressure is MEASURED (``mean_inlet_pressure``), not
  assumed equal to ``p0``: ``_shape_mean``'s 2000-point rectangle mean
  leaves a +0.37 dyn/cm^2 offset at the matched operating point.
* The system is linear time-invariant, so stiffness is checked exactly from
  the eigenvalues of A before integrating; near-degenerate compliances
  (e.g. Cm ~ 1e-8 cm^5/dyn) make the explicit RK45 step collapse, and
  ``auto_stiff_switch`` moves to Radau instead.
"""

import math
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from gestational_geometry import UV_TRUNK_LENGTH_CM, uv_radius_cm
from solve_arterial_flow_UA import (
    DP_DEFAULT,
    P0_DEFAULT,
    UA_LENGTH_DEFAULT_CM,
    inlet_pressure,
    stiffness,
    waveform_metrics,
)
from solve_venous_flow_UV import simulate_uv_return

MMHG_TO_DYN_CM2 = 1333.22   # dyn/cm^2 per mmHg
ML_MIN_TO_ML_S = 1.0 / 60.0

RHO = 1.05      # blood density, g/cm^3 (matches solve_arterial_flow_UA)
MU = 0.035      # dynamic viscosity, poise

# Lumped conduit resistance R = coefficient*MU*L/(pi*r0^4).  The 1D core's
# momentum friction -22*pi*nu*q/A maps to a pressure drop 22*pi*MU*q/A^2 per
# unit length (the (A/rho)*dp/dx momentum-pressure correspondence), i.e. a
# lumped R with coefficient 22 -- 2.75x exact Poiseuille (coefficient 8,
# parabolic profile, used for the vein and in solve_0d_model's old R_a).
CONDUIT_COEFFICIENT_1D = 22.0
CONDUIT_COEFFICIENT_POISEUILLE = 8.0

# The matched 1D/0D benchmark (outputs/initial_1d_0d_results.md,
# outputs/matched_1d_0d_results.md): the calibration target for
# calibration/fit_pi_filter_matched.py.
MATCHED_1D_CASE = {
    "bpm": 140.0,
    "r0_cm": 0.13,
    "R_total": 3.0e4,
    "C_total": 4.0e-5,
    "p0": 3.0e4,
    "dp_inlet": 1.2e4,
    "L_cm": 72.2,
    "PI": 1.577,
    "RI": 0.840,
    "S_D": 6.255,
    "PSV_cm_s": 11.08,
    "EDV_cm_s": 1.77,
    "total_two_UA_flow_mL_min": 35.29,
}


class PiFilterPhysiologyWarning(UserWarning):
    """Physiologic sanity violation (negative pressure/flow/EDV) -- flagged, not raised."""


class PiFilterStiffnessWarning(UserWarning):
    """Explicit integrator step is outside the LTI stability region."""


def to_mmhg(p_dyn_cm2):
    """Pressure dyn/cm^2 -> mmHg (reporting only)."""
    return np.asarray(p_dyn_cm2, dtype=float) / MMHG_TO_DYN_CM2


def to_ml_min(q_ml_s):
    """Flow mL/s -> mL/min (reporting only)."""
    return np.asarray(q_ml_s, dtype=float) / ML_MIN_TO_ML_S


# ---------------------------------------------------------------------------
# Geometry-derived element values
# ---------------------------------------------------------------------------
def conduit_resistance(r0_cm, length_cm, coefficient=CONDUIT_COEFFICIENT_1D):
    """Lumped series resistance [dyn*s/cm^5] of a uniform conduit.

    ``coefficient=22`` matches the 1D core's Olufsen friction term;
    ``coefficient=8`` is exact Poiseuille (the form used by
    ``solve_venous_flow_UV`` for the vein).
    """
    return coefficient * MU * length_cm / (math.pi * r0_cm ** 4)


def conduit_inertance(r0_cm, length_cm):
    """Lumped inertance [dyn*s^2/cm^5]: rho*L/A (same form as solve_0d_model)."""
    return RHO * length_cm / (math.pi * r0_cm ** 2)


def tube_compliance(r0_cm, length_cm, eh_over_r):
    """Lumped compliance [cm^5/dyn] of a uniform vessel of length L.

    From the 1D core's tube law p(A) = p0 + (4/3)*(Eh/r0)*(1 - sqrt(A0/A)):
    dA/dp = (3/2)*A0/(Eh/r0) per unit length, so C = 3*A0*L/(2*(Eh/r0)).
    """
    area0 = math.pi * r0_cm ** 2
    return 3.0 * area0 * length_cm / (2.0 * eh_over_r)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PiFilterParams:
    """Pi-filter element values for ONE umbilical artery (CGS units).

    Rs      arterial-side series resistance        [dyn*s/cm^5]
    Ls      arterial inertance                     [dyn*s^2/cm^5]
    Cm      placental (microcirculatory) compliance [cm^5/dyn]
    R_UV    villous/venous path resistance         [dyn*s/cm^5]
    Cuv     umbilical-venous-side compliance       [cm^5/dyn]
    Rout    venous outlet resistance to Pout       [dyn*s/cm^5]
    Rp      optional Pm->Pout shunt leak (inf = OFF, handover default)
    Pout    venous outlet pressure [dyn/cm^2] (1D core's Pv default is 0)
    A0      outlet cross-sectional area [cm^2] (rigid; velocity = Q/A0)
    n_arteries  1 = per-artery params, 2 = merged shared-bed params
    """

    Rs: float
    Ls: float
    Cm: float
    R_UV: float
    Cuv: float
    Rout: float
    Rp: float = math.inf
    Pout: float = 0.0
    A0: float = math.pi * 0.13 ** 2
    n_arteries: int = 1
    label: str = "pi_filter"


def default_pi_filter_params(
    ga=26.0,
    r0_cm=MATCHED_1D_CASE["r0_cm"],
    ua_length_cm=UA_LENGTH_DEFAULT_CM,
    conduit_coefficient=CONDUIT_COEFFICIENT_1D,
    R_series_total=None,
    mean_flow_mL_s_per_artery=None,
    p0=P0_DEFAULT,
    Pout=0.0,
    Cm_prior=MATCHED_1D_CASE["C_total"],
    allow_geometry_extrapolation=False,
):
    """Build the prior parameter set for the matched 1D benchmark (or a GA).

    Pinned from geometry/physics (each source documented in the results
    table): Ls and Rs from the UA conduit at r0_cm/ua_length_cm, R_UV from
    the UV at ``ga`` (exact Poiseuille, consistent with
    ``solve_venous_flow_UV``), Cuv from the tube-law compliance of the UV
    (FLAGGED assumption: the arterial stiffness law applied to a vein --
    venous Eh/r0 is not independently measured here).  Cm is an
    order-of-magnitude prior taken from the matched 1D Windkessel
    compliance; the matched-case identifiability analysis shows Cm (and
    Cuv) are NOT identifiable from UA Doppler indices, so this prior pins
    the family rather than being fitted.

    The total series resistance is set by the DC mean-flow constraint
    ``R_series_total = (p_mean - Pout)/Q_mean`` with p_mean = p0 (the
    measured offset is 1.2e-5 relative and handled inside
    ``simulate_pi_filter``); Rout then closes the series sum.
    """
    Ls = conduit_inertance(r0_cm, ua_length_cm)
    Rs = conduit_resistance(r0_cm, ua_length_cm, conduit_coefficient)

    r_uv = float(uv_radius_cm(ga, allow_geometry_extrapolation))
    R_UV = conduit_resistance(r_uv, UV_TRUNK_LENGTH_CM,
                              CONDUIT_COEFFICIENT_POISEUILLE)
    Cuv = tube_compliance(r_uv, UV_TRUNK_LENGTH_CM, stiffness(r_uv))

    if R_series_total is None:
        if mean_flow_mL_s_per_artery is None:
            q_target = (MATCHED_1D_CASE["total_two_UA_flow_mL_min"]
                        * ML_MIN_TO_ML_S / 2.0)
        else:
            q_target = float(mean_flow_mL_s_per_artery)
        R_series_total = (p0 - Pout) / q_target
    Rout = R_series_total - Rs - R_UV
    if Rout <= 0:
        raise ValueError(
            f"R_series_total={R_series_total:.1f} cannot cover Rs={Rs:.1f} "
            f"+ R_UV={R_UV:.1f}: the conduit prior exceeds the DC mean-flow "
            "constraint. Reduce conduit_coefficient or r0_cm, or raise the "
            "flow target.")

    return PiFilterParams(Rs=Rs, Ls=Ls, Cm=float(Cm_prior), R_UV=R_UV,
                          Cuv=Cuv, Rout=Rout, Pout=Pout,
                          A0=math.pi * r0_cm ** 2)


def with_overrides(params, **overrides):
    """Return a copy with validated overrides applied."""
    unknown = set(overrides) - set(asdict(params))
    if unknown:
        raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
    return replace(params, **overrides)


def validate_params(params):
    """Return a list of human-readable parameter problems (empty = OK)."""
    issues = []
    for name in ("Rs", "Ls", "Cm", "R_UV", "Cuv", "Rout", "A0"):
        value = getattr(params, name)
        if not math.isfinite(value) or value <= 0:
            issues.append(f"{name} must be finite and positive (got {value!r})")
    if not math.isfinite(params.Rp) and params.Rp != math.inf:
        issues.append(f"Rp must be finite positive or inf (got {params.Rp!r})")
    elif math.isfinite(params.Rp) and params.Rp <= 0:
        issues.append(f"Rp must be positive (got {params.Rp!r})")
    if params.n_arteries not in (1, 2):
        issues.append(f"n_arteries must be 1 or 2 (got {params.n_arteries})")
    return issues


def to_shared_bed(params, n_arteries=2):
    """Merge n identical per-artery circuits into the exactly-equivalent
    shared-bed circuit.

    Each artery contributes its OWN series (Rs, Ls) branch, so the n
    branches merge in PARALLEL: every series impedance divides by n
    (Rs, Ls, R_UV, Rout, Rp) and every shunt compliance multiplies by n
    (Cm, Cuv, A0).  Dividing the shared-bed ODEs by n recovers the
    per-artery form exactly.
    """
    if n_arteries < 1:
        raise ValueError("n_arteries must be >= 1")
    if params.n_arteries != 1:
        raise ValueError("to_shared_bed expects per-artery params (n_arteries=1)")
    return replace(params,
                   Rs=params.Rs / n_arteries,
                   Ls=params.Ls / n_arteries,
                   R_UV=params.R_UV / n_arteries,
                   Rout=params.Rout / n_arteries,
                   Cm=params.Cm * n_arteries,
                   Cuv=params.Cuv * n_arteries,
                   Rp=params.Rp * n_arteries,
                   A0=params.A0 * n_arteries,
                   n_arteries=int(n_arteries),
                   label=f"{params.label}_shared{n_arteries}")


def from_shared_bed(params, n_arteries=2):
    """Inverse of :func:`to_shared_bed`."""
    if params.n_arteries != n_arteries:
        raise ValueError(
            f"params are n_arteries={params.n_arteries}, expected {n_arteries}")
    return replace(params,
                   Rs=params.Rs * n_arteries,
                   Ls=params.Ls * n_arteries,
                   R_UV=params.R_UV * n_arteries,
                   Rout=params.Rout * n_arteries,
                   Cm=params.Cm / n_arteries,
                   Cuv=params.Cuv / n_arteries,
                   Rp=params.Rp / n_arteries,
                   A0=params.A0 / n_arteries,
                   n_arteries=1,
                   label=params.label.split("_shared")[0])


# ---------------------------------------------------------------------------
# Circuit algebra
# ---------------------------------------------------------------------------
def lti_system_matrices(params):
    """A, B of dy/dt = A y + B u with y=[Q_UA, Pm, P_UV], u=[p_in(t), Pout].

    With Rp = inf the shunt row/column entries vanish (1/inf == 0), so one
    matrix pair covers both the handover circuit and the optional shunt.
    """
    inv_Ls = 1.0 / params.Ls
    inv_Cm = 1.0 / params.Cm
    inv_Cuv = 1.0 / params.Cuv
    inv_Rp = 0.0 if math.isinf(params.Rp) else 1.0 / params.Rp
    A = np.array([
        [-params.Rs * inv_Ls, -inv_Ls, 0.0],
        [inv_Cm, -(1.0 / params.R_UV + inv_Rp) * inv_Cm,
         (1.0 / params.R_UV) * inv_Cm],
        [0.0, (1.0 / params.R_UV) * inv_Cuv,
         -(1.0 / params.R_UV + 1.0 / params.Rout) * inv_Cuv],
    ])
    B = np.array([
        [inv_Ls, 0.0],
        [0.0, inv_Rp * inv_Cm],
        [0.0, (1.0 / params.Rout) * inv_Cuv],
    ])
    return A, B


def dc_solution(params, p_mean, Pout=None):
    """Exact DC (cycle-mean) solution of the linear circuit.

    Mean flow obeys Ohm's law through Rs in series with the parallel
    combination of the shunt (Rp, default off) and the R_UV + Rout branch.
    Cycle-mean storage currents are zero, so this is exact for any periodic
    forcing, not a linearization.
    """
    if Pout is None:
        Pout = params.Pout
    inv_Rp = 0.0 if math.isinf(params.Rp) else 1.0 / params.Rp
    R_par = 1.0 / (inv_Rp + 1.0 / (params.R_UV + params.Rout))
    R_dc = params.Rs + R_par
    q0 = (p_mean - Pout) / R_dc
    pm0 = Pout + q0 * R_par
    q_branch = q0 * R_par / (params.R_UV + params.Rout)
    puv0 = Pout + q_branch * params.Rout
    return {
        "R_dc": R_dc,
        "R_par": R_par,
        "Q0_mL_s": q0,
        "Pm0": pm0,
        "P_UV0": puv0,
        # per-circuit flow; interpret via params.n_arteries
        "mean_flow_mL_min_per_artery": q0 * 60.0 / params.n_arteries,
        "total_two_UA_flow_mL_min": q0 * 60.0 * 2.0 / params.n_arteries,
        "p_mean_used": p_mean,
        "Pout": Pout,
    }


def mean_inlet_pressure(T, p0, dp_inlet, systolic_frac=0.35, n=200_000):
    """Measured cycle-mean inlet pressure.

    The forcing normalizes its shape by a 2000-point rectangle mean
    (``_shape_mean``), which leaves a small offset from ``p0`` (+0.37
    dyn/cm^2 at the matched operating point).  DC staging and DC
    initialization use this measured value instead of assuming p0.
    """
    t = np.linspace(0.0, T, n, endpoint=False)
    return float(np.mean(inlet_pressure(t, T, p0, dp_inlet, systolic_frac)))


def pi_filter_rhs(t, y, params, p0, dp_inlet, T, systolic_frac=0.35):
    """Right-hand side of the handover's three ODEs (plus optional shunt)."""
    q, pm, puv = y
    p_in = inlet_pressure(t, T, p0, dp_inlet, systolic_frac)
    q_uv = (pm - puv) / params.R_UV
    dq = (p_in - params.Rs * q - pm) / params.Ls
    dpm = (q - q_uv) / params.Cm
    if math.isfinite(params.Rp):
        dpm -= (pm - params.Pout) / (params.Rp * params.Cm)
    dpuv = (q_uv - (puv - params.Pout) / params.Rout) / params.Cuv
    return [dq, dpm, dpuv]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_pi_filter(
    bpm,
    params=None,
    *,
    p0=P0_DEFAULT,
    dp_inlet=DP_DEFAULT,
    systolic_frac=0.35,
    n_cycles_min=None,
    n_cycles_max=400,
    tol=1e-4,
    n_phase=200,
    method="RK45",
    max_step_per_cycle=200,
    init="dc",
    diastolic_definition="minimum",
    per_artery=True,
    strict=False,
    auto_stiff_switch=True,
):
    """Integrate the pi-filter to a periodic state; return one full cycle.

    Periodicity requires BOTH the phase-aligned waveform (RMS relative to
    amplitude) and the cycle-mean levels of Q_UA, Pm, P_UV to repeat within
    ``tol`` -- the mean-level guard matters because the slowest circuit
    time constant can be several cardiac cycles (see module docstring).
    """
    if params is None:
        params = default_pi_filter_params()
    issues = validate_params(params)
    if issues:
        raise ValueError("invalid pi-filter parameters: " + "; ".join(issues))
    if per_artery and params.n_arteries == 2:
        params = from_shared_bed(params, params.n_arteries)

    T = 60.0 / float(bpm)
    max_step = T / float(max_step_per_cycle)
    p_mean = mean_inlet_pressure(T, p0, dp_inlet, systolic_frac)

    A_mat, _B = lti_system_matrices(params)
    eigs = np.linalg.eigvals(A_mat)
    stable_lti = bool(np.all(eigs.real < 0.0))
    eig_mag = np.abs(eigs)
    stiffness_ratio = float(eig_mag.max() / eig_mag.min()) if eig_mag.min() > 0 else math.inf
    stiff = bool(eig_mag.max() * max_step > 1.0)
    # slowest decay = eigenvalue real part closest to zero (all are negative
    # when stable); tau_max governs the mean-level transient (F3)
    tau_max = float(-1.0 / eigs.real.max()) if stable_lti else math.nan

    method_used = method
    if stiff and method in ("RK45", "RK23", "DOP853"):
        message = (
            f"pi-filter is stiff for the explicit step (|lambda|max*max_step = "
            f"{eig_mag.max() * max_step:.1f} > 1); smallest compliance "
            "timescale is under-resolved")
        warnings.warn(message, PiFilterStiffnessWarning, stacklevel=2)
        if auto_stiff_switch:
            method_used = "Radau"

    if n_cycles_min is None:
        if stable_lti:
            n_cycles_min = max(10, int(math.ceil(10.0 * tau_max / T)))
        else:
            n_cycles_min = 10
    if init == "newton_periodic":
        # the Newton state is already periodic to ~1e-10; checking early is
        # then a meaningful (and cheap) confirmation rather than a guard
        n_cycles_min = min(n_cycles_min, 2)

    dc = dc_solution(params, p_mean)
    if init == "dc":
        y = np.array([dc["Q0_mL_s"], dc["Pm0"], dc["P_UV0"]])
    elif init == "quiescent":
        y = np.zeros(3)
    elif init == "newton_periodic":
        y = _newton_periodic_state(params, p0, dp_inlet, T, systolic_frac,
                                   method_used, max_step, dc)
    else:
        raise ValueError("init must be 'dc', 'newton_periodic', or 'quiescent'")

    def rhs(t, y_):
        return pi_filter_rhs(t, y_, params, p0, dp_inlet, T, systolic_frac)

    phase_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    t_cursor = 0.0
    prev = None
    converged = False
    n_cycles_run = 0
    rms_rel_last = math.nan
    for _cycle in range(int(n_cycles_max)):
        y_cycle_start = y
        t_eval = t_cursor + phase_grid * T
        sol = solve_ivp(rhs, (t_cursor, t_cursor + T), y, method=method_used,
                        dense_output=True, t_eval=t_eval,
                        rtol=1e-8, atol=1e-10, max_step=max_step)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise ValueError(
                f"pi-filter integration diverged (cycle {n_cycles_run + 1}, "
                f"method={method_used}); parameters are likely outside the "
                "stable/physical region")
        q_cycle = sol.y[0]
        pm_cycle = sol.y[1]
        puv_cycle = sol.y[2]
        y = sol.sol(t_cursor + T)   # exact end-of-cycle state (F2 fix)
        t_cursor += T
        n_cycles_run += 1

        if prev is not None and n_cycles_run >= n_cycles_min:
            amp = max(q_cycle.max() - q_cycle.min(), 1e-12)
            rms_rel = float(np.sqrt(np.mean((q_cycle - prev["q"]) ** 2)) / amp)
            dq_mean = abs(np.mean(q_cycle) - prev["q_mean"]) / max(abs(np.mean(q_cycle)), 1e-12)
            dpm_mean = abs(np.mean(pm_cycle) - prev["pm_mean"]) / p0
            dpuv_mean = abs(np.mean(puv_cycle) - prev["puv_mean"]) / p0
            rms_rel_last = max(rms_rel, dq_mean, dpm_mean, dpuv_mean)
            if rms_rel_last < tol:
                converged = True
                prev = {"q": q_cycle, "pm": pm_cycle, "puv": puv_cycle,
                        "q_mean": np.mean(q_cycle), "pm_mean": np.mean(pm_cycle),
                        "puv_mean": np.mean(puv_cycle)}
                break
        prev = {"q": q_cycle, "pm": pm_cycle, "puv": puv_cycle,
                "q_mean": np.mean(q_cycle), "pm_mean": np.mean(pm_cycle),
                "puv_mean": np.mean(puv_cycle)}

    # ---- waveforms of the final (converged) cycle, wrap-inclusive ----------
    q = prev["q"]
    pm = prev["pm"]
    puv = prev["puv"]
    t = np.concatenate([phase_grid * T, [T]])
    q_m = np.concatenate([q, [q[0]]])
    q_mean = float(np.mean(q))
    area_m = np.full_like(t, params.A0)
    metrics = waveform_metrics(t, q_m, area_m, diastolic_definition)
    if params.n_arteries == 2:
        # waveform_metrics assumes per-artery flow (total = 2*q_mean); in the
        # merged shared-bed circuit Q_UA already IS the total flow
        metrics = dict(metrics)
        metrics["mean_flow_mL_s_per_artery"] = q_mean / params.n_arteries
        metrics["total_two_UA_flow_mL_min"] = q_mean * 60.0

    q_uv = (pm - puv) / params.R_UV
    q_out = (puv - params.Pout) / params.Rout
    q_uv_mean = float(np.mean(q_uv))
    q_out_mean = float(np.mean(q_out))
    imbalance = max(abs(q_mean - q_uv_mean), abs(q_uv_mean - q_out_mean))
    dpm_dt = (q - q_uv) / params.Cm
    if math.isfinite(params.Rp):
        dpm_dt = dpm_dt - (pm - params.Pout) / params.Rp
    dpuv_dt = (q_uv - q_out) / params.Cuv
    # exact cycle-mean storage currents: integrating Cm dPm/dt over the
    # cycle gives Cm*(Pm(end)-Pm(start))/T from the true cycle-end states --
    # zero in the periodic state, so these measure the periodicity residual
    # directly (unlike averaging the sampled derivative series, which is
    # dominated by endpoint bias)
    mean_storage_cm = params.Cm * (y[1] - y_cycle_start[1]) / T
    mean_storage_cuv = params.Cuv * (y[2] - y_cycle_start[2]) / T

    flags = {
        "positive_pressure": bool(min(pm.min(), puv.min()) > 0.0),
        "positive_mean_flow": bool(q_mean > 0.0),
        "positive_edv": bool(metrics["EDV_cm_s"] > 0.0),
        "periodic": bool(converged),
        "stable_lti": stable_lti,
        "stiff": stiff,
        "converged_within_budget": bool(converged),
        "method_used": method_used,
    }
    physiology_ok = all(flags[k] for k in
                        ("positive_pressure", "positive_mean_flow",
                         "positive_edv", "stable_lti"))
    if not physiology_ok:
        message = ("pi-filter state violates physiologic sanity: "
                   + ", ".join(k for k in ("positive_pressure",
                                           "positive_mean_flow",
                                           "positive_edv", "stable_lti")
                               if not flags[k]))
        if strict:
            raise ValueError(message)
        warnings.warn(message, PiFilterPhysiologyWarning, stacklevel=2)

    result = {
        "t": t,
        "p_in": inlet_pressure(t, T, p0, dp_inlet, systolic_frac),
        "Q_UA": q_m,
        "Pm": np.concatenate([pm, [pm[0]]]),
        "P_UV": np.concatenate([puv, [puv[0]]]),
        "Q_UV": np.concatenate([q_uv, [q_uv[0]]]),
        "Q_out": np.concatenate([q_out, [q_out[0]]]),
        "dPm_dt": np.concatenate([dpm_dt, [dpm_dt[0]]]),
        "dP_UV_dt": np.concatenate([dpuv_dt, [dpuv_dt[0]]]),
        "q_outlet": q_m,
        "u_outlet": q_m / params.A0,
        "A0": params.A0,
        "metrics": metrics,
        "T": T,
        "converged": bool(converged),
        "n_cycles_run": n_cycles_run,
        "rms_rel_last": rms_rel_last,
        "tol": tol,
        "mean_Q_UA_mL_s": q_mean,
        "mean_inlet_pressure": p_mean,
        "flow_conservation": {
            "Q_UA_mean": q_mean,
            "Q_UV_mean": q_uv_mean,
            "Q_out_mean": q_out_mean,
            "max_abs_imbalance_mL_s": float(imbalance),
            "max_abs_imbalance_rel": float(imbalance / max(abs(q_mean), 1e-12)),
            "mean_storage_Cm_dPm_dt": float(mean_storage_cm),
            "mean_storage_Cuv_dP_UV_dt": float(mean_storage_cuv),
        },
        "dc": dc,
        "params": asdict(params),
        "flags": flags,
        "diagnostics": {
            "lambda_eigs": eigs,
            "stiffness_ratio": stiffness_ratio,
            "tau_max_s": tau_max,
            "inertial_timescale_s": params.Ls / params.Rs,
            "placental_RC_timescale_s": params.Cm * (params.R_UV + params.Rout),
            "venous_RC_timescale_s": params.Cuv * params.Rout,
            "tav_periodic_mean": q_mean / params.A0,
            "tav_trapezoid_relative_diff":
                (metrics["TAV_cm_s"] - q_mean / params.A0)
                / max(abs(q_mean / params.A0), 1e-12),
        },
    }
    return result


def _newton_periodic_state(params, p0, dp_inlet, T, systolic_frac, method,
                           max_step, dc, max_iter=6, tol=1e-10):
    """Newton iteration on the one-cycle stroboscopic map Phi: y -> y(T|y).

    Converges to the exact periodic orbit in ~4 iterations (16 one-cycle
    integrations), removing the startup transient entirely -- faster than
    warm-starting through ~10*tau/T cycles and immune to slow mean-level
    transients.  Starts from the DC state.
    """
    def rhs(t, y_):
        return pi_filter_rhs(t, y_, params, p0, dp_inlet, T, systolic_frac)

    def strobe(y_):
        sol = solve_ivp(rhs, (0.0, T), y_, method=method, dense_output=True,
                        rtol=1e-8, atol=1e-10, max_step=max_step)
        return sol.sol(T)

    y = np.array([dc["Q0_mL_s"], dc["Pm0"], dc["P_UV0"]])
    for _ in range(max_iter):
        f = strobe(y) - y
        if np.linalg.norm(f) <= tol * max(np.linalg.norm(y), 1e-12):
            break
        jac = np.empty((3, 3))
        for i in range(3):
            h = 1e-6 * max(abs(y[i]), 1e-12)
            yp = y.copy()
            yp[i] += h
            jac[:, i] = (strobe(yp) - strobe(y)) / h
        delta = np.linalg.solve(np.eye(3) - jac, f)
        y = y + delta
    return y


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def pressures_mmhg(result):
    """Pressure waveforms of a result dict, converted to mmHg."""
    return {key: to_mmhg(result[key]) for key in ("p_in", "Pm", "P_UV")
            if key in result}


def uv_return_check(params, ga, allow_geometry_extrapolation=False):
    """Cross-check params.R_UV against the reduced 1D UV model's resistance.

    Also reports how small R_UV is relative to the total series resistance:
    the handover's ``R_UV`` is dominated by the fetoplacental/villous path,
    not by the cord vein itself (the cord-vein Poiseuille resistance alone
    is well under 1% of the total), which is why the venous drop is nearly
    invisible in the pressure traces.
    """
    r_uv = float(uv_radius_cm(ga, allow_geometry_extrapolation))
    reference = conduit_resistance(r_uv, UV_TRUNK_LENGTH_CM,
                                   CONDUIT_COEFFICIENT_POISEUILLE)
    t = np.linspace(0.0, 1.0, 101)
    uv_model = simulate_uv_return(t, np.full_like(t, 0.294), ga,
                                  allow_geometry_extrapolation=allow_geometry_extrapolation)
    total = params.Rs + params.R_UV + params.Rout
    return {
        "R_UV_params": params.R_UV,
        "R_UV_reference_formula": reference,
        "R_UV_uv_model": uv_model["resistance_dyn_s_cm5"],
        "R_UV_relative_error":
            abs(params.R_UV - reference) / reference,
        "R_UV_fraction_of_total": params.R_UV / total,
        "uv_radius_cm": r_uv,
        "uv_length_cm": UV_TRUNK_LENGTH_CM,
    }


def couple_uv_return(result, params, ga, radius_scale=1.0,
                     allow_geometry_extrapolation=False):
    """Drive the reduced 1D UV model with the pi-filter's placental efflux.

    One-way, quasi-steady coupling: the pi-filter determines the efflux
    waveform ``Q_out(t)`` through Rout -- the flow that has actually left
    the placental compliance, i.e. the storage-consistent venous inflow
    (the pi-filter analogue of the 1D solver's efflux ``2*(Pwk-Pv)/R2``
    versus the arterial inflow ``q_b``; see the solver README's UV
    bookkeeping note).  ``simulate_uv_return`` then computes velocity and
    pressure-drop waveforms in the cord vein given that inflow.  The UV
    does not feed back into the circuit: its Poiseuille resistance is a
    fraction of a percent of the total series resistance and its drop is
    already inside the Rout lump, so feeding back would double-count it.

    ``result`` must be a ``simulate_pi_filter`` output for ONE cardiac
    cycle; per-artery results (``n_arteries == 1``) are doubled, shared-bed
    results (``n_arteries == 2``) already carry the total flow.  ``params``
    must be the parameter set the result was computed with (an n_arteries
    mismatch would silently mis-scale the doubling).
    """
    if int(result["params"]["n_arteries"]) != int(params.n_arteries):
        raise ValueError(
            "result and params disagree on n_arteries "
            f"(result: {result['params']['n_arteries']}, "
            f"params: {params.n_arteries}); pass the params the result "
            "was computed with")
    total = result["Q_out"] * (2.0 / params.n_arteries)
    uv = simulate_uv_return(result["t"], total, ga, radius_scale=radius_scale,
                            allow_geometry_extrapolation=allow_geometry_extrapolation)
    # means over the EXCLUSIVE samples: the wrap-inclusive waveform repeats
    # the first sample at the end, which would bias np.mean
    mean_efflux = float(np.mean(total[:-1]))
    mean_inflow = (float(np.mean(result["Q_UA"][:-1]))
                   * (2.0 / params.n_arteries))
    return {
        "uv": uv,
        "efflux_total_mL_s": total,
        "mean_efflux_total_mL_min": mean_efflux * 60.0,
        "mean_arterial_inflow_total_mL_min": mean_inflow * 60.0,
        # in the periodic state the compliance stores no net volume, so the
        # cycle-mean efflux equals the cycle-mean inflow up to the
        # convergence residual
        "mean_efflux_minus_inflow_mL_min": (mean_efflux - mean_inflow) * 60.0,
        "ga": ga,
        "n_arteries": params.n_arteries,
        "radius_scale": radius_scale,
    }
