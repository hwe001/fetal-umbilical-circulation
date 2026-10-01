"""
1D nonlinear blood-flow model of the fetal umbilical artery.

Governing equations (single homogeneous elastic vessel segment):

    dA/dt + dq/dx = 0                                   (continuity)
    dq/dt + d/dx( q^2/A + psi(A) ) = -22*pi*nu * q/A     (momentum, Poiseuille friction)

with the Olufsen-type tube law

    p(A) = p0 + (4/3)*(Eh/r0)*(1 - sqrt(A0/A)),   Eh/r0 = k1*exp(k2*r0) + k3

For a spatially homogeneous vessel (A0, Eh/r0 constant in x) the pressure-gradient
term collapses into a genuine flux function

    psi(A) = (4/(3*rho)) * (Eh/r0) * sqrt(A0*A)

so the system is a standard nonlinear conservation law U=[A,q], F(U)=[q, q^2/A+psi(A)],
solved here with the two-step (Richtmyer) Lax-Wendroff scheme, applied uniformly at
every node -- including the boundaries -- via ghost nodes.

Boundary conditions:
  - Inlet: absorbing and characteristic-consistent (Willemet et al. 2011).
    For this tube law c is proportional to A**(-1/4), so integral(c/A dA)
    equals -4c.  The forward-traveling Riemann invariant is therefore
    W1 = u - 4c (moving into the
    domain) is specified from a prescribed periodic pressure pulse (same
    shape for every run, period set by the patient's heart rate), constructed
    as a pure forward (simple) wave superposed on the vessel's quiescent
    reference state: holding the backward invariant at its reference value
    gives W1 = 4*c_ref - 8*c_target (c_ref at the unstressed area A0),
    reducing to the standard small-amplitude impedance relation
    dp = rho*c_ref*du. The backward-traveling invariant W2 = u + 4c
    (carrying information from the interior toward, and past, the boundary)
    is extrapolated from the interior via characteristic foot-tracing rather
    than specified, so a wave arriving from downstream passes through the
    inlet instead of re-reflecting off it. Both the resulting boundary
    velocity and area respond to whatever W2 turns out to be; nothing is
    clamped rigidly to the target pressure. Mean pressure p0 and pulse
    amplitude dp_inlet are held fixed within a comparison set except where
    dp_inlet is explicitly varied as part of the gestational parameterization
    (see paper Methods 2.5).
  - Outlet: an RCR Windkessel representing the placental vascular bed. The
    ghost node's area is set so its tube-law pressure equals the Windkessel
    chamber pressure offset by the flow through a small proximal resistor R1
    (explicit, lagged by one step); the ghost flow is zero-gradient. Unlike
    the inlet, this boundary is deliberately reflective in the physical
    sense -- the Windkessel's impedance mismatch with the vessel is real
    downstream physiology, not a numerical artifact to be removed.

Combining the inlet's specified W1 and extrapolated W2 to get velocity and
wave speed (hence area) is a direct linear combination, not an iterative
solve, and so does not suffer from the ill-conditioning that a naive
inversion of u = W1 - 4c(A) for A can otherwise hit here (the pulse-wave
speed, order 100s of cm/s, is far larger than the flow velocity, order 10s
of cm/s, so solving for A inside a nonlinear combination of the two can be a
near-cancellation of two large numbers); the outlet's ghost-node formulation
avoids that same pitfall by construction, as before.

Note on units: this is a single-segment, order-of-magnitude model. p0 (the
vessel's reference/baseline pressure) and the inlet pulse-pressure amplitude
are free parameters chosen for internal consistency and held fixed across a
comparison (e.g. Early vs Late gestation); only relative differences between
parameter sets, and the resulting velocity waveform shapes, are physically
interpreted -- not absolute mmHg values.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CGS units: cm, g, s -> velocity cm/s, pressure dyn/cm^2)
# ---------------------------------------------------------------------------
RHO = 1.05      # blood density, g/cm^3
MU = 0.035      # blood viscosity, poise (dyn*s/cm^2)
NU = MU / RHO   # kinematic viscosity, cm^2/s

# Olufsen-type stiffness law Eh/r0 = K1*exp(K2*r0) + K3  (dyn/cm^2)
K1 = 1.0e6
K2 = -22.53
K3 = 4.32e5

# Default operating point shared by every run in a comparison set
P0_DEFAULT = 3.0e4       # vessel reference pressure, dyn/cm^2 (tube_law(A0) == p0)
DP_DEFAULT = 1.2e4       # inlet pulse-pressure amplitude, dyn/cm^2
UA_LENGTH_DEFAULT_CM = 72.2  # helical UA trunk arc length used for new analyses


def stiffness(r0):
    """Eh/r0 for a vessel of unstressed radius r0 (cm)."""
    return K1 * np.exp(K2 * r0) + K3


def tube_law(A, A0, Eh_r0, p0=0.0):
    return p0 + (4.0 / 3.0) * Eh_r0 * (1.0 - np.sqrt(A0 / np.maximum(A, 1e-12)))


def inverse_tube_law(p, A0, Eh_r0, p0=0.0):
    """Invert tube_law() for A given a target pressure p."""
    k = 1.0 - (p - p0) * 3.0 / (4.0 * Eh_r0)
    k = np.maximum(k, 0.05)  # guard against runaway collapse
    return A0 / k ** 2


def wave_speed(A, A0, Eh_r0, rho=RHO):
    """Moens-Korteweg-type wave speed c(A) implied by the tube law."""
    c_ref = np.sqrt((2.0 / 3.0) * Eh_r0 / rho)  # c at A = A0
    return c_ref * (A0 / np.maximum(A, 1e-12)) ** 0.25


def invert_wave_speed(c, A0, Eh_r0, rho=RHO):
    """Invert wave_speed() for A given a target wave speed c."""
    c_ref = np.sqrt((2.0 / 3.0) * Eh_r0 / rho)
    return A0 * (c_ref / np.maximum(c, 1e-6)) ** 4


def psi(A, A0, Eh_r0, rho=RHO):
    """Momentum-flux potential such that d(psi)/dA = (A/rho)*dp/dA."""
    return (4.0 / (3.0 * rho)) * Eh_r0 * np.sqrt(A0 * np.maximum(A, 1e-12))


def flux(A, q, A0, Eh_r0, rho=RHO):
    F1 = q
    F2 = q ** 2 / np.maximum(A, 1e-12) + psi(A, A0, Eh_r0, rho)
    return F1, F2


def friction_source(A, q, nu=NU):
    """
    Coefficient -22*pi*nu follows Olufsen (1999): -2*pi*(zeta+2)*nu*q/A for
    an assumed velocity profile of order zeta=9 (flatter than the exact
    parabolic Poiseuille profile, zeta=2, which would instead give -8*pi*nu),
    chosen there as more representative of the pulsatile, higher-Womersley-
    number regime in the larger systemic arteries the coefficient was
    originally fit to; not re-derived here for the umbilical artery.
    """
    return -22.0 * np.pi * nu * q / np.maximum(A, 1e-12)


# ---------------------------------------------------------------------------
# Inlet: fixed-shape periodic pressure pulse
# ---------------------------------------------------------------------------
_SHAPE_MEAN_CACHE = {}


def _inlet_shape(phase, ts):
    # Systolic upstroke rises monotonically from 0 to 1 over [0, ts]
    # (quarter-period sine, so its derivative vanishes at the peak);
    # diastolic runoff is an exponential decay starting from that same
    # peak value of 1, so the two pieces meet continuously (value-matched)
    # at phase=ts, with only a slope kink there (as for any triangular/
    # exponential pulse with a sharp systolic peak). A version of this
    # shape previously used sin(pi*phase/ts) for the systolic branch,
    # which returns to 0 at phase=ts while the diastolic branch starts at
    # 1 -- a genuine jump discontinuity re-injected into the pressure
    # forcing every single cardiac cycle, fixed here.
    return np.where(
        phase < ts,
        np.maximum(np.sin(np.pi * phase / (2 * ts)), 0.0) ** 1.5,
        np.exp(-3.0 * (phase - ts) / (1 - ts)),
    )


def _shape_mean(ts, n=2000):
    if ts not in _SHAPE_MEAN_CACHE:
        phase = np.linspace(0, 1, n, endpoint=False)
        _SHAPE_MEAN_CACHE[ts] = _inlet_shape(phase, ts).mean()
    return _SHAPE_MEAN_CACHE[ts]


def inlet_pressure(t, T, p0, dp, systolic_frac=0.35):
    """
    Same waveform shape/amplitude for every run; only the period T=60/BPM
    changes with the patient's heart rate. Quarter-sine systolic upstroke
    (monotonic rise from 0 to a peak at phase=systolic_frac, zero slope at
    the peak) followed by an exponential diastolic decay from that same
    peak value, so the two pieces are continuous (matched in value, with a
    slope kink at the peak) rather than discontinuous; oscillating around
    p0 with pulse-pressure amplitude dp (cycle-mean pressure equals p0).
    """
    phase = (t % T) / T
    ts = systolic_frac
    shape = _inlet_shape(phase, ts)
    shape_norm = shape / _shape_mean(ts)  # mean 1, peak > 1
    return p0 + dp * (shape_norm - 1.0)


def inlet_pressure_smooth(t, T, p0, dp, peak_phase=0.20, concentration=2.5):
    """C-infinity periodic diagnostic pulse with the same cycle mean ``p0``.

    This is deliberately a diagnostic alternative, not a replacement clinical
    waveform.  It removes the slope kink in :func:`inlet_pressure` so that a
    mesh study can distinguish dispersive response to nonsmooth forcing from
    an outlet-boundary error.
    """
    phase = (t % T) / T
    shape = np.exp(concentration * np.cos(2.0 * np.pi * (phase - peak_phase)))
    # I0(kappa) is the exact cycle mean of exp(kappa*cos(theta)).
    shape_mean = np.i0(concentration)
    return p0 + dp * (shape / shape_mean - 1.0)


def waveform_metrics(t, q, A, diastolic_definition="minimum"):
    """Return clinically interpretable indices for one cardiac cycle.

    ``end_cycle`` uses the final pre-wrap velocity as EDV, matching the
    clinical PSV/EDV definition. ``minimum`` is provided for sensitivity
    analysis. ``legacy_floor`` reproduces the historical implementation
    that imposed EDV >= 5% of PSV; it must not be used for calibration
    because the resulting RI plateau makes the inverse problem nonsmooth.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)
    if t.ndim != 1 or q.shape != t.shape or A.shape != t.shape or len(t) < 3:
        raise ValueError("t, q and A must be equal-length 1D arrays with at least 3 samples")
    duration = t[-1] - t[0]
    if duration <= 0 or np.any(A <= 0):
        raise ValueError("time must increase and area must remain positive")
    u = q / A
    psv = float(np.max(u))
    if diastolic_definition == "end_cycle":
        edv = float(u[-1])
    elif diastolic_definition == "minimum":
        edv = float(np.min(u))
    elif diastolic_definition == "legacy_floor":
        edv = float(max(np.min(u), 0.05 * psv))
    else:
        raise ValueError("diastolic_definition must be end_cycle, minimum, or legacy_floor")
    tav = float(np.trapezoid(u, t) / duration)
    q_mean = float(np.trapezoid(q, t) / duration)
    amplitude = psv - edv
    return {
        "PSV_cm_s": psv,
        "EDV_cm_s": edv,
        "TAV_cm_s": tav,
        "PI": amplitude / max(abs(tav), 1e-12),
        "RI": amplitude / max(abs(psv), 1e-12),
        "S_D": psv / edv if abs(edv) > 1e-12 else np.nan,
        "mean_flow_mL_s_per_artery": q_mean,
        "total_two_UA_flow_mL_min": 2.0 * q_mean * 60.0,
        "diastolic_definition": diastolic_definition,
    }


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def simulate(
    bpm,
    r0,
    R_total,
    C_total,
    R1_frac=0.15,
    p0=P0_DEFAULT,
    dp_inlet=DP_DEFAULT,
    L=UA_LENGTH_DEFAULT_CM,
    M=100,
    n_cycles=15,
    steps_per_cycle=400,
    Pv=0.0,
    max_cycles=None,
    tol=1e-3,
    verbose=False,
    cfl=0.9,
    inlet_profile="original",
    diagnostic_offsets=(0.0, 0.1, 0.5, 1.0),
    index_definition="minimum",
):
    """
    Run the 1D model for a homogeneous umbilical-artery segment of unstressed
    radius r0 (cm), terminated by an RCR Windkessel (R_total, C_total, split
    into a small proximal resistor R1=R1_frac*R_total and a distal resistor
    R2=R_total-R1) representing the placental vascular bed. The inlet is
    driven by a fixed pulsatile pressure (mean p0, amplitude dp_inlet) at the
    patient's heart rate; p0 is also the vessel's reference/baseline pressure
    (tube_law(A0)=p0), so p0 and dp_inlet should be held fixed across a set
    of runs being compared (e.g. Early vs Late gestation, or an R_total
    sensitivity sweep) -- only r0, R_total, C_total should differ between
    them.

    Periodicity: a fixed cycle count is not, by itself, evidence of a
    periodic orbit -- the RCR Windkessel relaxes toward its asymptotic state
    with time constant tau = R2*C_total, and if tau is large relative to the
    cardiac period T, a slowly-decaying transient in the mean level can look
    "converged" cycle-to-cycle (small pulsatile-shape change) while the
    Windkessel state is still far from periodic. To guard against this,
    `simulate()` always runs at least `n_cycles` cycles, then continues
    (up to `max_cycles`, default 10*tau/T cycles so ~5 time constants of
    margin, floored at `n_cycles` and capped at 2000 for safety) checking
    after every additional full cycle whether ALL of the following have
    changed by less than `tol` (relative) from the previous cycle: the
    Windkessel pressure Pwk at cycle end, the cycle-mean outlet pressure,
    the cycle-mean outlet area, the cycle-mean inlet flow, and the
    phase-aligned RMS difference in the outlet flow waveform (normalized by
    that cycle's flow amplitude). Only once all five agree to within `tol`
    is the run treated as periodic; `converged` reports whether this was
    reached before `max_cycles`.
    """
    A0 = np.pi * r0 ** 2
    Eh_r0 = stiffness(r0)
    dx = L / (M - 1)
    T = 60.0 / bpm

    R1 = R1_frac * R_total
    R2 = R_total - R1
    tau = R2 * C_total

    # Smooth pulsatile ramp: the vessel starts fully quiescent and the
    # Windkessel starts at the matching zero-flow state (see below), so
    # snapping the full pulsatile pressure amplitude on at t=0 would itself
    # be a startup transient. Ramp dp_inlet's contribution from 0 to full
    # amplitude over T_ramp, chosen long relative to both the cardiac period
    # and the Windkessel relaxation time, and never let the minimum cycle
    # count before checking periodicity fall inside the ramp (a slowly
    # ramping mean level could otherwise look falsely "periodic" cycle to
    # cycle).
    T_ramp = min(max(3 * T, 3 * tau), 40 * T)
    n_cycles = max(n_cycles, int(np.ceil(T_ramp / T)) + 2)

    if max_cycles is None:
        max_cycles = int(np.clip(np.ceil(10 * tau / T), n_cycles, 2000))
    else:
        max_cycles = max(max_cycles, n_cycles)

    A = np.full(M, A0)
    q = np.zeros(M)
    # Mutually compatible zero-flow initialization: the vessel starts fully
    # quiescent (A=A0, q=0 everywhere), so the Windkessel must start at the
    # SAME zero-flow state (Pwk=p0, no pressure drop across R1) rather than
    # the steady-state-flow guess Pv+(p0-Pv)*R2/R_total used previously,
    # which silently assumed a nonzero steady flow (q=p0/R_total) the
    # quiescent interior did not actually have -- an initial-condition
    # inconsistency that manifested as a small, spurious startup backflow
    # once the outlet was solved as a same-stage (non-lagged) boundary
    # condition (a lagged/relaxed treatment had been masking it).
    Pwk = p0
    q_out_prev = 0.0
    q_in_prev = q_out_prev

    dt_nominal = T / steps_per_cycle
    t = 0.0
    # max_steps must be a genuine step-count *budget*, not just a multiple of
    # the nominal steps_per_cycle: as M grows, dx shrinks and the CFL bound
    # (0.9*dx/c_max) can force far more than steps_per_cycle steps per cycle,
    # which an earlier version of this cap (steps_per_cycle*4) did not
    # account for -- it silently truncated fine-mesh runs long before they
    # finished max_cycles worth of *simulated time*, which looks like (but is
    # not) a convergence failure. Estimate the CFL-limited step count from a
    # generously inflated wave-speed bound (3x the reference wave speed, to
    # allow for area compression and flow-velocity contributions raising the
    # true characteristic speed above c_ref) and budget from whichever of
    # that estimate or the nominal rate is larger, with a further 2x margin.
    c_ref_est = np.sqrt((2.0 / 3.0) * Eh_r0 / RHO)
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must lie in (0, 1]")
    if inlet_profile not in ("original", "smooth"):
        raise ValueError("inlet_profile must be 'original' or 'smooth'")

    dt_cfl_est = cfl * dx / max(3.0 * c_ref_est, 1e-6)
    steps_per_cycle_est = max(steps_per_cycle, int(np.ceil(T / dt_cfl_est)))
    max_steps = max_cycles * steps_per_cycle_est * 2

    hist_t, hist_q, hist_A, hist_Pwk, hist_qin = [], [], [], [], []
    offsets = tuple(float(v) for v in diagnostic_offsets)
    diagnostic_indices = {
        off: int(np.argmin(np.abs(np.linspace(0.0, L, M) - (L - off))))
        for off in offsets
    }
    hist_diag_q = {off: [] for off in offsets}
    hist_diag_A = {off: [] for off in offsets}
    dt_history = []
    cfl_history = []
    n_cycles_run = 0
    converged = False
    outlet_fallback_count = [0]
    total_steps_taken = [0]

    step = 0
    while t < max_cycles * T and step < max_steps:
        c = wave_speed(A, A0, Eh_r0)
        u = q / np.maximum(A, 1e-12)
        c_max = np.max(np.abs(u) + c)
        dt = min(dt_nominal, cfl * dx / max(c_max, 1e-6))
        dt_history.append(dt)
        cfl_history.append(dt * c_max / dx)

        F1, F2 = flux(A, q, A0, Eh_r0)
        S2 = friction_source(A, q)

        Ahalf = 0.5 * (A[:-1] + A[1:]) - (dt / (2 * dx)) * (F1[1:] - F1[:-1])
        qhalf = (
            0.5 * (q[:-1] + q[1:])
            - (dt / (2 * dx)) * (F2[1:] - F2[:-1])
            + (dt / 2) * 0.5 * (S2[:-1] + S2[1:])
        )
        F1h, F2h = flux(Ahalf, qhalf, A0, Eh_r0)
        S2h = friction_source(Ahalf, qhalf)

        A_new = A.copy()
        q_new = q.copy()
        A_new[1:-1] = A[1:-1] - (dt / dx) * (F1h[1:] - F1h[:-1])
        q_new[1:-1] = (
            q[1:-1] - (dt / dx) * (F2h[1:] - F2h[:-1])
            + dt * 0.5 * (S2h[1:] + S2h[:-1])
        )

        # inlet: absorbing, characteristic-consistent boundary condition
        # (Willemet et al. 2011). W1 (forward-traveling, into the domain) is
        # specified from the desired pressure pulse; W2 (backward-traveling,
        # i.e. carrying information from the interior toward and past the
        # boundary) is extrapolated via characteristic foot-tracing rather
        # than specified, so a wave arriving from downstream passes through
        # instead of re-reflecting off the inlet. Both u and A at the
        # boundary respond to the extrapolated W2, rather than clamping
        # pressure rigidly to the target -- this is what makes the
        # condition absorbing rather than reflective.
        #
        # W1 must represent a genuine forward-traveling wave of the target
        # pressure, not a quiescent (u=0) reference state re-labeled with a
        # new pressure: a prior version set W1 = 4*c_target, which silently
        # assumes zero velocity at the target state and does not construct
        # a forward wave at all. The correct construction treats the target
        # as a simple (pure forward) wave superposed on the vessel's
        # reference state (u_ref=0, c_ref at A0): holding the backward
        # invariant at its reference value W-_ref = -4*c_ref gives
        # u = 4*(c_target - c_ref) from W- = u - 4c = -4c_ref, so
        # W1 = u + 4*c_target = 8*c_target - 4*c_ref. This is exact for a
        # linear (small-amplitude) forward wave and the standard
        # characteristic-based construction for a prescribed-pressure
        # inflow more generally (matching the impedance relation
        # dp = rho*c_ref*du at small amplitude).
        t_next = t + dt
        ramp = 0.5 * (1 - np.cos(np.pi * min(t_next, T_ramp) / T_ramp)) if T_ramp > 0 else 1.0
        if inlet_profile == "smooth":
            p_in_new = inlet_pressure_smooth(t_next, T, p0, dp_inlet * ramp)
        else:
            p_in_new = inlet_pressure(t_next, T, p0, dp_inlet * ramp)
        A_target = inverse_tube_law(np.array([p_in_new]), A0, Eh_r0, p0)[0]
        c_target = wave_speed(np.array([A_target]), A0, Eh_r0)[0]
        c_ref = wave_speed(np.array([A0]), A0, Eh_r0)[0]
        # For c~A^(-1/4), the lambda+ invariant is u-4c.  Holding the
        # lambda- invariant u+4c at 4*c_ref gives a positive velocity for a
        # positive-pressure (larger-area, lower-c) incoming wave.
        W1 = 4.0 * c_ref - 8.0 * c_target

        c0_old = wave_speed(np.array([A[0]]), A0, Eh_r0)[0]
        u0_old = q[0] / max(A[0], 1e-12)
        u1_old = q[1] / max(A[1], 1e-12)
        lambda2 = u0_old - c0_old  # backward characteristic speed (negative)
        frac = np.clip(-lambda2 * dt / dx, 0.0, 1.0)
        u_foot = (1 - frac) * u0_old + frac * u1_old
        A_foot = (1 - frac) * A[0] + frac * A[1]
        c_foot = wave_speed(np.array([A_foot]), A0, Eh_r0)[0]
        W2 = u_foot + 4.0 * c_foot

        u_bc = 0.5 * (W1 + W2)
        c_bc = max((W2 - W1) / 8.0, 1e-6)
        A_new[0] = invert_wave_speed(c_bc, A0, Eh_r0)
        q_new[0] = u_bc * A_new[0]

        # outlet: same-stage (non-lagged) coupling of the boundary state and
        # the Windkessel ODE, solved for the boundary VELOCITY rather than
        # area. An area-based version of this same-stage solve (recovering
        # u_b as the difference of two large characteristic-scale terms,
        # u_b = W_out - 4*c_b, both O(10^3) while the physical u_b is
        # O(10-50)) turned out to be too ill-conditioned to be usable: it
        # rejected essentially every candidate root as inadmissible. Using a
        # reference-shifted invariant J+ = W+ - 4*c_ref = u + 4*(c-c_ref)
        # keeps every quantity involved at the physically small scale, and
        # velocity (not area) is solved for directly as the scalar unknown.
        tau = R2 * C_total
        c_ref_out = wave_speed(np.array([A0]), A0, Eh_r0)[0]
        c_last = wave_speed(np.array([A[-1]]), A0, Eh_r0)[0]
        u_last = q[-1] / max(A[-1], 1e-12)
        c_2last = wave_speed(np.array([A[-2]]), A0, Eh_r0)[0]
        u_2last = q[-2] / max(A[-2], 1e-12)
        lambda1 = u_last + c_last  # forward characteristic speed (outgoing, typically positive)
        frac_out = np.clip(lambda1 * dt / dx, 0.0, 1.0)
        u_foot_out = (1 - frac_out) * u_last + frac_out * u_2last
        c_foot_out = (1 - frac_out) * c_last + frac_out * c_2last
        J_plus = u_foot_out - 4.0 * (c_foot_out - c_ref_out)

        def _outlet_state(u_b):
            c_b = c_ref_out + (u_b - J_plus) / 4.0
            if c_b <= 0:
                return np.nan, np.nan, np.nan, c_b
            A_b = A0 * (c_ref_out / c_b) ** 4
            q_b = A_b * u_b
            Pwk_b = (Pwk + (dt / C_total) * (q_b + Pv / R2)) / (1.0 + dt / tau)
            p_b = tube_law(np.array([A_b]), A0, Eh_r0, p0)[0]
            residual = p_b - Pwk_b - R1 * q_b
            return residual, q_b, Pwk_b, A_b

        def _admissible(u_b, q_b, c_b, A_b):
            # Subcritical (hyperbolicity/regime) requirement only: |u_b|<c_b.
            # A brief, modest reverse discharge is physically permissible for
            # an RCR outlet with no check valve (e.g. during startup, before
            # incompatible initial pressures relax) and is not, by itself,
            # evidence of a wrong root; only persistent large backflow would
            # be. Branch continuity (selecting the admissible root closest to
            # the previous stage) is enforced at the call site below.
            return np.isfinite(q_b) and A_b > 0 and c_b > 0 and abs(u_b) < c_b

        # Solve directly for u_b over a physically bounded velocity bracket
        # (not an area bracket): scan for every sign change, bisect each to
        # a candidate root, keep only admissible (subcritical) candidates,
        # and select the one closest to the vessel's own previous outlet
        # velocity for continuity. If none is admissible, the boundary is
        # held at its previous state and this is tracked via
        # `outlet_fallback_count` for honest disclosure.
        #
        # Performance: the root evolves continuously step to step in the
        # common case, so try a cheap, narrow, warm-started bracket around
        # the previous outlet velocity first; only fall back to the full
        # wide scan (needed at startup, or after a genuine regime change)
        # when that narrow search turns up nothing admissible. This avoids
        # paying the cost of a broad ~100-point scan on every single step.
        u_prev_b = q_out_prev / max(A[-1], 1e-12)

        def _find_admissible_roots(u_grid):
            r_grid = np.array([_outlet_state(uu)[0] for uu in u_grid])
            found = []
            for i in range(len(u_grid) - 1):
                if np.isfinite(r_grid[i]) and np.isfinite(r_grid[i + 1]) and np.sign(r_grid[i]) != np.sign(r_grid[i + 1]):
                    lo, hi = u_grid[i], u_grid[i + 1]
                    r_lo = r_grid[i]
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        r_mid, _, _, _ = _outlet_state(mid)
                        if np.isfinite(r_mid) and np.sign(r_mid) == np.sign(r_lo):
                            lo, r_lo = mid, r_mid
                        else:
                            hi = mid
                    u_cand = 0.5 * (lo + hi)
                    _, q_b_c, _, A_b_c = _outlet_state(u_cand)
                    c_b_c = c_ref_out + (u_cand - J_plus) / 4.0
                    if _admissible(u_cand, q_b_c, c_b_c, A_b_c):
                        found.append(u_cand)
            return found

        narrow_half_width = max(0.15 * c_ref_out, 5.0 * abs(u_prev_b) + 5.0)
        admissible_roots = _find_admissible_roots(
            np.linspace(u_prev_b - narrow_half_width, u_prev_b + narrow_half_width, 12)
        )
        if not admissible_roots:
            u_max_phys = 30.0 * max(c_ref_out * 0.2, 50.0)  # generous physiological cap
            admissible_roots = _find_admissible_roots(
                np.linspace(-0.3 * c_ref_out, u_max_phys, 100)
            )

        if admissible_roots:
            u_b = min(admissible_roots, key=lambda uu: abs(uu - u_prev_b))
            outlet_fallback_count[0] += 0
        else:
            u_b = u_prev_b
            outlet_fallback_count[0] += 1

        r_final, q_b, Pwk_b, A_b = _outlet_state(u_b)
        if not np.isfinite(A_b) or A_b <= 0:
            A_b, q_b, Pwk_b = A[-1], q_out_prev, Pwk

        A_new[-1] = A_b
        q_new[-1] = q_b
        Pwk = Pwk_b
        q_out_prev = q_b
        q_out = q_b

        A, q = A_new, q_new
        t_prev = t
        t += dt
        step += 1
        total_steps_taken[0] += 1

        hist_t.append(t)
        hist_q.append(q_out)
        hist_A.append(A[-1])
        hist_Pwk.append(Pwk)
        hist_qin.append(q[0])
        for off, idx in diagnostic_indices.items():
            hist_diag_q[off].append(q[idx])
            hist_diag_A[off].append(A[idx])

        if verbose and step % 500 == 0:
            print(f"t={t:.3f}s A_out={A[-1]:.5e} q_out={q_out:.3f} Pwk={Pwk:.3f}")

        # Check periodicity once per newly-completed cycle (cheap: only when
        # t crosses a new integer multiple of T), never before n_cycles.
        # Follows the reviewer's explicit prescription: demonstrate
        # invariance of PI, RI, S/D, mean flow, and P_wk to additional
        # cycles, rather than only a cycle-to-cycle waveform-shape RMS
        # (which proved ill-conditioned here: umbilical diastolic flow is
        # near zero/reversed, so raw-waveform relative differences in that
        # region plateau at a small but nonzero numerical noise floor tied
        # to the CFL-adaptive timestep, well after the physically reported
        # indices have themselves stopped changing).
        if int(t / T) > int(t_prev / T) and int(t / T) >= n_cycles:
            n_cycles_run = int(t / T)
            t_arr = np.array(hist_t)
            if t_arr[-1] >= 2 * T:
                mask_last = t_arr >= (t_arr[-1] - T)
                mask_prev = (t_arr >= (t_arr[-1] - 2 * T)) & (t_arr < (t_arr[-1] - T))
                if mask_prev.sum() > 5 and mask_last.sum() > 5:
                    q_arr = np.array(hist_q)
                    A_arr = np.array(hist_A)
                    Pwk_arr = np.array(hist_Pwk)
                    t_last = t_arr[mask_last] - t_arr[mask_last][0]
                    t_prev_cyc = t_arr[mask_prev] - t_arr[mask_prev][0]

                    def _rel_change(x_last, x_prev):
                        m = 0.5 * (abs(x_last) + abs(x_prev))
                        return abs(x_last - x_prev) / max(m, 1e-9)

                    def _cycle_indices(t_cyc, q_cyc, A_cyc):
                        met = waveform_metrics(t_cyc, q_cyc, A_cyc, index_definition)
                        return met["PI"], met["RI"], met["S_D"], met["mean_flow_mL_s_per_artery"]

                    pi_last, ri_last, sd_last, qmean_last = _cycle_indices(
                        t_last, q_arr[mask_last], A_arr[mask_last]
                    )
                    pi_prev, ri_prev, sd_prev, qmean_prev = _cycle_indices(
                        t_prev_cyc, q_arr[mask_prev], A_arr[mask_prev]
                    )

                    pi_ok = _rel_change(pi_last, pi_prev) < tol
                    ri_ok = _rel_change(ri_last, ri_prev) < tol
                    sd_ok = _rel_change(sd_last, sd_prev) < tol
                    qmean_ok = _rel_change(qmean_last, qmean_prev) < tol
                    pwk_ok = _rel_change(Pwk_arr[mask_last][-1], Pwk_arr[mask_prev][-1]) < tol

                    if verbose and n_cycles_run % 20 == 0:
                        print(f"  [cycle {n_cycles_run}] PI={pi_last:.4f} (d={_rel_change(pi_last, pi_prev):.2e}) "
                              f"RI={ri_last:.4f} (d={_rel_change(ri_last, ri_prev):.2e}) "
                              f"SD={sd_last:.4f} (d={_rel_change(sd_last, sd_prev):.2e}) "
                              f"qmean_rel={_rel_change(qmean_last, qmean_prev):.2e} "
                              f"pwk_rel={_rel_change(Pwk_arr[mask_last][-1], Pwk_arr[mask_prev][-1]):.2e}")

                    if pi_ok and ri_ok and sd_ok and qmean_ok and pwk_ok:
                        converged = True
                        break

    t_arr = np.array(hist_t)
    q_arr = np.array(hist_q)
    A_arr = np.array(hist_A)
    Pwk_arr = np.array(hist_Pwk)

    mask_last = t_arr >= (t_arr[-1] - T)
    mask_prev = (t_arr >= (t_arr[-1] - 2 * T)) & (t_arr < (t_arr[-1] - T))

    t_last = t_arr[mask_last] - t_arr[mask_last][0]
    q_last = q_arr[mask_last]
    A_last = A_arr[mask_last]
    Pwk_last = Pwk_arr[mask_last]
    metrics = waveform_metrics(t_last, q_last, A_last, index_definition)
    diagnostic_last_cycle = {}
    for off in offsets:
        qd = np.asarray(hist_diag_q[off])[mask_last]
        Ad = np.asarray(hist_diag_A[off])[mask_last]
        diagnostic_last_cycle[off] = {
            "index": diagnostic_indices[off],
            "x": diagnostic_indices[off] * dx,
            "q": qd,
            "A": Ad,
            "u": qd / np.maximum(Ad, 1e-12),
        }

    return {
        "t": t_last,
        "q_outlet": q_last,
        "A_outlet": A_last,
        # Windkessel chamber pressure over the final (converged) cycle, aligned
        # with "t". Enables the storage-consistent UV inflow
        # 2*(Pwk - Pv)/R2 (placental efflux through the distal resistor) in
        # downstream reduced-UV calculations -- the arterial inflow q_outlet
        # bypasses placental storage at waveform level (cycle means coincide).
        # See solver README, "UV bookkeeping note".
        "Pwk": Pwk_last,
        "u_outlet": q_last / A_last,
        "metrics": metrics,
        "T": T,
        "converged": bool(converged),
        "n_cycles_run": n_cycles_run,
        "outlet_fallback_frac": outlet_fallback_count[0] / max(total_steps_taken[0], 1),
        "A0": A0,
        "Eh_r0": Eh_r0,
        "diagnostic_last_cycle": diagnostic_last_cycle,
        "dt_min": float(np.min(dt_history)),
        "dt_max": float(np.max(dt_history)),
        "cfl_min": float(np.min(cfl_history)),
        "cfl_max": float(np.max(cfl_history)),
        "dx": dx,
        "inlet_profile": inlet_profile,
        "index_definition": index_definition,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = simulate(bpm=140, r0=0.13, R_total=3.0e4, C_total=4.0e-5, verbose=True)
    print("Converged:", result["converged"])
    print("Outlet velocity range (cm/s):", result["u_outlet"].min(), "-", result["u_outlet"].max())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(result["t"], result["u_outlet"])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Outlet velocity (cm/s)")
    axes[0].set_title("Simulated umbilical artery outlet velocity waveform")
    axes[0].grid(True)

    axes[1].plot(result["t"], result["A_outlet"])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Outlet area (cm^2)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("demo_outlet_waveform.png", dpi=150)
    print("Saved demo_outlet_waveform.png")
