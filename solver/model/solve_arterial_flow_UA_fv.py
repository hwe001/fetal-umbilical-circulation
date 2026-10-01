"""Stage-consistent finite-volume reference solver for the 1D UA/RCR model.

This is intentionally a conservative, first-order-in-space reference method
(Rusanov flux + SSP-RK2).  Its purpose is to establish a robust convergent
baseline before adding higher-order MUSCL reconstruction.  Unlike the legacy
solver, vessel unknowns are cell averages; the algebraic RCR boundary state is
used to form a boundary *flux* and never overwrites the last interior cell.
"""

import numpy as np

from solve_arterial_flow_UA import (
    RHO, NU, stiffness, tube_law, inverse_tube_law, wave_speed,
    inlet_pressure, inlet_pressure_smooth, friction_source, psi,
)


def _physical_flux(A, q, A0, Eh):
    return np.stack((q, q*q/np.maximum(A, 1e-12) + psi(A, A0, Eh)), axis=0)


def _rusanov(AL, qL, AR, qR, A0, Eh):
    FL = _physical_flux(np.asarray(AL), np.asarray(qL), A0, Eh)
    FR = _physical_flux(np.asarray(AR), np.asarray(qR), A0, Eh)
    uL = qL / np.maximum(AL, 1e-12)
    uR = qR / np.maximum(AR, 1e-12)
    a = np.maximum(np.abs(uL) + wave_speed(AL, A0, Eh),
                   np.abs(uR) + wave_speed(AR, A0, Eh))
    return 0.5*(FL + FR) - 0.5*a*np.stack((AR-AL, qR-qL), axis=0)


def _inlet_state(Ai, qi, t, T, p0, dp, ramp, A0, Eh, profile):
    pressure = (inlet_pressure_smooth if profile == "smooth" else inlet_pressure)(
        t, T, p0, dp*ramp
    )
    At = inverse_tube_law(np.array([pressure]), A0, Eh, p0)[0]
    ct = wave_speed(np.array([At]), A0, Eh)[0]
    cref = wave_speed(np.array([A0]), A0, Eh)[0]
    # Because c(A) ~ A^(-1/4), integral(c/A dA) = -4c.  Hence the
    # invariant carried by lambda+=u+c is u-4c, while that carried by
    # lambda-=u-c is u+4c (the opposite sign assignment from the legacy
    # implementation).  A positive-pressure forward wave has ct<cref and
    # therefore u=4(cref-ct)>0 when the outgoing lambda- invariant is held
    # at its reference value.
    Wplus = 4.0*cref - 8.0*ct
    ui = qi/max(Ai, 1e-12)
    ci = wave_speed(np.array([Ai]), A0, Eh)[0]
    Wminus = ui + 4.0*ci
    ub = 0.5*(Wplus + Wminus)
    cb = max((Wminus-Wplus)/8.0, 1e-8)
    Ab = A0*(cref/cb)**4
    return Ab, Ab*ub


def _outlet_state(Ai, qi, Pwk, R1, p0, A0, Eh, previous_u=0.0):
    """Solve the instantaneous characteristic/R1 relation at an RK stage."""
    cref = wave_speed(np.array([A0]), A0, Eh)[0]
    ui = qi/max(Ai, 1e-12)
    ci = wave_speed(np.array([Ai]), A0, Eh)[0]
    # Shifted outgoing lambda+ invariant: (u-4c)+4cref.
    Jplus = ui - 4.0*(ci-cref)

    def state(u):
        c = cref + (u-Jplus)/4.0
        if c <= 0:
            return np.nan, np.nan, np.nan
        A = A0*(cref/c)**4
        q = A*u
        p = tube_law(np.array([A]), A0, Eh, p0)[0]
        return p-Pwk-R1*q, A, q

    # Locate all subcritical roots and follow the branch nearest the previous
    # boundary velocity.  The wide scan is a diagnostic safety net.
    umax = max(300.0, 0.8*cref)
    grid = np.linspace(-0.8*cref, umax, 25)
    vals = [state(u)[0] for u in grid]
    roots = []
    def bisect(lo, hi, flo, fhi):
        for _ in range(40):
            mid = 0.5*(lo+hi)
            fm = state(mid)[0]
            if not np.isfinite(fm):
                hi = mid
                continue
            if abs(fm) < 1e-8 or abs(hi-lo) < 1e-10:
                return mid
            if flo*fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5*(lo+hi)

    for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            continue
        if flo == 0 or flo*fhi < 0:
            try:
                root = lo if flo == 0 else bisect(lo, hi, flo, fhi)
                _, A, q = state(root)
                c = cref + (root-Jplus)/4.0
                if A > 0 and abs(root) < c:
                    roots.append((root, A, q))
            except ValueError:
                pass
    if not roots:
        raise RuntimeError("No admissible instantaneous RCR boundary root")
    return min(roots, key=lambda item: abs(item[0]-previous_u))


def simulate(bpm, r0, R_total, C_total, dp_inlet, M=200, L=20.0,
             R1_frac=0.15, p0=3.0e4, Pv=0.0, cfl=0.45,
             n_cycles=15, inlet_profile="original",
             diagnostic_offsets=(0.0, 0.1, 0.5, 1.0)):
    A0 = np.pi*r0*r0
    Eh = stiffness(r0)
    R1, R2 = R1_frac*R_total, (1.0-R1_frac)*R_total
    T = 60.0/bpm
    tau = R2*C_total
    ramp_time = min(max(3*T, 3*tau), 40*T)
    n_cycles = max(n_cycles, int(np.ceil(ramp_time/T))+2)
    dx = L/M
    x = (np.arange(M)+0.5)*dx
    A = np.full(M, A0)
    q = np.zeros(M)
    Pwk = p0
    previous_u = 0.0
    t = 0.0

    offsets = tuple(float(v) for v in diagnostic_offsets)
    indices = {off: int(np.argmin(np.abs(x-(L-off)))) for off in offsets}
    ht, hPwk, hb_u = [], [], []
    hq = {off: [] for off in offsets}
    hA = {off: [] for off in offsets}
    dts, cfls = [], []

    def rhs(AA, qq, PP, tt, prev_u):
        ramp = 0.5*(1-np.cos(np.pi*min(tt, ramp_time)/ramp_time)) if ramp_time else 1.0
        AL, qL = _inlet_state(AA[0], qq[0], tt, T, p0, dp_inlet,
                              ramp, A0, Eh, inlet_profile)
        ub, AR, qR = _outlet_state(AA[-1], qq[-1], PP, R1, p0, A0, Eh, prev_u)
        F = np.empty((2, M+1))
        F[:, 0] = _rusanov(AL, qL, AA[0], qq[0], A0, Eh)
        F[:, 1:M] = _rusanov(AA[:-1], qq[:-1], AA[1:], qq[1:], A0, Eh)
        F[:, M] = _rusanov(AA[-1], qq[-1], AR, qR, A0, Eh)
        dA = -(F[0, 1:]-F[0, :-1])/dx
        dq = -(F[1, 1:]-F[1, :-1])/dx + friction_source(AA, qq)
        dP = (qR-(PP-Pv)/R2)/C_total
        return dA, dq, dP, ub, AR, qR

    end = n_cycles*T
    while t < end:
        speed = np.max(np.abs(q/np.maximum(A, 1e-12))+wave_speed(A, A0, Eh))
        dt = min(cfl*dx/max(speed, 1e-12), end-t)
        dA1, dq1, dP1, ub1, _, _ = rhs(A, q, Pwk, t, previous_u)
        A1, q1, P1 = A+dt*dA1, q+dt*dq1, Pwk+dt*dP1
        if np.min(A1) <= 0:
            raise RuntimeError("Non-positive provisional area")
        dA2, dq2, dP2, ub2, Ab2, qb2 = rhs(A1, q1, P1, t+dt, ub1)
        A = A + 0.5*dt*(dA1+dA2)
        q = q + 0.5*dt*(dq1+dq2)
        Pwk = Pwk + 0.5*dt*(dP1+dP2)
        previous_u = ub2
        t += dt
        dts.append(dt); cfls.append(dt*speed/dx)
        ht.append(t); hPwk.append(Pwk); hb_u.append(ub2)
        for off, idx in indices.items():
            hq[off].append(q[idx]); hA[off].append(A[idx])

    ht = np.asarray(ht)
    mask = ht >= ht[-1]-T
    result = {
        "t": ht[mask]-ht[mask][0], "T": T, "Pwk": np.asarray(hPwk)[mask],
        "u_boundary": np.asarray(hb_u)[mask], "dx": dx,
        "dt_min": float(np.min(dts)), "dt_max": float(np.max(dts)),
        "cfl_max": float(np.max(cfls)), "n_cycles_run": n_cycles,
        "diagnostic_last_cycle": {},
    }
    for off, idx in indices.items():
        qd, Ad = np.asarray(hq[off])[mask], np.asarray(hA[off])[mask]
        result["diagnostic_last_cycle"][off] = {
            "index": idx, "x": x[idx], "q": qd, "A": Ad,
            "u": qd/np.maximum(Ad, 1e-12),
        }
    return result
