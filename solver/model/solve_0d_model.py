"""
0D (lumped-parameter) alternative to the 1D nonlinear PDE model, per
zero_d_model_sketch.md. The umbilical artery itself is represented as a
single lumped inertance-plus-resistance element (Poiseuille resistance
R_a ~ r0^-4, inertance L ~ r0^-2) in series with the same three-element
Windkessel used throughout this project, giving a 2-state linear ODE system
driven by the same periodic inlet pressure pulse used in the 1D model:

    L dq/dt  = p_in(t) - R_prox*q - P_C          (R_prox = R_a + R1)
    C dP_C/dt = q - (P_C - Pv)/R2

No spatial mesh, no CFL condition, no characteristic boundary algebra --
the specific failure mode that made the 1D model's Late-gestation regime
unstable cannot arise here by construction.
"""

import numpy as np
from scipy.integrate import solve_ivp

from solve_arterial_flow_UA import _inlet_shape, _shape_mean, P0_DEFAULT, DP_DEFAULT

RHO = 1.05          # g/cm^3
MU = 0.035          # dynamic viscosity, poise (dyn*s/cm^2)


def lumped_params(r0, R_total, R1_frac=0.15, vessel_length=20.0):
    A0 = np.pi * r0 ** 2
    R_a = 8.0 * MU * vessel_length / (np.pi * r0 ** 4)
    L_inert = RHO * vessel_length / (np.pi * r0 ** 2)
    R1 = R1_frac * R_total
    R2 = R_total - R1
    R_prox = R_a + R1
    return A0, R_a, L_inert, R1, R2, R_prox


def simulate_0d(bpm, r0, R_total, C_total, R1_frac=0.15, p0=P0_DEFAULT,
                 dp_inlet=DP_DEFAULT, vessel_length=20.0, Pv=0.0,
                 n_cycles_min=10, n_cycles_max=400, tol=1e-4,
                 systolic_frac=0.35, n_phase=200):
    A0, R_a, L_inert, R1, R2, R_prox = lumped_params(r0, R_total, R1_frac, vessel_length)
    T = 60.0 / bpm
    tau = R2 * C_total

    shape_mean = _shape_mean(systolic_frac)

    def p_in(t):
        phase = (t % T) / T
        shape = _inlet_shape(np.array([phase]), systolic_frac)[0]
        shape_norm = shape / shape_mean
        return p0 + dp_inlet * (shape_norm - 1.0)

    def rhs(t, y):
        q, Pc = y
        dq = (p_in(t) - R_prox * q - Pc) / L_inert
        dPc = (q - (Pc - Pv) / R2) / C_total
        return [dq, dPc]

    # initial state: quiescent, mutually consistent (q=0, Pc=p0), matching
    # the 1D model's post-fix initialization
    y0 = [0.0, p0]

    phase_grid = np.linspace(0, 1, n_phase, endpoint=False)
    t_cursor = 0.0
    y_cursor = y0
    prev_cycle_q = None
    n_cycles_run = 0
    converged = False

    max_cycles = min(n_cycles_max, max(n_cycles_min, int(np.ceil(10 * tau / T))))

    for cyc in range(max_cycles):
        t_span = (t_cursor, t_cursor + T)
        t_eval = t_cursor + phase_grid * T
        sol = solve_ivp(rhs, t_span, y_cursor, t_eval=t_eval, method="RK45",
                         rtol=1e-8, atol=1e-10, max_step=T / 200)
        q_cycle = sol.y[0]
        y_cursor = sol.y[:, -1]
        t_cursor += T
        n_cycles_run = cyc + 1

        if prev_cycle_q is not None and n_cycles_run >= n_cycles_min:
            amp = q_cycle.max() - q_cycle.min()
            rms_diff = np.sqrt(np.mean((q_cycle - prev_cycle_q) ** 2))
            if (rms_diff / max(amp, 1e-12)) < tol:
                converged = True
                prev_cycle_q = q_cycle
                break
        prev_cycle_q = q_cycle

    q_last = prev_cycle_q
    u_last = q_last / A0
    t_last = phase_grid * T

    return dict(
        t=t_last, q_outlet=q_last, u_outlet=u_last, T=T,
        converged=bool(converged), n_cycles_run=n_cycles_run,
        A0=A0, R_a=R_a, L_inert=L_inert, R1=R1, R2=R2,
    )


def indices_from_result(r):
    u = r["u_outlet"]
    v_sys = u.max()
    v_dias = max(u.min(), 0.05 * v_sys)
    v_mean = np.trapezoid(u, r["t"]) / (r["t"][-1] - r["t"][0])
    sd = v_sys / max(v_dias, 1e-9)
    pi = (v_sys - v_dias) / max(v_mean, 1e-9)
    ri = (v_sys - v_dias) / max(v_sys, 1e-9)
    return sd, pi, ri


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = simulate_0d(bpm=161.0, r0=0.05, R_total=4.0e5, C_total=4.0e-5, dp_inlet=8.3e4)
    sd, pi, ri = indices_from_result(r)
    print(f"Early-like: SD={sd:.3f} PI={pi:.3f} RI={ri:.3f} conv={r['converged']} "
          f"n_cyc={r['n_cycles_run']} time={time.time()-t0:.2f}s")

    t0 = time.time()
    r = simulate_0d(bpm=149.3, r0=0.10, R_total=7364.5, C_total=4.0e-5, dp_inlet=2.17e4)
    sd, pi, ri = indices_from_result(r)
    print(f"Late-like: SD={sd:.3f} PI={pi:.3f} RI={ri:.3f} conv={r['converged']} "
          f"n_cyc={r['n_cycles_run']} time={time.time()-t0:.2f}s")
