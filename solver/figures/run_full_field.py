"""
Full-spatial-field variant of solve_arterial_flow_UA.simulate(): identical
physics, boundary conditions, and convergence criteria (see the module
docstring there), but additionally records the FULL A(x), q(x) state -- not
just the outlet time series -- at evenly spaced phase points within the
final converged cardiac cycle, plus the Windkessel chamber pressure Pwk(t)
over that cycle (needed for the paper's qven(t) = (Pwk(t)-Pv)/R2 lumped
venous-flow relation, Section II-F, reused here unchanged for the UV colour
in the manuscript's Fig. 6 and the companion interactive flow viewer).

Run at the production operating point from calibration/run_full_recalibration.py
(results/full_recalibration.json, table_I entry for the requested GA), i.e.
the actual validated fit -- not an ad hoc or illustrative one.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import solve_arterial_flow_UA as m


def simulate_full_field(bpm, r0, R_total, C_total, L, M, n_phases=48,
                         R1_frac=0.15, p0=m.P0_DEFAULT, dp_inlet=m.DP_DEFAULT,
                         n_cycles=15, steps_per_cycle=400, Pv=0.0,
                         max_cycles=None, tol=1e-3, verbose=False, cfl=0.9):
    A0 = np.pi * r0 ** 2
    Eh_r0 = m.stiffness(r0)
    dx = L / (M - 1)
    T = 60.0 / bpm

    R1 = R1_frac * R_total
    R2 = R_total - R1
    tau = R2 * C_total

    T_ramp = min(max(3 * T, 3 * tau), 40 * T)
    n_cycles = max(n_cycles, int(np.ceil(T_ramp / T)) + 2)
    if max_cycles is None:
        max_cycles = int(np.clip(np.ceil(10 * tau / T), n_cycles, 2000))
    else:
        max_cycles = max(max_cycles, n_cycles)

    A = np.full(M, A0)
    q = np.zeros(M)
    Pwk = p0
    q_out_prev = 0.0

    dt_nominal = T / steps_per_cycle
    t = 0.0
    c_ref_est = np.sqrt((2.0 / 3.0) * Eh_r0 / m.RHO)
    dt_cfl_est = cfl * dx / max(3.0 * c_ref_est, 1e-6)
    steps_per_cycle_est = max(steps_per_cycle, int(np.ceil(T / dt_cfl_est)))
    max_steps = max_cycles * steps_per_cycle_est * 2

    hist_t, hist_q, hist_A, hist_Pwk = [], [], [], []
    buf_t, buf_A, buf_q, buf_Pwk = [], [], [], []
    buf_window = 1.25 * T

    n_cycles_run = 0
    converged = False
    outlet_fallback_count = [0]
    total_steps_taken = [0]

    step = 0
    while t < max_cycles * T and step < max_steps:
        c = m.wave_speed(A, A0, Eh_r0)
        u = q / np.maximum(A, 1e-12)
        c_max = np.max(np.abs(u) + c)
        dt = min(dt_nominal, cfl * dx / max(c_max, 1e-6))

        F1, F2 = m.flux(A, q, A0, Eh_r0)
        S2 = m.friction_source(A, q)

        Ahalf = 0.5 * (A[:-1] + A[1:]) - (dt / (2 * dx)) * (F1[1:] - F1[:-1])
        qhalf = (0.5 * (q[:-1] + q[1:]) - (dt / (2 * dx)) * (F2[1:] - F2[:-1])
                 + (dt / 2) * 0.5 * (S2[:-1] + S2[1:]))
        F1h, F2h = m.flux(Ahalf, qhalf, A0, Eh_r0)
        S2h = m.friction_source(Ahalf, qhalf)

        A_new = A.copy()
        q_new = q.copy()
        A_new[1:-1] = A[1:-1] - (dt / dx) * (F1h[1:] - F1h[:-1])
        q_new[1:-1] = (q[1:-1] - (dt / dx) * (F2h[1:] - F2h[:-1])
                       + dt * 0.5 * (S2h[1:] + S2h[:-1]))

        t_next = t + dt
        ramp = 0.5 * (1 - np.cos(np.pi * min(t_next, T_ramp) / T_ramp)) if T_ramp > 0 else 1.0
        p_in_new = m.inlet_pressure(t_next, T, p0, dp_inlet * ramp)
        A_target = m.inverse_tube_law(np.array([p_in_new]), A0, Eh_r0, p0)[0]
        c_target = m.wave_speed(np.array([A_target]), A0, Eh_r0)[0]
        c_ref = m.wave_speed(np.array([A0]), A0, Eh_r0)[0]
        W1 = 4.0 * c_ref - 8.0 * c_target

        c0_old = m.wave_speed(np.array([A[0]]), A0, Eh_r0)[0]
        u0_old = q[0] / max(A[0], 1e-12)
        u1_old = q[1] / max(A[1], 1e-12)
        lambda2 = u0_old - c0_old
        frac = np.clip(-lambda2 * dt / dx, 0.0, 1.0)
        u_foot = (1 - frac) * u0_old + frac * u1_old
        A_foot = (1 - frac) * A[0] + frac * A[1]
        c_foot = m.wave_speed(np.array([A_foot]), A0, Eh_r0)[0]
        W2 = u_foot + 4.0 * c_foot

        u_bc = 0.5 * (W1 + W2)
        c_bc = max((W2 - W1) / 8.0, 1e-6)
        A_new[0] = m.invert_wave_speed(c_bc, A0, Eh_r0)
        q_new[0] = u_bc * A_new[0]

        tau = R2 * C_total
        c_ref_out = m.wave_speed(np.array([A0]), A0, Eh_r0)[0]
        c_last = m.wave_speed(np.array([A[-1]]), A0, Eh_r0)[0]
        u_last = q[-1] / max(A[-1], 1e-12)
        c_2last = m.wave_speed(np.array([A[-2]]), A0, Eh_r0)[0]
        u_2last = q[-2] / max(A[-2], 1e-12)
        lambda1 = u_last + c_last
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
            p_b = m.tube_law(np.array([A_b]), A0, Eh_r0, p0)[0]
            residual = p_b - Pwk_b - R1 * q_b
            return residual, q_b, Pwk_b, A_b

        def _admissible(u_b, q_b, c_b, A_b):
            return np.isfinite(q_b) and A_b > 0 and c_b > 0 and abs(u_b) < c_b

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
            np.linspace(u_prev_b - narrow_half_width, u_prev_b + narrow_half_width, 12))
        if not admissible_roots:
            u_max_phys = 30.0 * max(c_ref_out * 0.2, 50.0)
            admissible_roots = _find_admissible_roots(
                np.linspace(-0.3 * c_ref_out, u_max_phys, 100))

        if admissible_roots:
            u_b = min(admissible_roots, key=lambda uu: abs(uu - u_prev_b))
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

        A, q = A_new, q_new
        t_prev = t
        t += dt
        step += 1
        total_steps_taken[0] += 1

        hist_t.append(t); hist_q.append(q_b); hist_A.append(A[-1]); hist_Pwk.append(Pwk)

        buf_t.append(t); buf_A.append(A.copy()); buf_q.append(q.copy()); buf_Pwk.append(Pwk)
        while buf_t and buf_t[0] < t - buf_window:
            buf_t.pop(0); buf_A.pop(0); buf_q.pop(0); buf_Pwk.pop(0)

        if verbose and step % 2000 == 0:
            print(f"t={t:.3f}s A_out={A[-1]:.5e} q_out={q_b:.3f} Pwk={Pwk:.3f}")

        if int(t / T) > int(t_prev / T) and int(t / T) >= n_cycles:
            n_cycles_run = int(t / T)
            t_arr = np.array(hist_t)
            if t_arr[-1] >= 2 * T:
                mask_last = t_arr >= (t_arr[-1] - T)
                mask_prev = (t_arr >= (t_arr[-1] - 2 * T)) & (t_arr < (t_arr[-1] - T))
                if mask_prev.sum() > 5 and mask_last.sum() > 5:
                    q_arr = np.array(hist_q); A_arr = np.array(hist_A); Pwk_arr = np.array(hist_Pwk)
                    t_last = t_arr[mask_last] - t_arr[mask_last][0]
                    t_prev_cyc = t_arr[mask_prev] - t_arr[mask_prev][0]

                    def _rel_change(x_last, x_prev):
                        mm = 0.5 * (abs(x_last) + abs(x_prev))
                        return abs(x_last - x_prev) / max(mm, 1e-9)

                    def _cycle_indices(t_cyc, q_cyc, A_cyc):
                        u_cyc = q_cyc / np.maximum(A_cyc, 1e-12)
                        v_sys = u_cyc.max(); v_dias = max(u_cyc.min(), 0.05 * v_sys)
                        v_mean = np.trapezoid(u_cyc, t_cyc) / max(t_cyc[-1] - t_cyc[0], 1e-9)
                        pi = (v_sys - v_dias) / max(abs(v_mean), 1e-9)
                        ri = (v_sys - v_dias) / max(abs(v_sys), 1e-9)
                        sd = v_sys / max(abs(v_dias), 1e-9)
                        qmean = np.trapezoid(q_cyc, t_cyc) / max(t_cyc[-1] - t_cyc[0], 1e-9)
                        return pi, ri, sd, qmean

                    pi_last, ri_last, sd_last, qmean_last = _cycle_indices(t_last, q_arr[mask_last], A_arr[mask_last])
                    pi_prev, ri_prev, sd_prev, qmean_prev = _cycle_indices(t_prev_cyc, q_arr[mask_prev], A_arr[mask_prev])

                    ok = (_rel_change(pi_last, pi_prev) < tol and _rel_change(ri_last, ri_prev) < tol
                          and _rel_change(sd_last, sd_prev) < tol and _rel_change(qmean_last, qmean_prev) < tol
                          and _rel_change(Pwk_arr[mask_last][-1], Pwk_arr[mask_prev][-1]) < tol)
                    if verbose:
                        print(f"  [cycle {n_cycles_run}] PI={pi_last:.4f} RI={ri_last:.4f} ok={ok}")
                    if ok:
                        converged = True
                        break

    buf_t = np.array(buf_t)
    t_end = buf_t[-1]
    t_start = t_end - T
    phase_targets = t_start + np.linspace(0, T, n_phases, endpoint=False)
    snap_idx = [int(np.argmin(np.abs(buf_t - pt))) for pt in phase_targets]
    phases_A = np.array([buf_A[i] for i in snap_idx])
    phases_q = np.array([buf_q[i] for i in snap_idx])
    phases_Pwk = np.array([buf_Pwk[i] for i in snap_idx])
    phase_t = buf_t[snap_idx] - t_start

    x = np.linspace(0, L, M)
    return {
        "x_cm": x, "L_cm": L, "M": M,
        "phase_t_s": phase_t, "T_s": T,
        "A": phases_A, "q": phases_q, "Pwk": phases_Pwk,
        "A0": A0, "Eh_r0": Eh_r0, "p0": p0,
        "R1": R1, "R2": R2, "C_total": C_total, "Pv": Pv,
        "converged": bool(converged), "n_cycles_run": n_cycles_run,
        "outlet_fallback_frac": outlet_fallback_count[0] / max(total_steps_taken[0], 1),
        "r0": r0, "bpm": bpm, "R_total": R_total, "dp_inlet": dp_inlet,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "calibration"))
    from gestational_geometry import ua_radius_cm

    GA = 25
    recal = json.loads((ROOT / "results" / "full_recalibration.json").read_text())
    fit = next(r for r in recal["table_I"] if r["ga_weeks"] == GA)
    L_cm = fit["length_cm"]
    dx_cm = recal["geometry"]["production_dx_cm"]
    M = int(round(L_cm / dx_cm)) + 1
    r0 = ua_radius_cm(GA, allow_extrapolation=False)
    print(f"GA={GA} L={L_cm} M={M} r0={r0:.6f} "
          f"R_total={fit['R_total_dyn_s_cm5']:.2f} dp_inlet={fit['dp_inlet_dyn_cm2']:.2f}")

    result = simulate_full_field(
        bpm=140.0, r0=r0, R_total=fit["R_total_dyn_s_cm5"], C_total=4.0e-5,
        L=L_cm, M=M, n_phases=48, dp_inlet=fit["dp_inlet_dyn_cm2"], p0=m.P0_DEFAULT,
        n_cycles=15, verbose=True,
    )
    print("converged:", result["converged"], "cycles:", result["n_cycles_run"],
          "outlet_fallback_frac:", result["outlet_fallback_frac"])
    out_path = Path(__file__).parent / f"full_field_GA{GA}.npz"
    np.savez(out_path, **result)
    print("saved", out_path)
