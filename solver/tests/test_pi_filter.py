import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import placental_pi_filter as pf
from gestational_geometry import uv_radius_cm
from solve_arterial_flow_UA import waveform_metrics
from solve_venous_flow_UV import simulate_uv_return

BPM = pf.MATCHED_1D_CASE["bpm"]


@pytest.fixture(scope="module")
def params():
    return pf.default_pi_filter_params()


@pytest.fixture(scope="module")
def result(params):
    return pf.simulate_pi_filter(BPM, params)


# ---------------------------------------------------------------------------
# 1. DC solution consistency
# ---------------------------------------------------------------------------
def test_dc_solution_matches_lti_dc_gain(params):
    A, B = pf.lti_system_matrices(params)
    p_mean, Pout = 3.0e4, 0.0
    u = np.array([p_mean, Pout])
    y_lti = np.linalg.solve(A, -B @ u)
    dc = pf.dc_solution(params, p_mean, Pout)
    y_dc = np.array([dc["Q0_mL_s"], dc["Pm0"], dc["P_UV0"]])
    assert np.allclose(y_dc, y_lti, rtol=1e-10)
    assert np.isclose(dc["R_dc"], params.Rs + params.R_UV + params.Rout,
                      rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Mean flow obeys Ohm's law on the DC resistance (measured p_in mean, F3)
# ---------------------------------------------------------------------------
def test_mean_flow_ohms_law(params):
    r = pf.simulate_pi_filter(BPM, params, tol=1e-6)
    assert r["converged"]
    dc = pf.dc_solution(params, r["mean_inlet_pressure"])
    q_sim = r["mean_Q_UA_mL_s"]
    assert np.isclose(q_sim, dc["Q0_mL_s"], rtol=1e-5)
    total = r["metrics"]["total_two_UA_flow_mL_min"]
    assert np.isclose(total, 2.0 * q_sim * 60.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Periodic convergence within budget; tau_max reported (F3)
# ---------------------------------------------------------------------------
def test_periodic_convergence_within_budget(result):
    assert result["converged"] is True
    assert result["n_cycles_run"] <= 60
    tau = result["diagnostics"]["tau_max_s"]
    assert np.isfinite(tau) and tau > 0
    # the cycle budget must reflect the slowest circuit time constant
    assert result["n_cycles_run"] >= 10.0 * tau / result["T"]


# ---------------------------------------------------------------------------
# 4. Mean-level convergence: shape-only budget leaves a measurable offset,
#    the tau-driven budget removes it (F3 guard)
# ---------------------------------------------------------------------------
def test_mean_level_convergence(result, params):
    r_short = pf.simulate_pi_filter(BPM, params, n_cycles_min=10)
    assert r_short["flow_conservation"]["max_abs_imbalance_rel"] <= 1e-3
    assert result["flow_conservation"]["max_abs_imbalance_rel"] <= 1e-5


# ---------------------------------------------------------------------------
# 5. Positivity at physiologic parameters
# ---------------------------------------------------------------------------
def test_positivity_physiologic(result):
    assert result["Pm"].min() > 0
    assert result["P_UV"].min() > 0
    assert result["Q_UA"].min() > 0
    assert result["metrics"]["EDV_cm_s"] > 0
    for key in ("positive_pressure", "positive_mean_flow", "positive_edv",
                "periodic", "stable_lti"):
        assert result["flags"][key] is True, key


# ---------------------------------------------------------------------------
# 6. Mean-flow conservation across the three ODEs in the periodic state.
#    The residual (storage) term is the periodicity drift C*dP/T, which
#    scales with the convergence tolerance tol: at tol=1e-6 it is ~1e-5
#    of the mean flow (tol*p0*Cm/(T*Q_mean)).
# ---------------------------------------------------------------------------
def test_mean_flow_conservation(params):
    r = pf.simulate_pi_filter(BPM, params, tol=1e-6)
    fc = r["flow_conservation"]
    q_mean = abs(fc["Q_UA_mean"])
    assert abs(fc["Q_UA_mean"] - fc["Q_UV_mean"]) <= 1e-5 * q_mean
    assert abs(fc["Q_UV_mean"] - fc["Q_out_mean"]) <= 1e-5 * q_mean
    assert abs(fc["mean_storage_Cm_dPm_dt"]) <= 1e-4 * q_mean
    assert abs(fc["mean_storage_Cuv_dP_UV_dt"]) <= 1e-4 * q_mean


# ---------------------------------------------------------------------------
# 7. Metric identities; EDV = minimum (legacy-floor guard)
# ---------------------------------------------------------------------------
def test_metrics_index_identities(result):
    m = result["metrics"]
    assert m["diastolic_definition"] == "minimum"
    assert np.isclose(m["S_D"], 1.0 / (1.0 - m["RI"]), rtol=1e-12)
    assert np.isclose(m["PI"], (m["PSV_cm_s"] - m["EDV_cm_s"]) / m["TAV_cm_s"],
                      rtol=1e-12)
    assert np.isclose(m["RI"], (m["PSV_cm_s"] - m["EDV_cm_s"]) / m["PSV_cm_s"],
                      rtol=1e-12)
    u = result["Q_UA"] / result["A0"]
    assert np.isclose(m["EDV_cm_s"], u.min(), rtol=1e-12)
    assert np.isclose(m["PSV_cm_s"], u.max(), rtol=1e-12)
    assert np.isclose(m["mean_flow_mL_s_per_artery"],
                      result["Q_UA"][:-1].mean(), rtol=1e-12)
    # cross-check against a direct waveform_metrics call
    direct = waveform_metrics(result["t"], result["Q_UA"],
                              np.full_like(result["t"], result["A0"]))
    assert np.isclose(direct["PI"], m["PI"], rtol=1e-12)


# ---------------------------------------------------------------------------
# 8. Two-UA shared-bed equivalence (exactly the same ODEs)
# ---------------------------------------------------------------------------
def test_shared_bed_equivalence(params):
    r_per = pf.simulate_pi_filter(BPM, params)
    r_shared = pf.simulate_pi_filter(BPM, pf.to_shared_bed(params, 2),
                                     per_artery=True)
    assert np.max(np.abs(r_per["Q_UA"] - r_shared["Q_UA"])) < 1e-12
    assert np.isclose(r_shared["metrics"]["PI"], r_per["metrics"]["PI"],
                      rtol=1e-12)
    assert r_shared["params"]["n_arteries"] == 1  # converted back internally
    # exact DC algebra: the merged circuit's DC flow is n x the per-artery one
    p_mean = 3.0e4
    dc_per = pf.dc_solution(params, p_mean)
    dc_sh = pf.dc_solution(pf.to_shared_bed(params, 2), p_mean)
    assert np.isclose(dc_sh["Q0_mL_s"], 2.0 * dc_per["Q0_mL_s"], rtol=1e-12)
    # running the merged circuit directly agrees to convergence tolerance
    # (different float associativity and convergence path)
    total_shared = pf.simulate_pi_filter(
        BPM, pf.to_shared_bed(params, 2), per_artery=False)
    assert total_shared["params"]["n_arteries"] == 2
    assert np.isclose(
        total_shared["metrics"]["total_two_UA_flow_mL_min"],
        2.0 * r_per["metrics"]["mean_flow_mL_s_per_artery"] * 60.0,
        rtol=1e-4)


# ---------------------------------------------------------------------------
# 9. R_UV consistent with the reduced 1D UV model; negligible vs the total
# ---------------------------------------------------------------------------
def test_r_uv_matches_reduced_uv_model(params):
    check = pf.uv_return_check(params, ga=26.0)
    assert np.isclose(check["R_UV_params"], check["R_UV_reference_formula"],
                      rtol=1e-12)
    assert np.isclose(check["R_UV_params"], check["R_UV_uv_model"], rtol=1e-9)
    assert np.isclose(uv_radius_cm(26.0), 0.29, rtol=1e-12)
    t = np.linspace(0.0, 1.0, 501)
    uv = simulate_uv_return(t, np.full_like(t, 0.294), 26.0)
    assert np.isclose(params.R_UV, uv["resistance_dyn_s_cm5"], rtol=1e-9)
    assert check["R_UV_fraction_of_total"] < 0.02, (
        "the cord-vein Poiseuille resistance should be a small fraction of "
        "the total series resistance; the handover's R_UV path is dominated "
        "by the villous circulation, not the cord vein")


# ---------------------------------------------------------------------------
# 10. Stiff parameter set: exact eigenvalue pre-check + Radau auto-switch
# ---------------------------------------------------------------------------
def test_stiff_parameter_set_reported_and_survives(params):
    stiff_params = pf.with_overrides(params, Cm=1e-9, Cuv=1e-9)
    with pytest.warns(pf.PiFilterStiffnessWarning):
        r = pf.simulate_pi_filter(BPM, stiff_params)
    assert r["flags"]["stiff"] is True
    assert r["flags"]["method_used"] == "Radau"
    assert np.all(np.isfinite(r["Q_UA"]))
    assert r["diagnostics"]["stiffness_ratio"] > 100


# ---------------------------------------------------------------------------
# 11. Non-physiologic parameters are flagged (warned), not raised
# ---------------------------------------------------------------------------
def test_non_physiologic_params_flagged_not_raised(params):
    bad = pf.with_overrides(params, Rs=5e3, Rout=9.5e4)
    with pytest.warns(pf.PiFilterPhysiologyWarning):
        r = pf.simulate_pi_filter(BPM, bad)
    assert r["flags"]["positive_edv"] is False
    with pytest.raises(ValueError):
        pf.simulate_pi_filter(BPM, bad, strict=True)


# ---------------------------------------------------------------------------
# 12. Finite shunt leak enters the DC resistance in parallel
# ---------------------------------------------------------------------------
def test_rp_shunt_changes_dc(params):
    Rp = 1e5
    leaky = pf.with_overrides(params, Rp=Rp)
    dc = pf.dc_solution(leaky, 3.0e4)
    r_par = Rp * (params.R_UV + params.Rout) / (Rp + params.R_UV + params.Rout)
    assert np.isclose(dc["R_dc"], params.Rs + r_par, rtol=1e-12)
    r = pf.simulate_pi_filter(BPM, leaky, tol=1e-6)
    r_base = pf.simulate_pi_filter(BPM, params, tol=1e-6)
    assert (r["metrics"]["total_two_UA_flow_mL_min"]
            > r_base["metrics"]["total_two_UA_flow_mL_min"])


# ---------------------------------------------------------------------------
# 13. Newton periodic initialization lands on the same periodic orbit
# ---------------------------------------------------------------------------
def test_newton_periodic_matches_warm_start(params):
    r_warm = pf.simulate_pi_filter(BPM, params)
    r_newton = pf.simulate_pi_filter(BPM, params, init="newton_periodic")
    assert r_newton["converged"] is True
    assert r_newton["n_cycles_run"] <= 8
    assert np.allclose(r_newton["Q_UA"], r_warm["Q_UA"], rtol=1e-5)


# ---------------------------------------------------------------------------
# 14. Provenance lock: default parameters equal the documented prior values
# ---------------------------------------------------------------------------
def test_default_params_are_the_matched_case():
    p = pf.default_pi_filter_params()
    # conduit prior: coefficient 22 (1D Olufsen friction) at r0=0.13, L=72.2
    assert np.isclose(p.Rs, 22.0 * pf.MU * 72.2 / (np.pi * 0.13 ** 4), rtol=1e-12)
    assert np.isclose(p.Ls, 1.05 * 72.2 / (np.pi * 0.13 ** 2), rtol=1e-12)
    # UV Poiseuille at GA 26 (diameter 5.8 mm)
    assert np.isclose(p.R_UV,
                      8.0 * pf.MU * pf.UV_TRUNK_LENGTH_CM
                      / (np.pi * 0.29 ** 4), rtol=1e-12)
    # DC mean-flow closure at the matched benchmark
    q_target = pf.MATCHED_1D_CASE["total_two_UA_flow_mL_min"] / 2.0 / 60.0
    r_series = 3.0e4 / q_target
    assert np.isclose(p.Rout, r_series - p.Rs - p.R_UV, rtol=1e-9)
    assert np.isclose(p.A0, np.pi * 0.13 ** 2, rtol=1e-12)
    assert p.Cm == pf.MATCHED_1D_CASE["C_total"]
    assert p.Rp == float("inf")  # shunt OFF by default (handover form)


# ---------------------------------------------------------------------------
# 15. UV coupling: storage-consistent efflux drives the reduced 1D UV model
# ---------------------------------------------------------------------------
def test_uv_coupling_conserves_mean_flow(params, result):
    c = pf.couple_uv_return(result, params, ga=26.0)
    # the rigid UV imposes the flow it is given
    assert np.isclose(c["uv"]["mean_flow_mL_min"],
                      c["mean_efflux_total_mL_min"], rtol=1e-12)
    # periodic state: cycle-mean efflux equals cycle-mean inflow up to the
    # convergence residual
    rel = abs(c["mean_efflux_minus_inflow_mL_min"]) \
        / c["mean_arterial_inflow_total_mL_min"]
    assert rel <= 1e-4


def test_uv_coupling_velocity_and_drop_consistency(params, result):
    c = pf.couple_uv_return(result, params, ga=26.0)
    uv = c["uv"]
    area = np.pi * uv["radius_cm"] ** 2
    assert np.isclose(uv["mean_velocity_cm_s"],
                      uv["mean_flow_mL_min"] / 60.0 / area, rtol=1e-9)
    # mean drop is dominated by R*q_mean; the inertial term is small
    expected = uv["resistance_dyn_s_cm5"] * c["mean_efflux_total_mL_min"] / 60.0
    assert 0.9 * expected <= np.mean(uv["pressure_drop_dyn_cm2"]) \
        <= 1.1 * expected


def test_uv_coupling_shared_bed_matches_per_artery(params, result):
    c_per = pf.couple_uv_return(result, params, ga=26.0)
    # run the merged circuit DIRECTLY and couple it with its own params
    shared = pf.to_shared_bed(params, 2)
    r_shared = pf.simulate_pi_filter(BPM, shared, per_artery=False)
    c_shared = pf.couple_uv_return(r_shared, shared, ga=26.0)
    assert np.allclose(c_per["efflux_total_mL_s"],
                       c_shared["efflux_total_mL_s"], rtol=1e-9)
    # mean level differs only by the merged run's convergence-path noise
    assert np.isclose(c_shared["uv"]["mean_velocity_cm_s"],
                      c_per["uv"]["mean_velocity_cm_s"], rtol=1e-7)


def test_uv_coupling_rejects_n_arteries_mismatch(params, result):
    # a per-artery result coupled with shared-bed params would silently
    # halve the total efflux -- must be rejected
    with pytest.raises(ValueError, match="n_arteries"):
        pf.couple_uv_return(result, pf.to_shared_bed(params, 2), ga=26.0)
