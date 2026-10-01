import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import solve_arterial_flow_UA as ua
from gestational_geometry import (
    UA_TRUNK_LENGTH_CM,
    UV_TRUNK_LENGTH_CM,
    flow_from_tav_mL_min,
    uv_radius_cm,
)
from solve_venous_flow_UV import simulate_uv_return


def test_waveform_metrics_known_signal():
    t = np.linspace(0.0, 1.0, 1001)
    u = 20.0 + 10.0 * np.sin(2.0 * np.pi * t)
    A = np.full_like(t, 0.1)
    q = A * u
    metrics = ua.waveform_metrics(t, q, A, "end_cycle")
    assert np.isclose(metrics["PSV_cm_s"], 30.0, atol=1e-3)
    assert np.isclose(metrics["EDV_cm_s"], 20.0, atol=1e-3)
    assert np.isclose(metrics["TAV_cm_s"], 20.0, atol=1e-3)
    assert np.isclose(metrics["PI"], 0.5, atol=1e-3)
    assert np.isclose(metrics["RI"], 1.0 / 3.0, atol=1e-3)


def test_legacy_floor_is_explicit():
    t = np.linspace(0.0, 1.0, 5)
    A = np.ones_like(t)
    q = np.array([5.0, 20.0, -2.0, 8.0, 5.0])
    legacy = ua.waveform_metrics(t, q, A, "legacy_floor")
    raw = ua.waveform_metrics(t, q, A, "minimum")
    assert legacy["EDV_cm_s"] == 1.0
    assert raw["EDV_cm_s"] == -2.0


def test_uv_reference_and_flow_conversion():
    assert UA_TRUNK_LENGTH_CM == 72.2
    assert UV_TRUNK_LENGTH_CM == 61.78
    assert ua.UA_LENGTH_DEFAULT_CM == UA_TRUNK_LENGTH_CM
    assert np.isclose(uv_radius_cm(26.0), 0.29)
    expected = np.pi * (5.8 / 20.0) ** 2 * 10.0 * 60.0
    assert np.isclose(flow_from_tav_mL_min(5.8, 10.0), expected)


def test_rigid_uv_conserves_imposed_return_flow():
    t = np.linspace(0.0, 1.0, 1001)
    q = 2.0 + 0.2 * np.sin(2 * np.pi * t)
    result = simulate_uv_return(t, q, 26.0)
    assert np.isclose(result["mean_flow_mL_min"], 120.0)
    assert np.allclose(result["flow_mL_s"], q)
