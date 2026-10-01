"""
Export a full_field_GA*.npz result to the companion geometry repo's
interactive flow viewer (umbilical-geometry/viewer/flow_data.js).

Field values are resampled onto a normalized arc-length fraction (0=inlet,
1=outlet), not absolute cm: the solver's segment length (Section III-B's
calibrated 72.2 cm) need not exactly equal the arc length of whichever
specific synthetic UA path is rendered in the viewer, so both the export
here and the client-side lookup in flow_app.js key on fraction of length,
not physical distance.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
VIEWER = HERE.parent.parent / "umbilical-geometry" / "viewer"

MMHG_PER_DYN_CM2 = 1.0 / 1333.22


def export(npz_path, meta_extra=None):
    ff = np.load(npz_path)
    x_cm = ff["x_cm"]; L_cm = float(ff["L_cm"])
    A = ff["A"]; q = ff["q"]; Pwk = ff["Pwk"]
    A0 = float(ff["A0"]); Eh_r0 = float(ff["Eh_r0"]); p0 = float(ff["p0"])
    R2 = float(ff["R2"]); Pv = float(ff["Pv"])
    phase_t = ff["phase_t_s"]; T = float(ff["T_s"])
    n_phases = A.shape[0]

    p_field = (p0 + (4.0 / 3.0) * Eh_r0 * (1.0 - np.sqrt(A0 / A))) * MMHG_PER_DYN_CM2
    q_field = q
    p_uv = Pwk * MMHG_PER_DYN_CM2
    # total UV inflow = placental efflux from BOTH arteries (symmetric pair),
    # i.e. twice the single-UA distal-resistor flow -- the storage-consistent
    # quantity per manuscript Section II-G (NOT the arterial inflow q_b)
    q_uv = 2.0 * (Pwk - Pv) / R2

    N_S = 101
    s_frac = np.linspace(0.0, 1.0, N_S)
    x_frac = x_cm / L_cm
    p_resampled = np.array([np.interp(s_frac, x_frac, p_field[k]) for k in range(n_phases)])
    q_resampled = np.array([np.interp(s_frac, x_frac, q_field[k]) for k in range(n_phases)])

    data = {
        "meta": {
            "ga_wk": 25,
            "pitch_mm": 50,
            "L_cm": round(L_cm, 2),
            "bpm": int(round(60.0 / T)),
            "T_s": round(T, 4),
            "converged": bool(ff["converged"]),
            "n_cycles_run": int(ff["n_cycles_run"]),
            "outlet_fallback_frac": float(ff["outlet_fallback_frac"]),
            "note": "Production fit from solver/results/full_recalibration.json (Table I, real reconstructed UA length).",
            **(meta_extra or {}),
        },
        "s_frac": s_frac.tolist(),
        "t_frac": (phase_t / T).tolist(),
        "p_mmHg": [[round(float(v), 4) for v in row] for row in p_resampled],
        "q_mL_s": [[round(float(v), 5) for v in row] for row in q_resampled],
        "p_uv_mmHg": [round(float(v), 4) for v in p_uv],
        "q_uv_mL_s": [round(float(v), 5) for v in q_uv],
        "p_min": round(float(min(p_resampled.min(), p_uv.min())), 3),
        "p_max": round(float(max(p_resampled.max(), p_uv.max())), 3),
        "q_min": round(float(min(q_resampled.min(), q_uv.min())), 3),
        "q_max": round(float(max(q_resampled.max(), q_uv.max())), 3),
    }

    out_path = VIEWER / "flow_data.js"
    out_path.write_text("window.FLOW_DATA = " + json.dumps(data) + ";\n")
    print("p range mmHg:", data["p_min"], data["p_max"])
    print("q range mL/s:", data["q_min"], data["q_max"])
    print("wrote", out_path)


if __name__ == "__main__":
    export(HERE / "full_field_GA25.npz")
