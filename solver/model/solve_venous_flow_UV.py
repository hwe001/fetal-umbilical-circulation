"""Reduced 1D rigid-tube umbilical-vein return model.

For a uniform, nearly central UV, integrating the 1D momentum balance along
the vessel gives pressure drop = viscous resistance*flow + inertance*dQ/dt.
This retains distributed length, radius, viscous and inertial effects without
using inappropriate 3D CFD for the long cord vessel.
"""

import numpy as np

from gestational_geometry import UV_TRUNK_LENGTH_CM, circular_area_cm2, uv_radius_cm

RHO = 1.05
MU = 0.035


def simulate_uv_return(t, total_ua_flow_mL_s, ga, fetal_venous_pressure_dyn_cm2=0.0,
                       length_cm=UV_TRUNK_LENGTH_CM, radius_scale=1.0,
                       allow_geometry_extrapolation=False):
    t = np.asarray(t, dtype=float)
    q = np.asarray(total_ua_flow_mL_s, dtype=float)
    if t.ndim != 1 or q.shape != t.shape or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("t and flow must be equal-length arrays with increasing time")
    radius = uv_radius_cm(ga, allow_geometry_extrapolation) * radius_scale
    area = circular_area_cm2(radius)
    resistance = 8.0 * MU * length_cm / (np.pi * radius**4)
    inertance = RHO * length_cm / area
    dqdt = np.gradient(q, t, edge_order=2)
    pressure_drop = resistance * q + inertance * dqdt
    return {
        "t": t,
        "flow_mL_s": q,
        "velocity_cm_s": q / area,
        "placental_venous_pressure_dyn_cm2": fetal_venous_pressure_dyn_cm2 + pressure_drop,
        "pressure_drop_dyn_cm2": pressure_drop,
        "mean_flow_mL_min": float(np.trapezoid(q, t) / (t[-1] - t[0]) * 60.0),
        "mean_velocity_cm_s": float(np.trapezoid(q / area, t) / (t[-1] - t[0])),
        "radius_cm": float(radius), "length_cm": float(length_cm),
        "resistance_dyn_s_cm5": float(resistance),
        "inertance_dyn_s2_cm5": float(inertance),
    }
