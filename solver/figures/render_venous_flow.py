from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'calibration'))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solve_arterial_flow_UA as ua
from gestational_geometry import ua_radius_cm, uv_radius_cm

d = json.load(open(str(Path(__file__).resolve().parents[1] / "results" / "full_recalibration.json")))
r25 = next(r for r in d["table_II"] if r["ga_weeks"] == 25)
R_total = r25["R_total_dyn_s_cm5"]; dp = r25["dp_inlet_dyn_cm2"]; p0 = r25["p0_dyn_cm2"]
r0 = ua_radius_cm(25, allow_extrapolation=False)
L = 72.2; dx = d["geometry"]["production_dx_cm"]; M = int(round(L / dx)) + 1
result = ua.simulate(bpm=140.0, r0=r0, R_total=R_total, C_total=4e-5, p0=p0, dp_inlet=dp,
                      L=L, M=M, n_cycles=15, max_cycles=None, index_definition="minimum")
u = result["u_outlet"]; t = result["t"]; T = result["T"]
A0 = result["A0"]; Eh_r0 = result["Eh_r0"]
A_out = result["A_outlet"]; q_out = result["q_outlet"]
R1 = 0.15 * R_total; R2 = R_total - R1
p_out = ua.tube_law(A_out, A0, Eh_r0, p0)
Pwk = p_out - R1 * q_out
# total UV inflow = placental efflux from BOTH arteries (symmetric pair):
# twice the single-UA distal-resistor flow, NOT the arterial inflow q_b
# (which bypasses placental storage; cycle means coincide, waveforms do not)
q_ven = 2.0 * Pwk / R2
uv_r = uv_radius_cm(25, allow_extrapolation=False)
u_uv = q_ven / (np.pi * uv_r ** 2)

u_norm = u / u.max()
u_uv_norm = u_uv / u.max()
t_ms = (t - t[0]) * 1000

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(t_ms, u_norm, color="#c0392b", linewidth=1.8, label="UA (arterial)")
ax.plot(t_ms, u_uv_norm, color="#2e6f9e", linewidth=1.8, label="UV (Windkessel-implied)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Velocity, normalized to arterial peak")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = str(Path(__file__).resolve().parent / "fig_venous_flow.png")
plt.savefig(out, dpi=250)
print("saved", out)
print("peak-to-trough pct of arterial peak:", 100 * (u_uv_norm.max() - u_uv_norm.min()))
