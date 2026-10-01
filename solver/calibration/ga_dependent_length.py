"""GA-dependent UA path length: L_UA(GA) = (61.78->72.2 helix factor) x pooled population-median cord length (Linde et al. 2018, S1 Table).
Recalibrate R_total and dp_inlet to the cohort PI/RI at that GA; then report mean and Womersley-centreline peak velocity and flow."""
from pathlib import Path
import sys, json, csv, time, statistics as st
import numpy as np
from scipy.special import jv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'calibration'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import solve_arterial_flow_UA as ua
import fit_ga_targets as fg
from gestational_geometry import ua_radius_cm

GA = int(sys.argv[1]); OUT = sys.argv[2]
TARGETS = {19: (1.295, 0.74), 23: (1.19, 0.71), 24: (1.165, 0.705), 25: (1.14, 0.70), 29: (0.96, 0.63)}
SEED72 = {19: (17238, 40978), 23: (7864, 16004), 24: (4870, 13428), 25: (4684, 11085), 29: (1199, 6224)}   # existing 72.2 cm fits
HELIX = 72.2 / 61.78
rows = list(csv.DictReader(open(str(Path(__file__).resolve().parents[1] / 'data' / 'norway_cord_length_percentiles.csv'))))
cord = st.mean(float(r['p50_cm']) for r in rows if int(r['ga_weeks']) == GA)
L = round(cord * HELIX, 1)
pi_t, ri_t = TARGETS[GA]; R72, dp72 = SEED72[GA]
R0, dp0 = R72 * (L / 72.2) ** 1.8, dp72 * (L / 72.2)
if GA == 24: R0, dp0 = 1654.0 * (L / 40.0) ** 1.8, 7241.0 * (L / 40.0)
print(f'GA {GA}: pooled median cord {cord:.2f} cm -> UA path {L} cm; seed R={R0:.0f} dp={dp0:.0f}; targets PI {pi_t} RI {ri_t}', flush=True)

calls = {'n': 0}
_orig = ua.simulate


def logged(*a, **k):
    t = time.time(); r = _orig(*a, **k); calls['n'] += 1; m = r['metrics']
    print(f"  GA{GA} call {calls['n']:3d} R={k.get('R_total'):.0f} dp={k.get('dp_inlet'):.0f} -> PI {m['PI']:.3f} RI {m['RI']:.3f} PSV {m['PSV_cm_s']:.1f} conv={r['converged']} ({time.time()-t:.0f}s)", flush=True)
    return r


ua.simulate = logged
valid = lambda r: abs(r['metrics']['PI'] - pi_t) < 0.03 and abs(r['metrics']['RI'] - ri_t) < 0.03 and r['periodic_convergence']
res = fg.calibrate(GA, L, pi_t, ri_t, [R0, dp0], dx_cm=0.4, max_nfev=25)
if not valid(res):
    print(f'GA{GA}: first attempt not on target (PI {res["metrics"]["PI"]:.3f}); restarting from its end point', flush=True)
    res = fg.calibrate(GA, L, pi_t, ri_t, [res['R_total_dyn_s_cm5'], res['dp_inlet_dyn_cm2']], dx_cm=0.4, max_nfev=25)
ok = valid(res); m = res['metrics']

# velocity: mean and Womersley centreline peak, from the fitted model
R, dp = res['R_total_dyn_s_cm5'], res['dp_inlet_dyn_cm2']
sim = _orig(bpm=140.0, r0=ua_radius_cm(GA, allow_extrapolation=False), R_total=R, C_total=4e-5, p0=30000, dp_inlet=dp, L=L, M=int(round(L / 0.4)) + 1, n_cycles=15, max_cycles=80, index_definition='minimum')
t, u, A = sim['t'], sim['u_outlet'], sim['A_outlet']; T = sim['T']; mk = t >= t[-1] - T
a = float(np.sqrt(np.mean(A[mk]) / np.pi)); N = 256; tg = np.linspace(0, T, N, endpoint=False); ug = np.interp(tg, t[mk] - t[mk][0], u[mk])
F = np.fft.rfft(ug) / N; om = 2 * np.pi / T; uc = np.full(N, 2 * F[0].real)
for n in range(1, len(F)):
    if abs(F[n]) < 1e-3 * abs(F[1]) and n > 12: break
    z = 1j ** 1.5 * a * np.sqrt(n * om / ua.NU); ratio = (1 - 1 / jv(0, z)) / (1 - 2 * jv(1, z) / (z * jv(0, z)))
    uc += 2 * (F[n] * ratio * np.exp(1j * n * om * tg)).real
out = dict(ga=GA, cord_median_cm=cord, L_UA_cm=L, R_total=R, dp_inlet=dp, PI=m['PI'], RI=m['RI'], PI_target=pi_t, RI_target=ri_t, mean_PSV=float(ug.max()), centreline_PSV=float(uc.max()),
           flow_mL_min=m['total_two_UA_flow_mL_min'], converged=bool(res['periodic_convergence']), on_target=bool(ok), radius_cm=a, calls=calls['n'])
json.dump(out, open(OUT, 'w'), indent=2); print('RESULT', out, flush=True)
