"""Peak-to-mean velocity correction (Womersley) for the 24-week model at each valid trunk length, plus measured peak systolic velocity."""
from pathlib import Path
import sys, json, os
import numpy as np, pandas as pd
from scipy.special import jv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import solve_arterial_flow_UA as ua
from gestational_geometry import ua_radius_cm

S = sys.argv[1]
fits = {}
for L in ('72.2', '60', '50'):
    r = json.load(open(f'{S}/L{L}.json')); fits[float(L)] = (r['R_total_dyn_s_cm5'], r['dp_inlet_dyn_cm2'])
cont = json.load(open(f'{S}/continuation.json'))
for L in ('45.0', '40.0'):
    r = cont[L]; assert r['valid']; fits[float(L)] = (r['R_total_dyn_s_cm5'], r['dp_inlet_dyn_cm2'])
NU = ua.NU


def womersley_ratio(Qn, n, omega, a):
    """centreline velocity / mean velocity for one harmonic of flow (complex amplitude Qn, harmonic n)."""
    al = a * np.sqrt(n * omega / NU); z = 1j ** 1.5 * al
    return (1 - 1 / jv(0, z)) / (1 - 2 * jv(1, z) / (z * jv(0, z))), al


rows = []
for L, (R, dp) in sorted(fits.items(), reverse=True):
    res = ua.simulate(bpm=140.0, r0=ua_radius_cm(24, allow_extrapolation=False), R_total=R, C_total=4e-5, p0=30000, dp_inlet=dp, L=L,
                      M=int(round(L / 0.4)) + 1, n_cycles=15, max_cycles=80, index_definition='minimum')
    t, u, A = res['t'], res['u_outlet'], res['A_outlet']; T = res['T']; m = t >= t[-1] - T
    tt = t[m] - t[m][0]; uu = u[m]; a = float(np.sqrt(np.mean(A[m]) / np.pi))
    N = 256; tg = np.linspace(0, T, N, endpoint=False); ug = np.interp(tg, tt, uu)       # one cycle on a uniform grid
    F = np.fft.rfft(ug) / N; omega = 2 * np.pi / T
    uc = np.full(N, F[0].real)                                                          # mean velocity: parabolic limit does not apply to the DC term
    uc = np.full(N, 2 * F[0].real, dtype=float)                                         # steady component: Poiseuille, centreline = 2 x mean
    als = []
    for n in range(1, len(F)):
        if abs(F[n]) < 1e-3 * abs(F[1]) and n > 12: break
        ratio, al = womersley_ratio(F[n], n, omega, a); als.append(al)
        uc += 2 * (F[n] * ratio * np.exp(1j * n * omega * tg)).real
    mean_peak, cl_peak = float(ug.max()), float(uc.max())
    def idx(v): return float((v.max() - v.min()) / v.mean()), float((v.max() - v.min()) / v.max())
    mPI, mRI = idx(ug); cPI, cRI = idx(uc)
    rows.append(dict(L_cm=L, R=R, dp=dp, PI=res['metrics']['PI'], RI=res['metrics']['RI'], radius_cm=a, womersley_alpha_1=als[0],
                     mean_PSV=mean_peak, centreline_PSV=cl_peak, factor=cl_peak / mean_peak, flow_mL_min=res['metrics']['total_two_UA_flow_mL_min'], mean_PI=mPI, mean_RI=mRI, centreline_PI=cPI, centreline_RI=cRI))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{S}/length_sensitivity_24wk.csv', index=False)

# measured peak systolic velocity, 23-24 weeks, umbilical artery panels; one value per patient folder (median) to avoid pseudoreplication
o = pd.read_csv(os.path.join(os.path.dirname(S), 'ocr_panels_anonymised.csv'))
u = o[(o.vessel == 'umb') & o.ga.between(23, 24) & o.PS.notna() & o.PI.between(0.3, 3)].copy()
u = u.drop_duplicates(subset=['PI', 'RI', 'SD', 'PS', 'ED', 'HR'])
u['PSabs'] = u.PS.abs()
pf = u.groupby('pid').PSabs.median()
out = dict(n_frames=int(len(u)), n_women=int(u.pid.nunique()), frame_median=float(u.PSabs.median()), frame_q1=float(u.PSabs.quantile(.25)), frame_q3=float(u.PSabs.quantile(.75)),
           woman_median=float(pf.median()), woman_q1=float(pf.quantile(.25)), woman_q3=float(pf.quantile(.75)), woman_min=float(pf.min()), woman_max=float(pf.max()))
json.dump(out, open(f'{S}/measured_PSV_24wk.json', 'w'), indent=2); print(out)
