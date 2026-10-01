"""Continuation refit at 24 weeks: PI/RI held at cohort targets while the UA trunk length is shortened step by step."""
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'calibration'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import solve_arterial_flow_UA as ua
import fit_ga_targets as fg

OUT = sys.argv[1]
calls = {'n': 0, 't0': time.time()}
_orig = ua.simulate


def logged(*a, **k):
    t = time.time(); r = _orig(*a, **k); calls['n'] += 1
    m = r['metrics']
    print(f"  call {calls['n']:3d} L={k.get('L')} R={k.get('R_total'):.0f} dp={k.get('dp_inlet'):.0f} -> PI {m['PI']:.3f} RI {m['RI']:.3f} PSV {m['PSV_cm_s']:.1f} conv={r['converged']} ({time.time()-t:.1f}s)", flush=True)
    return r


ua.simulate = logged
start = json.load(open(OUT + '/L50.json'))            # the valid fit already obtained at 50 cm
R, dp = start['R_total_dyn_s_cm5'], start['dp_inlet_dyn_cm2']
results = {50.0: start}
for L in (45.0, 40.0, 35.0, 30.0):
    print(f'=== L={L} start R={R:.0f} dp={dp:.0f}', flush=True)
    r = fg.calibrate(24, L, 1.165, 0.705, [R, dp], dx_cm=0.4, max_nfev=15)
    m = r['metrics']
    ok = abs(m['PI'] - 1.165) < 0.03 and abs(m['RI'] - 0.705) < 0.03 and r['periodic_convergence']
    print(f"=== L={L} done: PI {m['PI']:.3f} RI {m['RI']:.3f} PSV {m['PSV_cm_s']:.1f} TAV {m['TAV_cm_s']:.1f} flow {m['total_two_UA_flow_mL_min']:.1f} valid={ok}", flush=True)
    r['valid'] = bool(ok); results[L] = r
    json.dump({str(k): v for k, v in results.items()}, open(OUT + '/continuation.json', 'w'), indent=2)
    if not ok:
        print('target not met here; stopping the continuation so no invalid fit seeds the next step', flush=True); break
    R, dp = r['R_total_dyn_s_cm5'], r['dp_inlet_dyn_cm2']
print('finished', flush=True)
