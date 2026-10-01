"""Quick reproducibility check (about 1-2 min): re-run the production forward solve at the stored
Table I parameters for one gestational age and compare PI and RI with results/full_recalibration.json.
Usage:  python reproduce_check.py [GA_weeks]   (default 25; choices 19, 23, 25, 29)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "calibration"))
from run_full_recalibration import production_check

ga = int(sys.argv[1]) if len(sys.argv) > 1 else 25
stored = json.loads((ROOT / "results" / "full_recalibration.json").read_text(encoding="utf-8"))
fit = next(r for r in stored["table_I"] if r["ga_weeks"] == ga)
new = production_check(fit, include_uv=False)
old = fit["production"]["metrics"]
ok = True
for k in ("PI", "RI", "S_D", "total_two_UA_flow_mL_min"):
    a, b = new["metrics"][k], old[k]
    good = abs(a - b) <= 1e-3 * max(1.0, abs(b))
    ok &= good
    print(f"GA {ga} {k:>26}: re-run {a:.5f}  stored {b:.5f}  {'OK' if good else 'DIFFERENT'}")
print("converged:", new["converged"], "| fallback fraction:", new["fallback_fraction"])
sys.exit(0 if ok and new["converged"] else 1)
