# -*- coding: utf-8 -*-
"""UV pulsatility robustness across the admissible compliance plateau.

Editor concern #4: the ~80-fold pulsatility reduction (arterial 168% ->
efflux 2.1%) is a headline claim, but compliance is not identifiable from
UA Doppler.  If the filtering effect depended strongly on the (unpinned)
compliances, the claim would be fragile.  This script quantifies it:

* sweep (Cm, Cuv) over the admissible plateau identified in
  fit_pi_filter_matched.py (Cm, Cuv in 1e-5..1.6e-4 cm^5/dyn at
  delta-J <= 1) and over a wider 1e-6..1e-3 bracket;
* at each point, re-simulate the matched benchmark and compute the
  efflux and arterial pulsatility, (max-min)/mean;
* report the WORST case within the plateau, plus corners outside it.

Read-only with respect to the results JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np

SOLVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLVER / "model"))
import placental_pi_filter as pf


def puls(signal):
    core = np.asarray(signal)[:-1]
    return float((core.max() - core.min()) / max(abs(core.mean()), 1e-12))


def main():
    data = json.loads((SOLVER / "results" / "pi_filter_matched.json")
                      .read_text(encoding="utf-8"))
    fp = data["fitted_params"]
    base = pf.default_pi_filter_params()
    params = pf.with_overrides(base, Rs=fp["Rs"], Rout=fp["Rout"])
    bpm = 140.0

    run0 = pf.simulate_pi_filter(bpm, params, init="newton_periodic")
    art0 = puls(run0["Q_UA"])
    eff0 = puls(run0["Q_out"])
    print(f"reference: arterial {art0*100:.1f}%  efflux {eff0*100:.2f}%")

    rows = []
    cm_grid = [1e-6, 1e-5, 3.2e-5, 1e-4, 1.6e-4, 1e-3]
    cuv_grid = [1e-6, 1e-5, 3.2e-5, 1e-4, 1.6e-4, 1e-3]
    for cm in cm_grid:
        for cuv in cuv_grid:
            p = pf.with_overrides(params, Cm=cm, Cuv=cuv)
            try:
                r = pf.simulate_pi_filter(bpm, p, init="newton_periodic")
                m = r["metrics"]
                rows.append({
                    "Cm": cm, "Cuv": cuv,
                    "in_plateau": bool(cm >= 1e-5 and cuv >= 1e-5),
                    "efflux_pulsatility": puls(r["Q_out"]),
                    "arterial_pulsatility": puls(r["Q_UA"]),
                    "RI": m["RI"], "PI": m["PI"],
                    "converged": r["converged"],
                })
            except ValueError:
                rows.append({"Cm": cm, "Cuv": cuv, "in_plateau":
                             bool(cm >= 1e-5 and cuv >= 1e-5),
                             "diverged": True})

    ok = [r for r in rows if r.get("converged")]
    plateau = [r for r in ok if r["in_plateau"]]
    outside = [r for r in ok if not r["in_plateau"]]
    summary = {
        "arterial_pulsatility_reference": art0,
        "efflux_pulsatility_reference": eff0,
        "plateau_points": len(plateau),
        "efflux_pulsatility_max_in_plateau":
            max(r["efflux_pulsatility"] for r in plateau),
        "efflux_pulsatility_min_in_plateau":
            min(r["efflux_pulsatility"] for r in plateau),
        "worst_point_in_plateau": max(plateau,
                                      key=lambda r: r["efflux_pulsatility"]),
        "outside_points": [{k: r[k] for k in ("Cm", "Cuv",
                            "efflux_pulsatility") if k in r}
                           for r in outside],
        "grid": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "grid"},
                     indent=1, default=str))
    (SOLVER / "results" / "uv_pulsatility_robustness.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("wrote", SOLVER / "results" / "uv_pulsatility_robustness.json")


if __name__ == "__main__":
    main()
