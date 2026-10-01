# Fetal umbilical circulation: atlas, geometry, viewers and solver

Companion repository for the manuscript **Anatomy-Informed Modeling of the Fetal Umbilical Circulation Across Gestation** (Tang, Peng and Ho; in preparation). It collects everything behind the paper in one place, in three folders.

| Folder | What it holds | Live viewer |
|---|---|---|
| [`atlas/`](atlas) | MR-digitised fetal anatomy atlas of a single fetus: 26 source meshes forming 23 named structures, one STL per part, parsing and assembly scripts | [atlas viewer](https://hwe001.github.io/fetal-umbilical-circulation/atlas/viewers/fetus_atlas_viewer.html) |
| [`umbilical-geometry/`](umbilical-geometry) | Digitised cord and umbilical-vein centreline, synthetic paired umbilical arteries, cord sheath and placenta; geometry-generation scripts; the two viewers | [geometry viewer](https://hwe001.github.io/fetal-umbilical-circulation/umbilical-geometry/viewer/), [cardiac-cycle flow viewer](https://hwe001.github.io/fetal-umbilical-circulation/umbilical-geometry/viewer/flow.html) |
| [`solver/`](solver) | 1-D compliant pulsatile umbilical-artery solver (Lax-Wendroff, RCR Windkessel), reduced UV return model, calibration to PI/RI, sensitivity, path-length and Womersley analyses, stored results, tests | none |

## How the pieces connect

atlas -> the specimen's UV, cord and placenta surfaces -> `umbilical-geometry/` (UV centreline, synthetic UA pair, growth curves) -> UA length and radius used by `solver/` -> pressure and flow fields exported to `umbilical-geometry/viewer/flow_data.js` (`solver/figures/export_flow_data.py`).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest solver/tests -q          # 22 tests, about 1 min
python solver/reproduce_check.py 25       # re-runs the 25-week production solve and compares with the stored results
```

The viewers are self-contained HTML: open the files directly, or use the live links above. Each folder has its own README with provenance and reproduction steps; see `solver/README.md` for the model and the saved results.

## Data and privacy

- All geometry comes from one fetus imaged in utero and segmented in 2018-2019; no patient identifiers are present.
- The Doppler calibration targets are cohort medians (aggregates, in `solver/data/`). Patient-level Doppler records are **not** included. `solver/calibration/pi_information_analysis.py` needs the cohort workbook (path via the `UA_COHORT_XLSX` environment variable); its aggregate output is stored in `solver/results/`.
- Cord-length percentiles are transcribed from Linde et al., PLOS ONE 2018 (S1 Table, CC BY); see `solver/data/norway_cord_length_percentiles_SOURCE.txt`.

## Status

The repository accompanies a manuscript under preparation. A Zenodo DOI will be added after acceptance. The 23-week gestation-dependent cord-length fit did not converge, so `solver/results/ga_dependent_length/GA23.json` is marked `on_target: false` (PI 1.30 vs target 1.19) and is not used in the paper.

## Licence

MIT (`LICENSE`).
