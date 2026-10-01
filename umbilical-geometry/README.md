# Fetal Umbilical Geometry

Real, patient-derived 3D geometry of the fetal umbilical vein (UV), a
synthetic umbilical artery (UA) pair generated from it, and the surrounding
cord sheath and placenta — companion data and code for the manuscript **"Anatomy-Informed Modeling of the Fetal Umbilical Circulation Across Gestation"**

**Live viewers:** [geometry viewer](https://hwe001.github.io/fetal-umbilical-circulation/umbilical-geometry/viewer/) and [cardiac-cycle flow viewer](https://hwe001.github.io/fetal-umbilical-circulation/umbilical-geometry/viewer/flow.html). Both are fully self-contained, so you can also clone the repository and open `viewer/index.html` and `viewer/flow.html` locally (or serve `viewer/` with any static file server); there is no build step. Drag the **gestational age** slider to see
the vein, arteries, and cord grow along their literature-anchored growth
curves; drag **coiling** to see the arteries wind tighter or looser around
the vein while staying inside the real digitized cord boundary throughout.

This folder sits beside [`../atlas`](../atlas) (the 23-structure digitised fetal atlas this specimen's UV, cord and placenta are drawn from) and [`../solver`](../solver) (the 1-D pulsatile UA solver, calibration and results shown in the flow viewer). It focuses on the umbilical geometry pipeline at full fidelity.

## Why this repo exists

The umbilical arteries were too small to resolve at the original MRI
segmentation's resolution, so they're generated synthetically — helically
wound around the vein's real digitized centreline (parallel-transport
frame, real cord-boundary constraint, tapered convergence to a shared
placental insertion point). Getting that centreline right matters: an
earlier surface-derived extraction undershot the vein's true arc length by
~17%, traced back here to its source and fixed by going directly to the
original hand-digitized CMISS source mesh rather than re-deriving the
centreline from an exported surface a second time.

## What's here

```
source_mesh/
  cord_offset.exnode / .exelem   # the actual digitization record: a CMISS 1D
                                  # cubic Hermite mesh (32 nodes, 31 elements),
                                  # nodes placed by hand on the source MRI at
                                  # higher density where the cord's local
                                  # curvature was greater -- not a surface
                                  # re-derivation
scripts/
  parse_cmiss_cord.py            # parses the source mesh, computes arc length
                                  # via Gauss-Legendre quadrature -> 617.7mm
  build_authoritative_centerline.py  # orders the mesh into a path via element
                                  # connectivity, densely samples the Hermite
                                  # curve -> geometry/uv_centerline.npy
  generate_ua_pair.py             # helical UA pair around the UV centreline
                                  # (parallel-transport frame, real cord-sheath
                                  # boundary constraint, GA-dependent radius)
  export_scene_data.py           # packages everything for viewer/
  ga_growth_model.py              # UV/UA diameter-vs-GA growth curves
geometry/
  cord_sheath.stl, placenta.stl  # real digitized surfaces (from ../atlas)
  uv_centerline.npy              # the authoritative UV centreline (617.7mm)
  ua1.stl / ua2.stl, ua1_path.npy / ua2_path.npy   # the synthetic UA pair
viewer/
  index.html                     # geometry viewer: GA and coiling sliders (self-contained three.js)
  flow.html, flow_app.js,        # cardiac-cycle pressure/flow viewer on the UA/UV reconstruction;
  flow_data.js                   # flow_data.js is the 25-week production fit exported from ../solver
  scene_data.js, app.js          # packaged geometry and geometry-viewer code
```

## Reproducing

```bash
pip install numpy scipy
cd scripts
python parse_cmiss_cord.py                # -> 617.8mm
python build_authoritative_centerline.py   # -> geometry/uv_centerline.npy
python generate_ua_pair.py                 # -> geometry/ua1.stl, ua2.stl
python export_scene_data.py                # -> viewer/scene_data.js
```

## Validation

Two independent numbers from the source mesh match the manuscript's
reported values, confirming this is the correct source (not a different
digitization or a different vessel):

| Quantity | This repo | Manuscript |
|---|---|---|
| UV/cord arc length | 617.7 mm | 617.8 mm |
| Fetal-to-placental straight-line distance | 80.0 mm | 80.0 mm |

## Provenance and de-identification

Source data: a single fetus, MR imaged in utero at an estimated 34 weeks
gestational age, manually segmented in 2018-2019. All patient identifiers
were removed prior to any analysis; only anatomical geometry is included
here. No accession numbers, names, or scan dates are present in any file
in this repo. Consistent with the parent atlas
([`../atlas`](../atlas)), public since
2019.

## Citation

See the repository-level `CITATION.cff`. A Zenodo DOI will be added after the manuscript is accepted.

## License

MIT (see the repository-level [LICENSE](../LICENSE)).

## Flow viewer

`viewer/flow.html` animates the computed pressure and flow along the UA and the reduced UV return over one cardiac cycle. The fields come from `../solver` (production fit at 25 weeks, Table I; the UV quantity is the total placental efflux). The UV is the reduced one-way return model of the manuscript, not a coupled venous PDE solve. `viewer/flow_data.js` is regenerated by `../solver/figures/export_flow_data.py`.
