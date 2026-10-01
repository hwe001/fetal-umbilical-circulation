# Fetal Anatomy Atlas

An MR-digitized fetal anatomy atlas (a single fetus, ~23 segmented anatomical structures)
as a self-contained three.js viewer, with the geometry kept in a form that's directly
reusable for physiological simulation work rather than only for display.

Part of the companion repository for the manuscript "Anatomy-Informed Modeling of the Fetal Umbilical Circulation Across Gestation" (Tang, Peng and Ho, in preparation). The umbilical vessel geometry and flow viewers are in [`../umbilical-geometry`](../umbilical-geometry); the 1-D solver and calibration code are in [`../solver`](../solver). This folder covers only the whole-body anatomical atlas the umbilical geometry was extracted from.

**Live viewer**: https://hwe001.github.io/fetal-umbilical-circulation/atlas/viewers/fetus_atlas_viewer.html

## Why rebuild it

The original page uses the legacy three.js JSON model format (`formatVersion: 3`,
"Exported from LibZinc") - a bitmask-encoded face format that modern three.js can no
longer load directly (its `JSONLoader` was removed years ago). Rebuilding it required
writing a real parser for that format (see `scripts/parse_fetus_atlas.py`), which also
made it straightforward to export each anatomical part as a standard STL alongside the
viewer geometry - useful for anything downstream (voxelization, CCO vessel growth,
Poiseuille flow solves), not just rendering.

## Structure

```
scripts/
  parse_fetus_atlas.py     # legacy three.js JSON -> per-part STL + combined viewer binary
  assemble_fetus_viewer.py # inlines three.js + geometry + manifest into one HTML file
data_in/
  source_json/             # the 26 original BasicMesh_*.json + metadata.json (unmodified)
data_out/
  stl/                     # one STL per original mesh part (e.g. BasicMesh_12.stl = portal vein)
  fetus_manifest.json      # part number -> organ group name -> vertex/triangle counts
  fetus_atlas_geometry.bin # combined position+normal binary for all 23 groups (no color baked in)
viewers/
  fetus_atlas_viewer.html  # the actual viewer - fully self-contained, open directly in a browser
```

## The 23 anatomical structures

Grouped by system (see `ORGAN_STYLE` in `assemble_fetus_viewer.py` for the exact palette):

- **Cardiovascular:** aorta, heart, portal (vein), svc (superior vena cava)
- **Umbilical:** cord, placenta, uv (umbilical vein)
- **Abdominal:** bladder, kidney, liver, sex_organ_m
- **Thoracic:** lung
- **Neuro:** brain_left, brain_right, cerebellum, eye
- **Maternal:** womb
- **Body shell:** arm_left, arm_right, head, leg_left, leg_right, torso

Two of the original 26 mesh parts share a `GroupName` with another part at a much lower
resolution (`liver` and `torso` each have two parts, both kept and merged; `head` has two
parts but the higher-res one, `BasicMesh_23.json` at 45,216 triangles vs. `BasicMesh_9.json`
at 11,448, was excluded from the merged viewer binary only - keeping both pushed the
self-contained HTML too large. `BasicMesh_23.stl` is still written
to `data_out/stl/` for anyone who wants the higher-res head separately).

## Design choices, since this is meant to be reused later

- **Geometry has no color or scale baked in.** `fetus_atlas_geometry.bin` is pure
  position + source vertex normal; organ color is assigned per-`GroupName` in
  `assemble_fetus_viewer.py`, so re-theming the viewer never requires rebuilding the binary.
- **Real coordinates, untouched.** No re-centering or rescaling happens before STL export
  (only the *viewer* recenters, for camera framing) - so `data_out/stl/BasicMesh_12.stl`
  (portal vein) and `BasicMesh_6.stl`/`BasicMesh_10.stl` (liver) sit in the same coordinate
  frame as every other part, ready to feed into a CCO growth or voxelization pipeline the
  same way the real donor portal-vein cohort was used elsewhere.
- **Per-part STL, not just per-organ.** Where an organ has more than one source part
  (liver, torso), each part keeps its own STL rather than being pre-merged, in case a
  future use only wants one of them.

## Reproducing

```bash
python scripts/parse_fetus_atlas.py \
    --source-dir data_in/source_json \
    --stl-out-dir data_out/stl \
    --manifest-out data_out/fetus_manifest.json \
    --geometry-out data_out/fetus_atlas_geometry.bin

python scripts/assemble_fetus_viewer.py \
    --three-js path/to/three.min.js \
    --geometry data_out/fetus_atlas_geometry.bin \
    --manifest data_out/fetus_manifest.json \
    --out viewers/fetus_atlas_viewer.html
```

`three.min.js` (r160) isn't vendored here - grab any recent three.js UMD build.

## Provenance

Source data: a single fetus, MR-imaged in utero and manually segmented into 23
anatomical structures in 2018-2019. No patient identifiers are present in any file here.

## Contact

Questions about this atlas: [Dr Harvey Ho](mailto:harvey.nz@gmail.com). (The live
viewer builds the address at runtime.)

## License

MIT.
