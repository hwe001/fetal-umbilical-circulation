"""
Parse the legacy three.js JSON model format (formatVersion 3, "Exported from
LibZinc") used by hwe001.github.io/fetus/BasicMesh_*.json, and produce:
  1. one STL per original mesh part (for future CCO/voxelization/simulation
     reuse - e.g. BasicMesh_12.stl is the real fetal portal vein surface),
     with a manifest mapping part number -> organ group name -> real vertex
     count, so a future script can pick "liver" + "portal" directly.
  2. one combined binary geometry blob for the three.js viewer: per-organ
     grouped triangles (position + source vertex normal for smooth shading),
     no color baked in (color is assigned per-organ-name in the viewer JS,
     so re-theming doesn't require rebuilding this file).

Face format (legacy three.js JSONLoader bitmask, NOT something modern
three.js can load - this is why we parse it ourselves):
  type byte bits: 0=isQuad 1=hasMaterial 2=hasFaceUv 3=hasFaceVertexUv
                  4=hasFaceNormal 5=hasFaceVertexNormal 6=hasFaceColor
                  7=hasFaceVertexColor
  followed by: nVerts vertex indices, [material idx], [faceUv idx],
  [nVerts faceVertexUv idx], [faceNormal idx], [nVerts faceVertexNormal idx],
  [faceColor idx], [nVerts faceVertexColor idx]  (nVerts = 4 if isQuad else 3)
"""
import json, struct, math, argparse
from pathlib import Path

_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--source-dir", required=True, help="Directory with metadata.json + BasicMesh_*.json")
_cli.add_argument("--stl-out-dir", required=True, help="Directory to write per-part STL files")
_cli.add_argument("--manifest-out", required=True, help="Path for the part->group manifest JSON")
_cli.add_argument("--geometry-out", required=True, help="Path for the combined viewer geometry .bin")
_args = _cli.parse_args()

SP = Path(_args.source_dir)
STL_OUT = Path(_args.stl_out_dir); STL_OUT.mkdir(parents=True, exist_ok=True)
metadata = json.load(open(SP / "metadata.json"))

def parse_mesh(path):
    d = json.load(open(path))
    verts = d["vertices"]
    norms = d.get("normals")
    faces = d["faces"]
    n_verts = len(verts) // 3
    def V(i): return (verts[3*i], verts[3*i+1], verts[3*i+2])
    def N(i): return (norms[3*i], norms[3*i+1], norms[3*i+2])

    tris = []  # list of (posA,posB,posC, nA,nB,nC)
    i = 0
    n_quads = 0
    while i < len(faces):
        t = faces[i]; i += 1
        is_quad = t & 1
        has_material = t & 2
        has_face_uv = t & 4
        has_face_vertex_uv = t & 8
        has_face_normal = t & 16
        has_face_vertex_normal = t & 32
        has_face_color = t & 64
        has_face_vertex_color = t & 128
        nv = 4 if is_quad else 3

        vidx = faces[i:i+nv]; i += nv
        if has_material: i += 1
        if has_face_uv: i += 1
        if has_face_vertex_uv: i += nv
        if has_face_normal: i += 1
        nidx = None
        if has_face_vertex_normal:
            nidx = faces[i:i+nv]; i += nv
        if has_face_color: i += 1
        if has_face_vertex_color: i += nv

        if nidx is None:
            nidx = vidx  # fall back to vertex index if no separate normal index

        pts = [V(v) for v in vidx]
        nrm = [N(n) for n in nidx] if norms else None

        if is_quad:
            n_quads += 1
            # split quad (0,1,2,3) into triangles (0,1,2) and (0,2,3)
            tri_index_pairs = [(0,1,2), (0,2,3)]
        else:
            tri_index_pairs = [(0,1,2)]

        for a,b,c in tri_index_pairs:
            pa,pb,pc = pts[a], pts[b], pts[c]
            if nrm:
                na,nb,nc = nrm[a], nrm[b], nrm[c]
            else:
                # compute a flat face normal
                ux,uy,uz = pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]
                vx,vy,vz = pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]
                fx,fy,fz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
                l = math.sqrt(fx*fx+fy*fy+fz*fz) or 1.0
                na=nb=nc=(fx/l, fy/l, fz/l)
            tris.append((pa,pb,pc,na,nb,nc))
    return n_verts, tris, n_quads

def face_normal(pa, pb, pc):
    ux,uy,uz = pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]
    vx,vy,vz = pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]
    fx,fy,fz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
    l = math.sqrt(fx*fx+fy*fy+fz*fz)
    return (fx/l, fy/l, fz/l) if l > 1e-12 else (0.0,0.0,0.0)

def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\x00"*80)
        f.write(struct.pack("<I", len(tris)))
        for pa,pb,pc,na,nb,nc in tris:
            fn = face_normal(pa,pb,pc)
            f.write(struct.pack("<3f", *fn))
            f.write(struct.pack("<3f", *pa))
            f.write(struct.pack("<3f", *pb))
            f.write(struct.pack("<3f", *pc))
            f.write(struct.pack("<H", 0))

# BasicMesh_23 (head, 28574 verts / 45216 tris) is a much higher-res duplicate
# of BasicMesh_9 (head, 7791 verts / 11448 tris) - both map to GroupName
# "head". Keeping both alone pushed the combined viewer past the 16MB
# artifact limit, and 23 alone is ~24% of all triangles in the atlas for a
# part the viewer already renders via BasicMesh_9. Excluded from the merged
# viewer binary only - its STL is still written to disk for future reuse.
EXCLUDE_FROM_VIEWER = {23}

manifest = []
all_tris_by_group = {}
global_min = [1e18,1e18,1e18]; global_max = [-1e18,-1e18,-1e18]

for entry in metadata:
    part_id = int(entry["URL"].split("_")[1].split(".")[0])
    group = entry["GroupName"]
    src = SP / entry["URL"]
    n_verts, tris, n_quads = parse_mesh(src)
    stl_path = STL_OUT / f"BasicMesh_{part_id}.stl"
    write_stl(stl_path, tris)
    for pa,pb,pc,na,nb,nc in tris:
        for p in (pa,pb,pc):
            for k in range(3):
                global_min[k] = min(global_min[k], p[k])
                global_max[k] = max(global_max[k], p[k])
    if part_id not in EXCLUDE_FROM_VIEWER:
        all_tris_by_group.setdefault(group, []).extend(tris)
    manifest.append({
        "part": part_id, "group": group, "n_vertices": n_verts,
        "n_triangles": len(tris), "n_quads_in_source": n_quads,
        "stl": stl_path.name,
    })
    print(f"BasicMesh_{part_id}.json ({group}): {n_verts} verts, {len(tris)} tris ({n_quads} quads split) -> {stl_path.name}")

json.dump(manifest, open(_args.manifest_out, "w"), indent=2)
print(f"\nBounding box: {global_min} to {global_max}")
print(f"Groups: {sorted(all_tris_by_group.keys())}")

# --- combined binary geometry for the viewer: per-group triangles, position + vertex normal only ---
OUT_BIN = Path(_args.geometry_out)
with open(OUT_BIN, "wb") as f:
    groups_sorted = sorted(all_tris_by_group.keys())
    f.write(struct.pack("<I", len(groups_sorted)))
    total_tris = 0
    for group in groups_sorted:
        tris = all_tris_by_group[group]
        name_b = group.encode("utf-8")
        f.write(struct.pack("<I", len(name_b)))
        f.write(name_b)
        f.write(struct.pack("<I", len(tris)))
        for pa,pb,pc,na,nb,nc in tris:
            f.write(struct.pack("<3f", *pa))
            f.write(struct.pack("<3f", *na))
            f.write(struct.pack("<3f", *pb))
            f.write(struct.pack("<3f", *nb))
            f.write(struct.pack("<3f", *pc))
            f.write(struct.pack("<3f", *nc))
        total_tris += len(tris)
    print(f"\nWrote {OUT_BIN.name}: {len(groups_sorted)} groups, {total_tris} total triangles")
