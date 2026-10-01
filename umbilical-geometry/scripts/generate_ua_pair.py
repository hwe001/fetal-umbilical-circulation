"""
Generate the paired umbilical arteries (UA), helically wound around the
authoritative UV centreline (geometry/uv_centerline.npy, built by
build_authoritative_centerline.py), using a parallel-transport (Bishop)
frame -- same method as the companion manuscript's Section II-C and the
fetal-atlas-viewer/synthetic_ua pipeline this repo is a corrected sibling
of.

Anatomical parameters (Wan et al. 2025, arXiv:2502.14228, Table I):
  - Coiling pitch: 50mm/turn default (population mean ~0.17-0.20 spirals/cm)
  - Artery-vein wall separation: 1.5mm (Wharton's jelly gap)
  - Angular separation between the two arteries: 40 degrees
  - Left-handed (sinistral) coiling, the predominant real human pattern
  - Tapers to zero offset over the last 20mm of arc length so both
    arteries converge to the vein's own point at the placental insertion
"""
import struct
import os
import argparse
import numpy as np
from scipy.spatial import cKDTree
from ga_growth_model import uv_diameter_mm, ua_diameter_mm

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "..", "geometry")

_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--centerline", default=os.path.join(GEOM, "uv_centerline.npy"))
_cli.add_argument("--cord-stl", default=os.path.join(GEOM, "cord_sheath.stl"))
_cli.add_argument("--target-ga", type=float, default=34.0, help="this specimen's own estimated GA")
_cli.add_argument("--pitch-mm", type=float, default=50.0)
_cli.add_argument("--gap-mm", type=float, default=1.5)
_cli.add_argument("--delta-theta-deg", type=float, default=40.0)
_cli.add_argument("--left-handed", action="store_true", default=True)
_cli.add_argument("--right-handed", dest="left_handed", action="store_false")
_cli.add_argument("--taper-mm", type=float, default=20.0)
_cli.add_argument("--out-a1", default=os.path.join(GEOM, "ua1.stl"))
_cli.add_argument("--out-a2", default=os.path.join(GEOM, "ua2.stl"))
_args = _cli.parse_args()


def read_binary_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    arr = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        off += 12
        arr[i] = np.array(struct.unpack_from("<9f", data, off)).reshape(3, 3)
        off += 38
    return arr.reshape(-1, 3)


centerline = np.load(_args.centerline)
n = len(centerline)
seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
arc_length = np.concatenate([[0], np.cumsum(seg)])
print(f"centreline: {n} points, {arc_length[-1]:.1f}mm total length")

tangents = np.empty_like(centerline)
tangents[1:-1] = centerline[2:] - centerline[:-2]
tangents[0] = centerline[1] - centerline[0]
tangents[-1] = centerline[-1] - centerline[-2]
tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

normals = np.empty_like(centerline)
ref = np.array([0.0, 0.0, 1.0]) if abs(tangents[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
n0 = ref - np.dot(ref, tangents[0]) * tangents[0]
normals[0] = n0 / np.linalg.norm(n0)
for i in range(1, n):
    t_prev, t_cur = tangents[i - 1], tangents[i]
    axis = np.cross(t_prev, t_cur)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        normals[i] = normals[i - 1]
    else:
        axis /= axis_norm
        cos_a = np.clip(np.dot(t_prev, t_cur), -1, 1)
        angle = np.arctan2(axis_norm, cos_a)
        v = normals[i - 1]
        normals[i] = (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
                      + axis * np.dot(axis, v) * (1 - np.cos(angle)))
    normals[i] -= np.dot(normals[i], tangents[i]) * tangents[i]
    normals[i] /= np.linalg.norm(normals[i])
binormals = np.cross(tangents, normals)

uv_radius = uv_diameter_mm(_args.target_ga) / 2
ua_radius = ua_diameter_mm(_args.target_ga) / 2
literature_offset = uv_radius + _args.gap_mm + ua_radius

cord_pts = np.unique(read_binary_stl(_args.cord_stl), axis=0)
cord_tree = cKDTree(cord_pts)
local_cord_dist, _ = cord_tree.query(centerline)
SAFETY_MARGIN_MM = 0.3
FLOOR_MM = ua_radius * 1.2
local_max_offset = np.maximum(local_cord_dist - SAFETY_MARGIN_MM - ua_radius, FLOOR_MM)
offset_dist_profile = np.minimum(literature_offset, local_max_offset)
n_constrained = int((offset_dist_profile < literature_offset - 1e-6).sum())
print(f"target GA {_args.target_ga}w: UV radius={uv_radius:.2f}mm, UA radius={ua_radius:.2f}mm, "
      f"literature offset={literature_offset:.2f}mm; cord-boundary-constrained at "
      f"{n_constrained}/{len(centerline)} points")

handedness = -1.0 if _args.left_handed else 1.0
omega = 2 * np.pi / _args.pitch_mm * handedness
delta_theta = np.radians(_args.delta_theta_deg)
base_angle = -delta_theta / 2

taper = np.clip(arc_length / _args.taper_mm, 0, 1)
effective_offset = offset_dist_profile * taper


def helix_path(angle_offset):
    angles = base_angle + angle_offset + omega * arc_length
    return centerline + effective_offset[:, None] * (np.cos(angles)[:, None] * normals + np.sin(angles)[:, None] * binormals)


path_a1 = helix_path(0.0)
path_a2 = helix_path(delta_theta)


def frustum(pA, rA, pB, rB, sides=10):
    pA = np.array(pA); pB = np.array(pB)
    axis = pB - pA
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return []
    axis /= length
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ringA = [pA + rA * (np.cos(t) * u + np.sin(t) * v) for t in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
    ringB = [pB + rB * (np.cos(t) * u + np.sin(t) * v) for t in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris.append((ringA[i], ringA[j], ringB[i]))
        tris.append((ringA[j], ringB[j], ringB[i]))
    return tris


def normal_of(a, b, c):
    u = np.array(b) - np.array(a); v = np.array(c) - np.array(a)
    nrm = np.cross(u, v); l = np.linalg.norm(nrm)
    return nrm / l if l > 1e-12 else np.zeros(3)


def write_tube_stl(points, radius, out_path):
    tris = []
    for i in range(len(points) - 1):
        tris.extend(frustum(points[i], radius, points[i + 1], radius))
    with open(out_path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            nrm = normal_of(a, b, c)
            f.write(struct.pack("<3f", *nrm))
            f.write(struct.pack("<3f", *a))
            f.write(struct.pack("<3f", *b))
            f.write(struct.pack("<3f", *c))
            f.write(struct.pack("<H", 0))
    print(f"wrote {out_path}: {len(tris)} triangles")


write_tube_stl(path_a1, ua_radius, _args.out_a1)
write_tube_stl(path_a2, ua_radius, _args.out_a2)

seg1 = np.linalg.norm(np.diff(path_a1, axis=0), axis=1)
print(f"UA trunk arc length: {seg1.sum():.1f}mm")

np.save(os.path.join(GEOM, "ua1_path.npy"), path_a1.astype(np.float32))
np.save(os.path.join(GEOM, "ua2_path.npy"), path_a2.astype(np.float32))
np.save(os.path.join(GEOM, "local_cord_dist.npy"), local_cord_dist.astype(np.float32))
np.save(os.path.join(GEOM, "taper.npy"), taper.astype(np.float32))
