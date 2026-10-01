"""
Export everything the interactive viewer needs: cord sheath as an indexed,
radially-decomposed mesh (for live GA rescaling), the static placenta
surface, the UV centreline, and the raw ingredients (local cord-boundary
distance, taper profile, growth-curve anchors) needed to regenerate the UA
pair live in JS as the user moves the GA / coiling-index sliders --
rather than shipping one fixed precomputed UA pair.
"""
import struct
import base64
import json
import os
import numpy as np
from ga_growth_model import UV_GA, UV_DIAM_MM, UA_GA, UA_DIAM_MM

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "..", "geometry")
VIEWER = os.path.join(HERE, "..", "viewer")
REFERENCE_GA = 34.0  # this specimen's own estimated GA; cord scale = 1 here


def read_binary_stl_tris(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    tris = np.empty((n, 3, 3), dtype=np.float32)
    off = 84
    for i in range(n):
        off += 12
        tris[i] = np.array(struct.unpack_from("<9f", data, off)).reshape(3, 3)
        off += 38
    return tris


def stl_to_positions_normals(tris):
    positions = tris.reshape(-1, 3)
    normals = np.empty_like(positions)
    for i in range(len(tris)):
        a, b, c = tris[i]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        nrm = nrm / norm if norm > 1e-12 else np.array([0, 0, 1], dtype=np.float32)
        normals[i * 3:i * 3 + 3] = nrm
    return positions.astype(np.float32), normals.astype(np.float32)


def b64f32(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def b64u32(arr):
    return base64.b64encode(np.asarray(arr, dtype=np.uint32).tobytes()).decode("ascii")


if __name__ == "__main__":
    # cord: indexed + radially decomposed, for live GA rescaling
    cord = np.load(os.path.join(GEOM, "cord_radial.npz"))
    cord_verts, cord_radial, cord_centre, cord_faces = cord["verts"], cord["radial"], cord["centre"], cord["faces"]

    # placenta: static, non-interactive
    placenta_tris = read_binary_stl_tris(os.path.join(GEOM, "placenta.stl"))
    placenta_pos, placenta_norm = stl_to_positions_normals(placenta_tris)

    uv = np.load(os.path.join(GEOM, "uv_centerline.npy")).astype(np.float32)
    local_cord_dist = np.load(os.path.join(GEOM, "local_cord_dist.npy")).astype(np.float32)
    taper = np.load(os.path.join(GEOM, "taper.npy")).astype(np.float32)

    uv_arc = np.linalg.norm(np.diff(uv, axis=0), axis=1).sum()
    straight = np.linalg.norm(uv[0] - uv[-1])
    print(f"UV arc length: {uv_arc:.1f}mm, straight-line: {straight:.1f}mm")

    data = {
        "cord_verts_b64": b64f32(cord_verts.ravel()), "cord_radial_b64": b64f32(cord_radial.ravel()),
        "cord_centre_b64": b64f32(cord_centre.ravel()), "cord_faces_b64": b64u32(cord_faces.ravel()),
        "cord_n_verts": len(cord_verts), "cord_n_faces": len(cord_faces),

        "placenta_pos_b64": b64f32(placenta_pos.ravel()), "placenta_norm_b64": b64f32(placenta_norm.ravel()),
        "placenta_n": len(placenta_pos),

        "uv_pos_b64": b64f32(uv.ravel()), "uv_n": len(uv),
        "local_cord_dist_b64": b64f32(local_cord_dist), "taper_b64": b64f32(taper),

        "growth_curves": {
            "uv_ga": UV_GA.tolist(), "uv_diam_mm": UV_DIAM_MM.tolist(),
            "ua_ga": UA_GA.tolist(), "ua_diam_mm": UA_DIAM_MM.tolist(),
        },
        "meta": {
            "uv_arc_mm": float(uv_arc), "uv_straight_mm": float(straight),
            "reference_ga": REFERENCE_GA, "default_pitch_mm": 50.0,
            "gap_mm": 1.5, "delta_theta_deg": 40.0, "taper_mm": 20.0,
        },
    }
    out_path = os.path.join(VIEWER, "scene_data.js")
    with open(out_path, "w") as f:
        f.write("window.SCENE_DATA = ")
        json.dump(data, f)
        f.write(";\n")
    print("wrote", out_path, os.path.getsize(out_path), "bytes")
