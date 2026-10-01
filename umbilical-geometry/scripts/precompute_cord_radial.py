"""
Precompute, for the cord sheath STL, a per-vertex decomposition
(centreline point + radial offset vector) that lets GA-driven diameter
rescaling become a single cheap vector op in the viewer:

  new_position = centreline_point + radial_vector * scale

Also converts the raw STL (triangle soup, 3 duplicate vertices per
triangle) into an indexed format (unique vertices + a face index array),
both smaller and letting three.js's computeVertexNormals() work correctly
after vertices move. Same method as fetal-atlas-viewer's
ga_transition/scripts/build_ga_transition_geometry.py.
"""
import struct
import os
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.join(HERE, "..", "geometry")


def read_binary_stl(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 80)[0]
    arr = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        off += 12
        arr[i] = np.array(struct.unpack_from("<9f", data, off)).reshape(3, 3)
        off += 38
    return arr


def radial_decompose(stl_path, k=40):
    tris = read_binary_stl(stl_path)
    verts_flat = tris.reshape(-1, 3)
    unique_verts, inverse = np.unique(verts_flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    print(f"{stl_path}: {len(unique_verts)} unique vertices, {len(faces)} faces")

    tree = cKDTree(unique_verts)
    radial_vecs = np.empty_like(unique_verts)
    centre_pts = np.empty_like(unique_verts)
    for i, p in enumerate(unique_verts):
        dist, nb_idx = tree.query(p, k=k)
        nb = unique_verts[nb_idx]
        centroid = nb.mean(axis=0)
        cov = np.cov((nb - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent = eigvecs[:, np.argmax(eigvals)]
        offset = p - centroid
        radial = offset - np.dot(offset, tangent) * tangent
        radial_vecs[i] = radial
        centre_pts[i] = p - radial
    return unique_verts, radial_vecs, centre_pts, faces


if __name__ == "__main__":
    verts, radial, centre, faces = radial_decompose(os.path.join(GEOM, "cord_sheath.stl"))
    np.savez(os.path.join(GEOM, "cord_radial.npz"),
             verts=verts.astype(np.float32), radial=radial.astype(np.float32),
             centre=centre.astype(np.float32), faces=faces.astype(np.int32))
    print("saved", os.path.join(GEOM, "cord_radial.npz"))
