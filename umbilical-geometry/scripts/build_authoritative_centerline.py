"""
Build the dense, ordered UV/cord centreline from the real CMISS source mesh
(source_mesh/cord_offset.exnode/.exelem). Determines true path order via
element connectivity (a simple chain with two degree-1 endpoint nodes),
then densely samples the cubic Hermite curve within each element for a
smooth, high-resolution centreline usable for the synthetic-UA generation
pipeline (generate_ua_pair.py) or direct rendering.

Reproduces the two numbers used to validate this against the companion
manuscript: 617.7mm arc length (adopted value: 617.8mm) and 80.0mm
straight-line fetal-to-placental distance (reported value: 80.0mm).
"""
import os
import numpy as np
from parse_cmiss_cord import parse_exnode, parse_exelem, EXNODE, EXELEM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "geometry", "uv_centerline.npy")
N_PER_ELEMENT = 20


def hermite_eval(xi, p0, t0, p1, t1):
    h00 = 2 * xi**3 - 3 * xi**2 + 1
    h10 = xi**3 - 2 * xi**2 + xi
    h01 = -2 * xi**3 + 3 * xi**2
    h11 = xi**3 - xi**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def order_chain(nodes, elements):
    adjacency = {}
    for elem_id, n1, n2, scales in elements:
        adjacency.setdefault(n1, []).append((n2, (n1, n2, scales)))
        adjacency.setdefault(n2, []).append((n1, (n1, n2, scales)))
    degree1 = [n for n, adj in adjacency.items() if len(adj) == 1]
    assert len(degree1) == 2, f"expected a simple chain (2 endpoints), got {degree1}"

    chain = [degree1[0]]
    chain_elems = []
    current = degree1[0]
    visited_edges = set()
    while True:
        options = [(o, e) for o, e in adjacency[current] if tuple(sorted(e[:2])) not in visited_edges]
        if not options:
            break
        nxt, elem = options[0]
        visited_edges.add(tuple(sorted(elem[:2])))
        chain_elems.append(elem)
        chain.append(nxt)
        current = nxt
    return chain, chain_elems


if __name__ == "__main__":
    nodes = parse_exnode(EXNODE)
    elements = parse_exelem(EXELEM)
    chain, chain_elems = order_chain(nodes, elements)
    print(f"chain: {len(chain)} nodes; start={chain[0]} end={chain[-1]}")

    straight = np.linalg.norm(nodes[chain[0]][0] - nodes[chain[-1]][0])
    print(f"fetal-to-placental straight-line distance: {straight:.1f}mm")

    dense_points = []
    for (n1, n2, scales) in chain_elems:
        p0, d0 = nodes[n1]
        p1, d1 = nodes[n2]
        t0 = d0 * scales[1]
        t1 = d1 * scales[3]
        for xi in np.linspace(0, 1, N_PER_ELEMENT, endpoint=False):
            dense_points.append(hermite_eval(xi, p0, t0, p1, t1))
    dense_points.append(nodes[chain[-1]][0])
    dense_points = np.array(dense_points)

    seg = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    print(f"dense centreline: {len(dense_points)} points, arc length {seg.sum():.1f}mm")

    np.save(OUT, dense_points.astype(np.float32))
    print("saved", OUT)
