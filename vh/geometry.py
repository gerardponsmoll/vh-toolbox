"""Small geometric helpers shared by the demos."""

from __future__ import annotations

import numpy as np


def ray_mesh_intersection(
    origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> tuple[int, np.ndarray] | None:
    """First triangle hit by a ray, as (face index, barycentric coordinates), or None.

    Moller-Trumbore test against every triangle at once: solve
    origin + t * direction = a + u * (b - a) + v * (c - a) for (t, u, v).
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    e1, e2 = b - a, c - a

    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    parallel = np.abs(det) < 1e-12
    inv = 1.0 / np.where(parallel, 1.0, det)

    s = origin - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv

    hit = ~parallel & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not hit.any():
        return None
    face = int(np.where(hit, t, np.inf).argmin())
    return face, np.array([1.0 - u[face] - v[face], u[face], v[face]])
