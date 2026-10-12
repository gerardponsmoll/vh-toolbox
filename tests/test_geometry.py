import numpy as np

from vh.geometry import ray_mesh_intersection

# Two triangles facing the viewer: a near one at z = 1 and a far one at z = 0.
VERTICES = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float
)
FACES = np.array([[0, 1, 2], [3, 4, 5]])


def test_ray_hits_nearest_triangle():
    face, bary = ray_mesh_intersection([0.2, 0.3, 5.0], [0, 0, -1], VERTICES, FACES)
    assert face == 1
    assert np.allclose(bary @ VERTICES[FACES[face]], [0.2, 0.3, 1.0])


def test_ray_ignores_triangles_behind_it():
    face, _ = ray_mesh_intersection([0.2, 0.3, 0.5], [0, 0, -1], VERTICES, FACES)
    assert face == 0


def test_ray_misses():
    assert ray_mesh_intersection([2.0, 2.0, 5.0], [0, 0, -1], VERTICES, FACES) is None
    assert ray_mesh_intersection([0.2, 0.3, 5.0], [0, 0, 1], VERTICES, FACES) is None
