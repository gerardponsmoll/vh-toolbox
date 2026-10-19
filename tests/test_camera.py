import numpy as np

from vh.camera import intrinsics, look_at, pixel_ray, project, project_cylinder, world_to_camera
from vh.render import sample_surface, splat

K = intrinsics(500.0, 240.0, 180.0)
EYE, TARGET = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0])
R, T = look_at(EYE, TARGET)


def test_look_at_is_a_rotation_that_centres_the_target():
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(world_to_camera(EYE[None], R, T), 0.0)
    distance = np.linalg.norm(TARGET - EYE)
    assert np.allclose(world_to_camera(TARGET[None], R, T), [[0.0, 0.0, distance]])
    assert np.allclose(project(world_to_camera(TARGET[None], R, T), K), [[240.0, 180.0]])


def test_camera_y_axis_points_down():
    above = world_to_camera(TARGET[None] + [0.0, 0.1, 0.0], R, T)
    assert project(above, K)[0, 1] < 180.0


def test_every_point_on_a_pixel_ray_projects_to_that_pixel():
    pixel = np.array([100.0, 300.0])
    origin, direction = pixel_ray(pixel, K, R, T)
    assert np.allclose(origin, EYE)
    points = origin + np.array([[0.5], [2.0], [40.0]]) * direction
    assert np.allclose(project(world_to_camera(points, R, T), K), pixel)


def test_weak_perspective_matches_perspective_far_away():
    rng = np.random.default_rng(0)
    body = rng.uniform(-0.5, 0.5, size=(50, 3))

    def difference(distance: float) -> float:
        X = body + [0.0, 0.0, distance]
        f = 200.0 * distance  # keep the object the same size in the image
        K = intrinsics(f, 0.0, 0.0)
        return np.abs(project(X, K) - project(X, K, depth=distance)).max()

    assert difference(2.0) > 10.0
    assert difference(100.0) < 0.5


def test_image_size_depends_on_depth_on_a_plane_and_on_distance_on_a_cylinder():
    # Two equal vertical poles, 1 m tall, on a wall 5 m in front of the camera: one ahead, one 8 m to the side.
    K = intrinsics(100.0, 0.0, 0.0)
    poles = np.array([[0.0, -0.5, 5.0], [0.0, 0.5, 5.0], [8.0, -0.5, 5.0], [8.0, 0.5, 5.0]])

    def heights(pixels):
        return pixels[1::2, 1] - pixels[0::2, 1]

    assert np.allclose(heights(project(poles, K)), [100.0 / 5.0, 100.0 / 5.0])  # f H / Z for both
    assert np.allclose(heights(project_cylinder(poles, K)), [100.0 / 5.0, 100.0 / np.hypot(8.0, 5.0)])  # f H / distance
    assert np.allclose(project_cylinder(poles, K)[2, 0], 100.0 * np.arctan2(8.0, 5.0))


def test_distortion_keeps_the_principal_point_and_pushes_the_rest_outwards():
    X = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    plain, barrel = project(X, K), project(X, K, k1=0.2)
    assert np.allclose(barrel[0], plain[0])
    assert barrel[1, 0] > plain[1, 0]


def test_splat_keeps_the_nearest_point():
    pixels = np.array([[2.5, 1.5], [2.2, 1.9], [0.0, 0.0], [9.0, 9.0]])
    depth = np.array([5.0, 1.0, -1.0, 1.0])  # third is behind the camera, fourth outside
    colors = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]])
    image = splat(pixels, depth, colors, width=4, height=3, background=0.5)
    assert np.allclose(image[1, 2], [0, 1.0, 0])
    assert np.allclose(np.delete(image.reshape(-1, 3), 1 * 4 + 2, axis=0), 0.5)
    assert np.allclose(splat(pixels, depth, colors, 4, 3, size=3, background=0.5)[0, 1], [0, 1.0, 0])


def test_sample_surface_stays_on_the_mesh():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]])
    faces = np.array([[0, 1, 2], [3, 3, 3]])  # the second triangle has no area
    face, bary = sample_surface(vertices, faces, 200)
    assert (face == 0).all()
    assert np.allclose(bary.sum(axis=1), 1.0) and (bary >= 0).all()
