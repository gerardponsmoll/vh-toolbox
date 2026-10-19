import numpy as np

from vh.kinematics import exp_twist, forward_kinematics, point_jacobian, revolute_twist, twist_hat
from vh.rotations import (
    euler_to_matrix,
    exp_map,
    log_map,
    matrix_to_euler_xyz,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_multiply,
    quat_rotate,
    quat_to_matrix,
    slerp,
)

rng = np.random.default_rng(0)
R_RANDOM = [exp_map(r) for r in rng.normal(size=(20, 3))]


def test_exp_and_log_are_inverse():
    for R in R_RANDOM + [exp_map([np.pi, 0.0, 0.0]), exp_map([0.0, 0.0, np.pi - 1e-6])]:
        assert np.allclose(exp_map(log_map(R)), R, atol=1e-6)


def test_rotations_do_not_commute_in_3d():
    a, b = exp_map([np.pi / 2, 0, 0]), exp_map([0, np.pi / 2, 0])
    assert not np.allclose(a @ b, b @ a)


def test_euler_xyz_round_trip_and_extrinsic_is_reversed_intrinsic():
    for angles in rng.uniform(-1.5, 1.5, size=(20, 3)):
        R = euler_to_matrix(angles, "xyz")
        assert np.allclose(matrix_to_euler_xyz(R), angles)
        assert np.allclose(euler_to_matrix(angles[::-1], "zyx", intrinsic=False), R)


def test_gimbal_lock_loses_one_degree_of_freedom():
    # At b = 90 degrees only a + c matters.
    R1 = euler_to_matrix([0.3, np.pi / 2, 0.5], "xyz")
    R2 = euler_to_matrix([0.6, np.pi / 2, 0.2], "xyz")
    assert np.allclose(R1, R2)


def test_quaternions_rotate_like_matrices():
    for R in R_RANDOM:
        q = quat_from_matrix(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(quat_to_matrix(q), R)
        assert np.allclose(quat_to_matrix(-q), R)  # q and -q are the same rotation
        p = rng.normal(size=3)
        assert np.allclose(quat_rotate(q, p), R @ p)
    q1, q2 = quat_from_matrix(R_RANDOM[0]), quat_from_matrix(R_RANDOM[1])
    assert np.allclose(quat_to_matrix(quat_multiply(q1, q2)), R_RANDOM[0] @ R_RANDOM[1])


def test_slerp_has_constant_speed():
    q0, q1 = quat_from_axis_angle([0, 0, 0]), quat_from_axis_angle([0, 2.0, 0])
    angles = [np.linalg.norm(log_map(quat_to_matrix(slerp(q0, q1, t)))) for t in np.linspace(0, 1, 5)]
    assert np.allclose(np.diff(angles), 0.5)


def test_twist_exponential_matches_the_series():
    xi = revolute_twist([0, 0, 1], [1.0, 2.0, 0.0])
    G, M, term = np.eye(4), 0.7 * twist_hat(xi), np.eye(4)
    for k in range(1, 30):
        term = term @ M / k
        G = G + term
    assert np.allclose(exp_twist(xi, 0.7), G)
    assert np.allclose(exp_twist(xi, 0.7) @ [1.0, 2.0, 5.0, 1.0], [1.0, 2.0, 5.0, 1.0])  # points on the axis stay


def test_point_jacobian_matches_finite_differences():
    twists = [revolute_twist([0, 1, 0], [0, 0, 0]), revolute_twist([0, 0, 1], [0, 1, 0]), revolute_twist([0, 0, 1], [1, 1, 0])]
    p0 = np.array([2.0, 1.0, 0.0, 1.0])
    thetas = np.array([0.3, -0.4, 0.8])

    def tip(t):
        return (forward_kinematics(twists, t)[-1] @ p0)[:3]

    J = point_jacobian(twists, thetas, tip(thetas))
    numeric = np.stack([(tip(thetas + 1e-6 * e) - tip(thetas - 1e-6 * e)) / 2e-6 for e in np.eye(3)], axis=1)
    assert np.allclose(J, numeric, atol=1e-6)
