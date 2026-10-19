"""Ways to write down a 3D rotation, written to be read alongside the lecture slides.

The same rotation can be stored as

    a rotation matrix R      9 numbers, R^T R = I and det R = 1
    Euler angles             3 numbers, three rotations about coordinate axes
    a unit quaternion q      4 numbers, |q| = 1 (q and -q are the same rotation)
    axis-angle r = theta w   3 numbers, R = exp(theta [w]x) (the exponential map)

and this file converts between them. Quaternions are stored as (w, x, y, z).
"""

from __future__ import annotations

import numpy as np

from vh.smpl import rodrigues

AXES = {"x": 0, "y": 1, "z": 2}


def hat(w) -> np.ndarray:
    """The skew-symmetric matrix [w]x, so that hat(w) @ p = cross(w, p)."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_map(r) -> np.ndarray:
    """Axis-angle r = theta * w -> rotation matrix, with the Rodrigues formula
    R = I + sin(theta) [w]x + (1 - cos(theta)) [w]x^2."""
    return rodrigues(np.asarray(r, dtype=np.float64))[0]


def log_map(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle r = theta * w, with theta in [0, pi]."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3)
    if np.pi - angle < 1e-4:  # near half a turn sin is ~0: read the axis from R + I instead
        S = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(S)))
        axis = S[:, k] / np.sqrt(S[k, k])
        return angle * axis
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return angle * w


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    """Rotation by `angle` (radians) about the x, y or z axis."""
    return exp_map(angle * np.eye(3)[AXES[axis]])


def euler_to_matrix(angles, order: str = "xyz", intrinsic: bool = True) -> np.ndarray:
    """Euler angles (radians) -> rotation matrix.

    order: the axes of the three rotations, e.g. "xyz", "zyx" or "zxz".
    intrinsic: each rotation is about the axes of the body, already turned by
               the ones before: R = R_1 R_2 R_3. Extrinsic: about the fixed
               world axes, so the first rotation is applied first: R = R_3 R_2 R_1.
    """
    Rs = [axis_rotation(a, t) for a, t in zip(order, angles)]
    return Rs[0] @ Rs[1] @ Rs[2] if intrinsic else Rs[2] @ Rs[1] @ Rs[0]


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> intrinsic x-y-z Euler angles (radians), R = R_x(a) R_y(b) R_z(c).

    At b = +-90 degrees (gimbal lock) only a + c or a - c is defined; c is then set to 0.
    """
    b = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    if abs(R[0, 2]) < 1.0 - 1e-9:
        return np.array([np.arctan2(-R[1, 2], R[2, 2]), b, np.arctan2(-R[0, 1], R[0, 0])])
    return np.array([np.arctan2(R[2, 1], R[1, 1]), b, 0.0])


def quat_from_axis_angle(r) -> np.ndarray:
    """Axis-angle r = theta * w -> q = (cos(theta / 2), sin(theta / 2) w)."""
    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r)
    axis = r / angle if angle > 1e-12 else np.array([1.0, 0.0, 0.0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_to_axis_angle(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    angle = 2.0 * np.arctan2(np.linalg.norm(q[1:]), q[0])
    if abs(angle) < 1e-12:
        return np.zeros(3)
    return angle * q[1:] / np.linalg.norm(q[1:])


def quat_multiply(p, q) -> np.ndarray:
    """The quaternion product p q: first rotate by q, then by p."""
    pw, pv, qw, qv = p[0], np.asarray(p[1:]), q[0], np.asarray(q[1:])
    return np.concatenate([[pw * qw - pv @ qv], pw * qv + qw * pv + np.cross(pv, qv)])


def quat_conjugate(q) -> np.ndarray:
    return np.concatenate([[q[0]], -np.asarray(q[1:])])


def quat_rotate(q, p) -> np.ndarray:
    """Rotate the point p by the unit quaternion q: q (0, p) q*."""
    return quat_multiply(quat_multiply(q, np.concatenate([[0.0], p])), quat_conjugate(q))[1:]


def quat_to_matrix(q) -> np.ndarray:
    return exp_map(quat_to_axis_angle(q))


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    return quat_from_axis_angle(log_map(R))


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical linear interpolation: the shortest arc from q0 (t = 0) to q1 (t = 1), at constant speed."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    if q0 @ q1 < 0:  # q1 and -q1 are the same rotation: take the one on the shorter arc
        q1 = -q1
    angle = np.arccos(np.clip(q0 @ q1, -1.0, 1.0))
    if angle < 1e-8:
        return q0
    return (np.sin((1 - t) * angle) * q0 + np.sin(t * angle) * q1) / np.sin(angle)
