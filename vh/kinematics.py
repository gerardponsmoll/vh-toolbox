"""Kinematic chains with twists and the product of exponentials.

A joint that rotates about an axis with direction w through a point q is
described by its twist xi = (v, w), with v = -w x q. Turning the joint by theta
moves every point of the segments after it by the rigid motion exp(theta xi^).
A chain of joints maps a point p_b of the last segment to

    p_s = exp(theta_1 xi_1^) ... exp(theta_n xi_n^) G_sb(0) p_b      (product of exponentials)

with all twists given in the spatial (world) frame at the rest pose.

Reference: Murray, Li and Sastry, "A Mathematical Introduction to Robotic Manipulation".
"""

from __future__ import annotations

import numpy as np

from vh.rotations import exp_map, hat


def revolute_twist(axis, point) -> np.ndarray:
    """Twist (v, w) of a joint turning about `axis` through `point`."""
    w = np.asarray(axis, dtype=np.float64)
    w = w / np.linalg.norm(w)
    return np.concatenate([-np.cross(w, point), w])


def twist_hat(xi) -> np.ndarray:
    """The 4 x 4 matrix xi^ = [[w^, v], [0, 0]]."""
    T = np.zeros((4, 4))
    T[:3, :3], T[:3, 3] = hat(xi[3:]), xi[:3]
    return T


def exp_twist(xi, theta: float) -> np.ndarray:
    """exp(theta xi^) in closed form, for a twist with |w| = 1:

    R = exp(theta w^),  t = (I - R)(w x v) + w w^T v theta.
    """
    v, w = np.asarray(xi[:3]), np.asarray(xi[3:])
    R = exp_map(theta * w)
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = (np.eye(3) - R) @ np.cross(w, v) + np.outer(w, w) @ v * theta
    return G


def adjoint(G: np.ndarray) -> np.ndarray:
    """The 6 x 6 matrix that moves a twist (v, w) by the rigid motion G."""
    R, t = G[:3, :3], G[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3], Ad[:3, 3:], Ad[3:, 3:] = R, hat(t) @ R, R
    return Ad


def forward_kinematics(twists, thetas) -> list[np.ndarray]:
    """The motion of every segment: [exp(theta_1 xi_1^) ... exp(theta_i xi_i^) for i = 1..n]."""
    motions, G = [], np.eye(4)
    for xi, theta in zip(twists, thetas):
        G = G @ exp_twist(xi, theta)
        motions.append(G)
    return motions


def spatial_jacobian(twists, thetas) -> np.ndarray:
    """J (6, n): column i is twist i moved by the joints before it, xi_i' = Ad(exp(xi_1 theta_1) ... ) xi_i.

    J dtheta is the twist of the end segment, for any point attached to it.
    """
    columns, G = [], np.eye(4)
    for xi, theta in zip(twists, thetas):
        columns.append(adjoint(G) @ xi)
        G = G @ exp_twist(xi, theta)
    return np.stack(columns, axis=1)


def point_jacobian(twists, thetas, p) -> np.ndarray:
    """J_p (3, n): how fast the point p (on the end segment, in the world) moves per joint.

    Column i is xi_i'^ p = v_i' + w_i' x p: what p does if only joint i turns.
    """
    J = spatial_jacobian(twists, thetas)
    return J[:3] + np.cross(J[3:].T, p).T


def damped_least_squares(J: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    """The step dtheta that minimises |J dtheta - error|^2 + damping^2 |dtheta|^2.

    damping = 0 is a Gauss-Newton step. Near a singularity, where J loses rank,
    the damping keeps the step from becoming huge.
    """
    n = J.shape[1]
    return np.linalg.solve(J.T @ J + damping**2 * np.eye(n), J.T @ error)
