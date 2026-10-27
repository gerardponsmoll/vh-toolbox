"""Procrustes alignment and Iterative Closest Point (lectures 3.2 and 4.1).

Points are rows: X is (N, 2) or (N, 3). A similarity transform maps x to s R x + t.
"""

from __future__ import annotations

import numpy as np


def procrustes(X: np.ndarray, Y: np.ndarray, scale: bool = True):
    """s, R, t minimising sum_i |s R x_i + t - y_i|^2 for corresponding rows of X and Y.

    Centre both point sets, take the SVD of the cross-covariance Ybar^T Xbar = U S V^T,
    then R = U V^T (with the last column flipped if that would be a reflection),
    s = trace(S) / |Xbar|^2 and t = mean(Y) - s R mean(X).
    """
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    D = np.eye(X.shape[1])
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))  # a rotation, never a mirror
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc**2).sum()) if scale else 1.0
    t = my - s * R @ mx
    return s, R, t


def transform(X: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * X @ R.T + t


def closest_points(queries: np.ndarray, points: np.ndarray, chunk: int = 1024):
    """For every query, the index of the nearest point and the squared distance (brute force)."""
    index = np.empty(len(queries), dtype=np.int64)
    sq_dist = np.empty(len(queries))
    p2 = (points**2).sum(axis=1)
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        d = (q**2).sum(axis=1)[:, None] - 2.0 * q @ points.T + p2[None, :]
        index[start : start + chunk] = d.argmin(axis=1)
        sq_dist[start : start + chunk] = np.maximum(d[np.arange(len(q)), index[start : start + chunk]], 0.0)
    return index, sq_dist


def correspondences(X: np.ndarray, Y: np.ndarray, s: float, R: np.ndarray, t: np.ndarray, target_to_source: bool = True):
    """Closest-point pairs (source index, target index) and their squared distances.

    target_to_source (the lecture's choice): every target point y_i picks its closest
    transformed source point. The other way round, every source point picks a target point;
    then shrinking the source always lowers the error, so with scale it collapses.
    """
    moved = transform(X, s, R, t)
    if target_to_source:
        source_index, sq_dist = closest_points(Y, moved)
        return np.stack([source_index, np.arange(len(Y))], axis=1), sq_dist
    target_index, sq_dist = closest_points(moved, Y)
    return np.stack([np.arange(len(X)), target_index], axis=1), sq_dist


def icp_step(X: np.ndarray, Y: np.ndarray, s: float, R: np.ndarray, t: np.ndarray, trim: float = 0.0, scale: bool = True,
             target_to_source: bool = True):
    """One ICP iteration: closest-point pairs, drop the worst `trim` fraction, Procrustes on the rest.

    Returns the new (s, R, t), the RMS distance of the kept pairs before the update,
    and the kept pairs as (source index, target index).
    """
    pairs, sq_dist = correspondences(X, Y, s, R, t, target_to_source)
    keep = np.argsort(sq_dist)[: max(10, int(round(len(pairs) * (1.0 - trim))))]
    rms = float(np.sqrt(sq_dist[keep].mean()))
    pairs = pairs[keep]
    s, R, t = procrustes(X[pairs[:, 0]], Y[pairs[:, 1]], scale=scale)
    return (s, R, t), rms, pairs


def gradient_steps(X: np.ndarray, Y: np.ndarray, pairs: np.ndarray, s: float, R: np.ndarray, t: np.ndarray,
                   alpha: float, steps: int = 1, scale: bool = True):
    """Gradient descent on E/N = mean_i |f(x_i) - y_i|^2 for fixed pairs, instead of solving Procrustes.

    The parameters act about the source centroid xbar, which keeps them on a similar scale:
    f(x) = e^sigma R (x - xbar) + c, so s = e^sigma and t = c - s R xbar. With r_i = f(x_i) - y_i
    and b_i = s R (x_i - xbar), the gradients are
        dE/dc = 2 mean r_i,   dE/dsigma = 2 mean r_i . b_i,
        rotation: 2 mean b_i x r_i in 3D (R <- exp(-alpha g) R), 2 mean r_i . (J b_i) in 2D.
    A general deformation model is fitted the same way, with its own parameters.
    """
    xbar = X.mean(axis=0)
    A, Yp = X[pairs[:, 0]] - xbar, Y[pairs[:, 1]]
    sigma, c = np.log(s), s * R @ xbar + t
    for _ in range(steps):
        B = np.exp(sigma) * A @ R.T
        r = B + c - Yp
        g_c, g_sigma = 2 * r.mean(axis=0), 2 * (r * B).sum(axis=1).mean()
        if X.shape[1] == 2:
            g_theta = 2 * (r[:, 1] * B[:, 0] - r[:, 0] * B[:, 1]).mean()
            a = -alpha * g_theta
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ R
        else:
            from vh.rotations import exp_map

            R = exp_map(-alpha * 2 * np.cross(B, r).mean(axis=0)) @ R
        c, sigma = c - alpha * g_c, sigma - alpha * g_sigma * scale
    s = float(np.exp(sigma))
    return s, R, c - s * R @ xbar
