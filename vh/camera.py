"""The pinhole camera in plain NumPy, written to be read alongside the lecture slides.

A point X in the world becomes a pixel in three steps:

    1. world -> camera     X_c = R X + t                 (extrinsics)
    2. perspective divide  x   = (X_c / Z_c, Y_c / Z_c)  (the only nonlinear step)
    3. image plane -> pixels   (u, v, 1) = K (x, y, 1)   (intrinsics)

The camera frame has x to the right, y down and z along the viewing direction,
so that pixel (0, 0) is the top left corner of the image.
"""

from __future__ import annotations

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = (0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Extrinsics (R, t) of a camera at `eye` that looks at `target`.

    The rows of R are the camera axes written in world coordinates, and
    t = -R eye, so that the camera centre maps to the origin.
    """
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def intrinsics(f: float, cx: float, cy: float) -> np.ndarray:
    """K for square pixels: focal length f and principal point (cx, cy), all in pixels."""
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def world_to_camera(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points (N, 3) in the world -> the same points in the camera frame."""
    return X @ R.T + t


def project(X_cam: np.ndarray, K: np.ndarray, k1: float = 0.0, depth: float | None = None) -> np.ndarray:
    """Points (N, 3) in the camera frame -> pixels (N, 2).

    depth: None for perspective projection, where every point is divided by its
           own depth. Give one depth to divide all points by it instead: this is
           weak perspective, a good approximation when the object is small
           compared to its distance.
    k1:    radial lens distortion. Points move away from (k1 > 0) or towards
           (k1 < 0) the principal point, by a factor that grows with the squared radius.

    Points behind the camera (Z_c <= 0) have no image: remove them first.
    """
    z = X_cam[:, 2:3] if depth is None else depth
    x = X_cam[:, :2] / z
    x = x * (1.0 + k1 * (x**2).sum(axis=1, keepdims=True))
    return x @ K[:2, :2].T + K[:2, 2]


def project_cylinder(X_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Points (N, 3) in the camera frame -> pixels (N, 2) on a cylindrical panorama.

    The image is wrapped around the camera on a cylinder of radius f with a
    vertical axis, as when a panorama is stitched from photos taken while
    turning. u measures the angle to the left or right, so a point's size
    now depends on its distance sqrt(X^2 + Z^2) and not on its depth Z, and
    straight horizontal lines become curves.
    """
    angle = np.arctan2(X_cam[:, 0], X_cam[:, 2])
    height = X_cam[:, 1] / np.hypot(X_cam[:, 0], X_cam[:, 2])
    return np.stack([angle, height], axis=1) @ K[:2, :2].T + K[:2, 2]


def pixel_ray(pixel: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ray of world points that a pixel sees, as (origin, unit direction).

    The origin is the camera centre C = -R^T t. One image cannot tell where on
    this ray the point was (no lens distortion assumed).
    """
    direction = R.T @ np.linalg.solve(K, np.array([pixel[0], pixel[1], 1.0]))
    return -R.T @ t, direction / np.linalg.norm(direction)
