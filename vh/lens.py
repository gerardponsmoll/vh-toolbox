"""The thin (Gaussian) lens: where a point is sharp, and how blurred it is elsewhere.

All distances are along the optical axis, in the same unit (for example metres):

    f    focal length of the lens
    z    depth of a point in front of the lens
    s    distance from the lens to the sensor
    A    diameter of the aperture, A = f / N for f-number N

A point at depth z is sharp at the distance image_distance(f, z) behind the lens.
The ray through the centre of the lens is not bent, so every point lands where a
pinhole at the lens centre, at distance s from the sensor, would put it. A point
that is not sharp on the sensor becomes a disc around that position, with
diameter blur_diameter(A, s, f, z).
"""

from __future__ import annotations

import numpy as np


def image_distance(f: float, z: np.ndarray | float) -> np.ndarray | float:
    """Gauss's lens formula 1/z + 1/z' = 1/f, solved for z' (only for z > f)."""
    return 1.0 / (1.0 / f - 1.0 / np.asarray(z, dtype=np.float64))


def blur_diameter(A: float, s: float, f: float, z: np.ndarray | float) -> np.ndarray | float:
    """Diameter of the blur disc on a sensor at distance s for a point at depth z.

    The rays from the point fill the aperture (diameter A) and meet at z'. By
    similar triangles they cover a disc of diameter A * |s - z'| / z' on the
    sensor, which is A * s * |1/z_f - 1/z| where z_f is the depth in focus.
    """
    z_image = image_distance(f, z)
    return A * np.abs(s - z_image) / z_image


def disc_kernel(radius: float) -> np.ndarray:
    """A disc of the given radius in pixels, with soft edges, that sums to 1."""
    r = int(np.ceil(radius + 1.0))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    kernel = np.clip(radius + 0.5 - np.hypot(x, y), 0.0, 1.0)
    return kernel / kernel.sum()


def disc_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Blur an image (H, W) or (H, W, C) with a disc of the given radius in pixels."""
    if radius < 0.5:
        return image
    kernel = disc_kernel(radius)
    r = kernel.shape[0] // 2
    pad = [(r, r), (r, r)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image, pad, mode="edge")
    shape = padded.shape[:2]
    spectrum = np.fft.rfft2(np.pad(kernel, [(0, shape[0] - kernel.shape[0]), (0, shape[1] - kernel.shape[1])]), axes=(0, 1))
    if image.ndim == 3:
        spectrum = spectrum[..., None]
    blurred = np.fft.irfft2(np.fft.rfft2(padded, axes=(0, 1)) * spectrum, s=shape, axes=(0, 1))
    # The kernel's centre was at (r, r), so the result is shifted by r: the crop undoes it.
    return blurred[2 * r :, 2 * r :][: image.shape[0], : image.shape[1]]
