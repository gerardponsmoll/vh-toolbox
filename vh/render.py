"""A very small point renderer: enough to draw what a camera sees with NumPy only.

A surface is drawn as many points sampled on it. Each point is projected with
the camera model and written into the image, and where several points fall on
the same pixel the nearest one wins. Because every point goes through the
projection function on its own, any camera model can be drawn this way,
including lens distortion.
"""

from __future__ import annotations

import numpy as np


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random points on a mesh, uniform over its area.

    Returns the triangle of each point (count,) and its barycentric coordinates
    (count, 3), so the points can follow the mesh when its vertices move.
    weights: (F,) to sample some triangles more densely than others.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if weights is not None:
        area = area * weights
    face = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = rng.random((2, count))
    s = np.sqrt(r1)  # the square root makes the points uniform inside the triangle
    return face, np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)


def _shifted(offset: int, length: int) -> tuple[slice, slice]:
    """Slices (source, destination) that move an axis of `length` by `offset`."""
    return slice(max(0, -offset), length - max(0, offset)), slice(max(0, offset), length - max(0, -offset))


def splat(
    pixels: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    size: int | np.ndarray = 1,
    background: float = 1.0,
) -> np.ndarray:
    """Draw points into an image (height, width, 3), values in [0, 1].

    pixels: (N, 2) positions (u, v); depth: (N,) distance along the camera axis;
    colors: (N, 3). Each point covers size x size pixels. size can also be given
    per point (N,), so that near points are drawn larger than far ones.
    """
    if np.ndim(size) == 0:
        return _splat(pixels, depth, colors, width, height, int(size), background)[0]
    # One layer per point size, then keep the nearest layer in every pixel.
    image = np.full((height, width, 3), background, dtype=np.float64)
    z_buffer = np.full((height, width), np.inf)
    for one_size in np.unique(size):
        group = size == one_size
        layer, layer_z = _splat(pixels[group], depth[group], colors[group], width, height, int(one_size), background)
        nearer = layer_z < z_buffer
        image[nearer], z_buffer[nearer] = layer[nearer], layer_z[nearer]
    return image


def _splat(pixels, depth, colors, width: int, height: int, size: int, background: float):
    """Points of one size -> image and depth of the nearest point in every pixel."""
    u = np.floor(pixels[:, 0]).astype(np.int64)
    v = np.floor(pixels[:, 1]).astype(np.int64)
    inside = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    order = np.argsort(-depth[inside])  # far to near: the nearest point is written last
    index = (v * width + u)[inside][order]

    z_buffer = np.full(width * height, np.inf)
    image = np.full((width * height, 3), background, dtype=np.float64)
    z_buffer[index] = depth[inside][order]
    image[index] = colors[inside][order]
    z_buffer, image = z_buffer.reshape(height, width), image.reshape(height, width, 3)
    if size <= 1:
        return image, z_buffer

    # Grow every point to size x size pixels, still keeping the nearest one.
    grown_z, grown = z_buffer.copy(), image.copy()
    for dy in range(-(size // 2), size - size // 2):
        for dx in range(-(size // 2), size - size // 2):
            if dy == 0 and dx == 0:
                continue
            (src_y, dst_y), (src_x, dst_x) = _shifted(dy, height), _shifted(dx, width)
            nearer = z_buffer[src_y, src_x] < grown_z[dst_y, dst_x]
            grown_z[dst_y, dst_x][nearer] = z_buffer[src_y, src_x][nearer]
            grown[dst_y, dst_x][nearer] = image[src_y, src_x][nearer]
    return grown, grown_z


def draw_dots(image: np.ndarray, pixels: np.ndarray, color, size: int = 1) -> None:
    """Paint dots of size x size pixels over the image, in place."""
    height, width = image.shape[:2]
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    u = np.floor(pixels[:, 0]).astype(np.int64)
    v = np.floor(pixels[:, 1]).astype(np.int64)
    for dy in range(-(size // 2), size - size // 2):
        for dx in range(-(size // 2), size - size // 2):
            uu, vv = u + dx, v + dy
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            image[vv[inside], uu[inside]] = color


def draw_line(image: np.ndarray, start, end, color, size: int = 1) -> None:
    """Paint a straight line between two pixel positions, in place."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    steps = int(np.ceil(np.abs(end - start).max())) + 1
    draw_dots(image, start + np.linspace(0.0, 1.0, steps)[:, None] * (end - start), color, size)
