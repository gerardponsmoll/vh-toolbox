"""Simple coloured triangle meshes for the camera demos, and points sampled on them.

A Shape is a mesh with one colour per triangle. The demos draw what a camera
sees by sampling many points on the shape (`sample`) and splatting them into
an image (vh/render.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from vh.camera import project, project_cylinder, world_to_camera
from vh.render import draw_line, sample_surface, splat
from vh.smpl import SMPLModel, forward

SKIN = np.array([222, 184, 160]) / 255.0
BOX_COLORS = np.array([[0.85, 0.3, 0.25], [0.3, 0.65, 0.35], [0.25, 0.45, 0.85]])  # faces across x, y, z
LIGHT = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])
BODY_POSES = ["Reach towards the camera", "T-pose", "Arms down"]


@dataclass
class Shape:
    """A triangle mesh standing on the floor (y = 0), with one colour per triangle."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    face_colors: np.ndarray  # (F, 3), values in [0, 1]
    closed: bool = True  # a closed surface: triangles facing away from the camera can be skipped
    center: np.ndarray | None = None  # what the camera looks at; None: the middle of the object
    view: tuple[float, float, float] = (25.0, 10.0, 3.0)  # camera at the start: azimuth, elevation, distance
    detail: np.ndarray | float = 1.0  # per triangle (F,) or for all: how densely to sample it for the image

    def trimesh(self) -> trimesh.Trimesh:
        """The shape for the 3D view. Every triangle gets its own three vertices to keep its colour."""
        corners = self.vertices[self.faces].reshape(-1, 3)
        mesh = trimesh.Trimesh(corners, np.arange(len(corners)).reshape(-1, 3), process=False)
        rgb = (np.clip(self.face_colors, 0.0, 1.0) * 255).astype(np.uint8)
        mesh.visual.vertex_colors = np.repeat(np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], axis=1), 3, axis=0)
        return mesh


@dataclass
class Samples:
    """Points on a shape, ready to be drawn into an image."""

    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3), already shaded
    spacing: np.ndarray  # (N,) typical distance to the neighbouring points, in metres
    closed: bool


def sample(shape: Shape, count: int, seed: int = 0) -> Samples:
    """`count` points on the shape, denser where shape.detail is larger, with simple shading."""
    a, b, c = (shape.vertices[shape.faces[:, i]] for i in range(3))
    face_normals = np.cross(b - a, c - a)
    area = np.linalg.norm(face_normals, axis=1)
    face_normals = face_normals / np.maximum(area, 1e-12)[:, None]

    detail = np.broadcast_to(shape.detail, len(shape.faces))
    face, barycentric = sample_surface(shape.vertices, shape.faces, count, np.random.default_rng(seed), weights=detail)
    points = np.einsum("nk,nkd->nd", barycentric, shape.vertices[shape.faces[face]])
    normals = face_normals[face]
    spacing = np.sqrt(0.5 * (area * detail).sum() / (count * detail[face]))
    towards_light = normals @ LIGHT
    shade = 0.7 + 0.3 * (np.clip(towards_light, 0.0, 1.0) if shape.closed else np.abs(towards_light))
    return Samples(points, normals, shape.face_colors[face] * shade[:, None], spacing, shape.closed)


def checker_box(size, cell: float = 0.25, base_color=None) -> Shape:
    """A box with a checkerboard on every side. Straight, parallel edges show perspective well.

    base_color: one colour for all sides. By default the sides across x, y and z differ.
    """
    size = np.asarray(size, dtype=np.float64)
    vertices, faces, colors = [], [], []
    offset = 0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a, b = (axis + 1) % 3, (axis + 2) % 3
            if sign < 0:
                a, b = b, a  # keeps the triangles facing outwards
            na, nb = (max(1, round(size[k] / cell)) for k in (a, b))
            ga, gb = np.meshgrid(np.linspace(-0.5, 0.5, na + 1), np.linspace(-0.5, 0.5, nb + 1), indexing="ij")
            corners = np.zeros((na + 1, nb + 1, 3))
            corners[..., axis], corners[..., a], corners[..., b] = 0.5 * sign, ga, gb
            vertices.append(corners.reshape(-1, 3) * size)

            i, j = (g.ravel() for g in np.meshgrid(np.arange(na), np.arange(nb), indexing="ij"))
            p00 = offset + i * (nb + 1) + j
            p10, p01 = p00 + nb + 1, p00 + 1
            faces += [np.stack([p00, p10, p10 + 1], axis=1), np.stack([p00, p10 + 1, p01], axis=1)]
            color = (BOX_COLORS[axis] if base_color is None else np.asarray(base_color)) * np.where((i + j) % 2 == 0, 1.0, 0.6)[:, None]
            colors += [color, color]
            offset += (na + 1) * (nb + 1)
    vertices = np.concatenate(vertices)
    vertices[:, 1] -= vertices[:, 1].min()
    return Shape(vertices, np.concatenate(faces), np.concatenate(colors))


def ground(width: float, depth: float, cell: float, color, center=(0.0, 0.0)) -> Shape:
    """A tiled floor at y = 0 (only its top side), `width` along x and `depth` along z."""
    box = checker_box((width, 0.1, depth), cell=cell, base_color=color)
    a, b, c = (box.vertices[box.faces[:, i]] for i in range(3))
    top = np.cross(b - a, c - a)[:, 1] > 0
    return Shape(box.vertices + [center[0], -0.1, center[1]], box.faces[top], box.face_colors[top])


def body(model: SMPLModel, pose_name: str, color=SKIN) -> Shape:
    pose = np.zeros((model.num_joints, 3))
    if pose_name == BODY_POSES[0]:
        pose[16] = [0.0, -1.4, 0.0]  # left shoulder: arm forwards
        pose[17] = [0.0, 0.0, 1.2]  # right shoulder: arm down
    elif pose_name == BODY_POSES[2]:
        pose[16], pose[17] = [0.0, 0.0, -1.2], [0.0, 0.0, 1.2]
    vertices = forward(model, pose=pose).vertices
    vertices = vertices - [0.0, vertices[:, 1].min(), 0.0]
    return Shape(vertices, model.faces, np.tile(np.asarray(color, dtype=np.float64), (len(model.faces), 1)))


def stick_figure(color=SKIN, height: float = 1.75) -> Shape:
    """A person made of a capsule and a ball, for when no SMPL model is installed."""
    head = 0.12 * height
    torso = solid(trimesh.creation.capsule(height=0.75 * height - 2 * 0.15 * height, radius=0.15 * height), color, z_to_y=True)
    torso.vertices[:, 1] -= torso.vertices[:, 1].min()
    torso.vertices[:, 0] *= 1.3  # a bit wider than deep
    top = torso.vertices[:, 1].max()
    return merge([torso, solid(trimesh.creation.icosphere(subdivisions=2, radius=head), color, (0.0, top + head, 0.0))])


def solid(mesh: trimesh.Trimesh, color, position=(0.0, 0.0, 0.0), z_to_y: bool = False) -> Shape:
    """A trimesh shape in one colour, moved to `position`. z_to_y stands shapes built along z upright."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if z_to_y:
        vertices = vertices[:, [0, 2, 1]] * [1.0, 1.0, -1.0]  # a rotation by 90 degrees about x
    return Shape(vertices + position, np.asarray(mesh.faces), np.tile(np.asarray(color, dtype=np.float64), (len(mesh.faces), 1)))


def merge(parts: list[Shape], **kwargs) -> Shape:
    """Several shapes as one."""
    offsets = np.cumsum([0] + [len(part.vertices) for part in parts[:-1]])
    return Shape(
        np.concatenate([part.vertices for part in parts]),
        np.concatenate([part.faces + offset for part, offset in zip(parts, offsets)]),
        np.concatenate([part.face_colors for part in parts]),
        detail=np.concatenate([np.broadcast_to(part.detail, len(part.faces)) for part in parts]),
        **kwargs,
    )


def moved(shape: Shape, offset) -> Shape:
    return Shape(shape.vertices + offset, shape.faces, shape.face_colors, shape.closed, detail=shape.detail)


def take_picture(
    samples: Samples,
    eye: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    cylinder: bool = False,
    colors: np.ndarray | None = None,
    near: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """The image (height, width, 3) a camera takes of the samples, and a mask of the pixels they cover.

    cylinder: project onto a cylindrical panorama instead of a flat image plane.
    colors: (N, 3) to use instead of samples.colors.
    """
    colors = samples.colors if colors is None else colors
    keep = np.ones(len(samples.points), dtype=bool)
    if samples.closed:
        keep = np.einsum("nd,nd->n", samples.normals, samples.points - eye) < 0  # facing the camera
    X = world_to_camera(samples.points[keep], R, t)
    # How far a point is, as far as its size in the image is concerned:
    # its depth Z on an image plane, its distance from the axis on a cylinder.
    scale = np.hypot(X[:, 0], X[:, 2]) if cylinder else X[:, 2]
    front = scale > near
    X, scale, spacing, colors = X[front], scale[front], samples.spacing[keep][front], colors[keep][front]
    pixels = project_cylinder(X, K) if cylinder else project(X, K)
    # Every point large enough to leave no holes between neighbours: near points cover more
    # pixels, and so do points far off the axis, where the projection stretches surfaces.
    stretch = np.linalg.norm(X, axis=1) / scale
    size = np.clip(np.ceil(2.2 * spacing * K[0, 0] * stretch / scale), 1, 12).astype(np.int64)
    image = splat(pixels, scale, colors, width, height, size, background=-1.0)
    covered = image[..., 0] >= 0.0
    image[~covered] = 1.0
    return image, covered


def draw_segments(
    image: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    K: np.ndarray,
    color,
    cylinder: bool = False,
    pieces: int = 1,
    near: float = 0.05,
) -> None:
    """Draw 3D line segments (camera frame, (M, 3) each) into the image, in place.

    Segments are cut where they pass behind the camera. On a cylinder straight
    lines become curves, so every segment is drawn in `pieces` parts.
    """
    s = np.linspace(0.0, 1.0, pieces + 1)
    for a, b in zip(starts, ends):
        line = a + s[:, None] * (b - a)
        for p, q in zip(line[:-1], line[1:]):
            dp, dq = (np.hypot(x[0], x[2]) if cylinder else x[2] for x in (p, q))
            if dp < near and dq < near:
                continue
            if dp < near or dq < near:  # move the end behind the camera onto the near plane
                k = (near - dp) / (dq - dp)
                p, q = (p + k * (q - p), q) if dp < near else (p, p + k * (q - p))
            uv = project_cylinder(np.stack([p, q]), K) if cylinder else project(np.stack([p, q]), K)
            uv = clip_to_image(uv[0], uv[1], image.shape[1], image.shape[0])
            if uv is not None:
                draw_line(image, uv[0], uv[1], color)


def clip_to_image(a: np.ndarray, b: np.ndarray, width: int, height: int):
    """The part of the 2D segment a-b inside the image, or None (Liang-Barsky clipping)."""
    low, high = 0.0, 1.0
    d = b - a
    for p, q in [(-d[0], a[0]), (d[0], width - 1 - a[0]), (-d[1], a[1]), (d[1], height - 1 - a[1])]:
        if p == 0:
            if q < 0:
                return None
            continue
        k = q / p
        if p < 0:
            low = max(low, k)
        else:
            high = min(high, k)
        if low > high:
            return None
    return a + low * d, a + high * d


def show_camera(server, eye: np.ndarray, R: np.ndarray, K: np.ndarray, image: np.ndarray, cylinder: bool = False, size: float = 1.5) -> None:
    """Draw the camera in a viser scene: centre, image plane (or cylinder) with the image on it.

    The image plane is really at distance f (in pixels) from the centre. In the
    3D view it is drawn scaled down so that it is at most `size` metres wide;
    the angles, and so the frustum, stay the same.
    """
    import viser.transforms as tf

    height, width = image.shape[:2]
    image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    m = size / width  # metres per pixel in the 3D view
    rotation = tf.SO3.from_matrix(R.T)
    if not cylinder:
        def plane(u, v):
            return eye + m * (R.T @ [u - K[0, 2], v - K[1, 2], K[0, 0]])

        corners = [plane(u, v) for u, v in [(0, 0), (width, 0), (width, height), (0, height)]]
        edges = [[eye, c] for c in corners] + [[corners[k], corners[(k + 1) % 4]] for k in range(4)]
        edges += [[eye, plane(K[0, 2], K[1, 2])]]  # the optical axis
        server.scene.add_image(
            "/camera/image", image_u8, width * m, height * m, wxyz=rotation.wxyz,
            position=plane(width / 2, height / 2), cast_shadow=False,
        )
        for k in range(32):
            server.scene.remove_by_name(f"/camera/strip_{k}")
    else:
        # The image wrapped around a cylinder of radius f, drawn as narrow flat strips.
        server.scene.remove_by_name("/camera/image")
        strips = 32
        bounds = np.linspace(0, width, strips + 1)

        def cylinder_point(u, v):
            angle = (u - K[0, 2]) / K[0, 0]
            return eye + m * K[0, 0] * (R.T @ [np.sin(angle), (v - K[1, 2]) / K[0, 0], np.cos(angle)])

        edges = []
        for k in range(strips):
            u0, u1 = int(bounds[k]), int(bounds[k + 1])
            angle = ((u0 + u1) / 2 - K[0, 2]) / K[0, 0]
            turn = tf.SO3.from_y_radians(angle)
            server.scene.add_image(
                f"/camera/strip_{k}", image_u8[:, u0:u1], (u1 - u0) * m, height * m,
                wxyz=(rotation @ turn).wxyz, position=cylinder_point((u0 + u1) / 2, height / 2), cast_shadow=False,
            )
            for v in (0, height):
                edges.append([cylinder_point(u0, v), cylinder_point(u1, v)])
        edges += [[eye, cylinder_point(u, v)] for u in (0, width) for v in (0, height)]
        edges += [[cylinder_point(u, 0), cylinder_point(u, height)] for u in (0, width)]
        edges += [[eye, cylinder_point(K[0, 2], K[1, 2])]]
    server.scene.add_line_segments(
        "/camera/frustum", np.array(edges, dtype=np.float32), colors=(30, 30, 30), thickness=2.0, thickness_units="screen"
    )
