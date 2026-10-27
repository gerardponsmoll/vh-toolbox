"""Lecture 2: the pinhole camera.

A camera looks at an object. The 3D view shows the camera centre, its image
plane and the object; the panel shows the image this camera takes, drawn by
projecting points of the object with the functions in vh/camera.py. Move the
sliders to change where the camera is (extrinsics) and how it projects
(intrinsics). Click a point of the object to follow it through the projection.

    python demos/02_camera.py
    python demos/02_camera.py --mesh path/to/any_mesh.obj
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf

from vh import SMPLModel
from vh.camera import intrinsics, look_at, pixel_ray, project, world_to_camera
from vh.geometry import ray_mesh_intersection
from vh.render import draw_dots, draw_line, sample_surface, splat
from vh.scene import BODY_POSES, LIGHT, Shape, body, checker_box, merge, solid
from vh.viewer import configure

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

WIDTH, HEIGHT = 480, 360  # image size in pixels
PIXEL_SIZE = 0.00075  # metres; size of one pixel when the image plane is drawn in the 3D view
NEAR = 0.05  # metres; points closer to the camera than this are not drawn
POINTS_PER_M2 = 40_000  # how densely surfaces are sampled for the image
MAX_POINTS = 400_000
FLOOR = 4.0  # metres, side of the floor grid

PLAIN = np.array([120, 160, 210]) / 255.0
GRID_COLOR = np.array([0.72, 0.72, 0.72])
CAMERA_COLOR = (30, 30, 30)
RAY_COLOR = (240, 160, 0)
POINT_COLOR = (230, 30, 30)
MOVED_COLOR = (30, 110, 230)

PROJECTIONS = ["Perspective", "Weak perspective", "Both (weak perspective in red)"]


def colonnade(person: Shape | None = None) -> Shape:
    """A small scene with a lot of depth: two rows of columns leading to a house, trees, and a person.

    Parallel lines in depth (the rows of columns, the tiles) meet in a vanishing
    point, and things at very different distances make the dolly zoom easy to see.
    """
    stone, wall, roof, door = (0.90, 0.85, 0.72), (0.93, 0.90, 0.82), (0.72, 0.25, 0.20), (0.35, 0.22, 0.15)
    trunk, leaves = (0.45, 0.30, 0.18), (0.25, 0.55, 0.28)

    # Tiled ground, 6 m wide and 9 m deep. Only its top is kept.
    ground = checker_box((6.0, 0.1, 9.0), cell=0.5, base_color=(0.85, 0.82, 0.76))
    a, b, c = (ground.vertices[ground.faces[:, i]] for i in range(3))
    top = np.cross(b - a, c - a)[:, 1] > 0
    ground = Shape(ground.vertices + [0.0, -0.105, -1.0], ground.faces[top], ground.face_colors[top], detail=0.5)
    parts = [ground]

    for x in (-1.3, 1.3):
        for z in np.arange(-4.0, 2.5, 1.0):
            parts.append(solid(trimesh.creation.cylinder(radius=0.12, height=2.2, sections=24), stone, (x, 1.1, z), z_to_y=True))
            parts.append(solid(trimesh.creation.box((0.34, 0.08, 0.34)), stone, (x, 0.04, z)))
        parts.append(solid(trimesh.creation.box((0.34, 0.22, 6.5)), stone, (x, 2.31, -1.0)))

    # The house at the end: walls, a door, and a roof made of two sloping sides.
    parts.append(solid(trimesh.creation.box((2.0, 1.5, 1.2)), wall, (0.0, 0.75, -4.7)))
    parts.append(solid(trimesh.creation.box((0.5, 1.0, 0.06)), door, (0.0, 0.5, -4.08)))
    w, h, d = 1.2, 0.8, 0.7  # half width, height and half depth of the roof
    roof_vertices = np.array([[-w, 0, -d], [w, 0, -d], [0, h, -d], [-w, 0, d], [w, 0, d], [0, h, d]], dtype=np.float64)
    roof_faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], [1, 2, 5], [1, 5, 4]])
    parts.append(solid(trimesh.Trimesh(roof_vertices, roof_faces, process=False), roof, (0.0, 1.5, -4.7)))

    for x, z, pointed in [(-2.3, 1.6, True), (2.3, 0.4, False), (-2.4, -1.8, False), (2.3, -3.0, True)]:
        parts.append(solid(trimesh.creation.cylinder(radius=0.07, height=0.8, sections=12), trunk, (x, 0.4, z), z_to_y=True))
        if pointed:
            parts.append(solid(trimesh.creation.cone(radius=0.55, height=1.5, sections=24), leaves, (x, 0.7, z), z_to_y=True))
        else:
            parts.append(solid(trimesh.creation.icosphere(subdivisions=2, radius=0.6), leaves, (x, 1.3, z)))

    for x, z, radius, color in [(0.7, 1.3, 0.2, (0.85, 0.25, 0.2)), (-0.6, 2.2, 0.14, (0.2, 0.4, 0.85)), (0.5, -2.2, 0.25, (0.95, 0.75, 0.2))]:
        parts.append(solid(trimesh.creation.icosphere(subdivisions=2, radius=radius), color, (x, radius, z)))

    if person is not None:
        person.detail = 8.0  # the finest shape in the scene
        parts.append(person)
    return merge(parts, center=np.array([0.0, 0.9, 0.0]), view=(14.0, 12.0, 6.0))


def load_mesh(path: Path) -> Shape:
    """Any mesh file trimesh can read, scaled to 1.5 m and put on the floor. Assumes y is up."""
    mesh = trimesh.load(path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - [(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2]) * (1.5 / (high - low).max())
    if mesh.visual.kind in ("face", "vertex"):
        colors = np.asarray(mesh.visual.face_colors)[:, :3] / 255.0
    else:
        colors = np.tile(PLAIN, (len(mesh.faces), 1))
    return Shape(vertices, np.asarray(mesh.faces), colors, closed=bool(mesh.is_watertight))


def floor_grid_points(spacing: float = 0.01) -> np.ndarray:
    """Points along the lines of the floor grid, so that the floor shows up in the image."""
    along = np.arange(-FLOOR / 2, FLOOR / 2 + spacing, spacing)
    across = np.repeat(np.arange(-FLOOR / 2, FLOOR / 2 + 0.25, 0.5), len(along))
    along = np.tile(along, len(across) // len(along))
    zeros = np.zeros_like(along)
    return np.concatenate([np.stack([along, zeros, across], axis=1), np.stack([across, zeros, along], axis=1)])


def matrix_text(name: str, M: np.ndarray, digits: int) -> str:
    rows = ["  ".join(f"{value:9.{digits}f}" for value in row) for row in np.atleast_2d(M)]
    pad = " " * (len(name) + 3)
    return "\n".join((f"{name} = " if k == 0 else pad) + row for k, row in enumerate(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--mesh", type=Path, help="A mesh of your own (obj, ply, stl, ...), y axis up.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Objects are built when they are selected.
    builders = {}
    if args.mesh is not None:
        builders[args.mesh.name] = lambda: load_mesh(args.mesh)
    model_paths = sorted(args.model_dir.glob("smpl_*.npz"))
    if model_paths:
        path = next((p for p in model_paths if p.stem == "smpl_neutral"), model_paths[0])
        model = SMPLModel.load(path)
        builders["Scene"] = lambda: colonnade(body(model, BODY_POSES[2]))
        builders["SMPL body"] = lambda: body(model, gui_pose.value)
    else:
        print(f"No SMPL model in {args.model_dir}: showing the other objects only (see README).")
        builders["Scene"] = colonnade
    builders["Cube"] = lambda: checker_box((1.0, 1.0, 1.0))
    builders["Checkerboard"] = lambda: checker_box((2.0, 2.0, 0.04))

    server = viser.ViserServer(port=args.port)
    configure(server)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/floor", width=FLOOR, height=FLOOR, plane="xz", cell_size=0.5, section_size=0.5)

    server.initial_camera.position = (5.5, 3.2, 8.0)
    server.initial_camera.look_at = (0.5, 0.9, 1.0)

    lock = threading.Lock()
    syncing = False  # True while we set sliders from code, so their callbacks stay quiet
    playing = False

    # ---------------------------------------------------------------- GUI
    gui_image = server.gui.add_image(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), label="Camera image")
    gui_look = server.gui.add_button("Look through the camera", hint="Move the 3D view to the camera.")

    with server.gui.add_folder("Object"):
        gui_object = server.gui.add_dropdown("Object", list(builders), initial_value=next(iter(builders)))
        gui_pose = server.gui.add_dropdown("Pose", BODY_POSES, initial_value=BODY_POSES[0])

    with server.gui.add_folder("Extrinsics [R | t]: where the camera is"):
        gui_azimuth = server.gui.add_slider("Azimuth (deg)", min=-180.0, max=180.0, step=1.0, initial_value=25.0)
        gui_elevation = server.gui.add_slider("Elevation (deg)", min=-10.0, max=85.0, step=1.0, initial_value=10.0)
        gui_distance = server.gui.add_slider(
            "Distance (m)", min=0.8, max=12.0, step=0.05, initial_value=3.0, hint="From the camera to the object's centre."
        )

    with server.gui.add_folder("Intrinsics K: how it projects"):
        gui_f = server.gui.add_slider(
            "Focal length f (px)",
            min=100.0,
            max=3000.0,
            step=10.0,
            initial_value=500.0,
            hint="Distance from the camera centre to the image plane, in pixels.",
        )
        gui_cx = server.gui.add_slider("Principal point cx", min=0.0, max=float(WIDTH), step=1.0, initial_value=WIDTH / 2)
        gui_cy = server.gui.add_slider("Principal point cy", min=0.0, max=float(HEIGHT), step=1.0, initial_value=HEIGHT / 2)
        gui_k1 = server.gui.add_slider(
            "Lens distortion k1",
            min=-0.5,
            max=0.5,
            step=0.01,
            initial_value=0.0,
            hint="Radial distortion. 0 is an ideal pinhole camera.",
        )
        gui_reset = server.gui.add_button("Reset camera")

    with server.gui.add_folder("Projection"):
        gui_projection = server.gui.add_dropdown("Model", PROJECTIONS, initial_value=PROJECTIONS[0])

    with server.gui.add_folder("Dolly zoom"):
        gui_dolly = server.gui.add_checkbox(
            "Keep the object's size",
            False,
            hint="Ties focal length to distance (f / distance stays fixed), so the object keeps its size in the image.",
        )
        gui_play = server.gui.add_button("Play", hint="Move the camera away and back while zooming in.")

    with server.gui.add_folder("A point and its ray"):
        gui_along = server.gui.add_slider(
            "Move the point along its ray",
            min=0.3,
            max=2.0,
            step=0.01,
            initial_value=1.0,
            hint="1 is the point you clicked (red). The blue point is at this multiple of its distance.",
        )
    gui_numbers = server.gui.add_markdown("")

    # --------------------------------------------------------------- state
    shape: Shape = None  # the object in the scene
    points = normals = colors = None  # points sampled on it, for the image
    point_size = None  # per sampled point: typical distance to its neighbours, in metres
    target = np.zeros(3)  # what the camera looks at: the centre of the object
    selected = (0, np.full(3, 1 / 3))  # the clicked point, as (triangle, barycentric coordinates)
    grid_points = floor_grid_points()
    dolly_ratio = gui_f.value / gui_distance.value
    free_pose = None  # (eye, R) after the camera was dragged in the 3D view; None: use the sliders
    last_handle_use = 0.0  # time of the last interaction with the camera handle
    shown_object = None  # name of the object the camera position was last set for

    # Arrows and rings on the camera centre: drag them to move and rotate the camera.
    handle = server.scene.add_transform_controls("/camera_handle", scale=0.3, disable_sliders=True, depth_test=False)

    def selected_point() -> np.ndarray:
        face, barycentric = selected
        return barycentric @ shape.vertices[shape.faces[face]]

    def camera() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Eye, R, t and K. With the sliders the camera orbits the object and looks at its centre."""
        K = intrinsics(gui_f.value, gui_cx.value, gui_cy.value)
        if free_pose is not None:
            eye, R = free_pose
            return eye, R, -R @ eye, K
        azimuth, elevation = np.radians(gui_azimuth.value), np.radians(gui_elevation.value)
        direction = [np.sin(azimuth) * np.cos(elevation), np.sin(elevation), np.cos(azimuth) * np.cos(elevation)]
        eye = target + gui_distance.value * np.array(direction)
        R, t = look_at(eye, target)
        return eye, R, t, K

    def on_image_plane(pixel, eye: np.ndarray, R: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Where a pixel is in the world, on the image plane drawn at distance f in front of the centre."""
        return eye + PIXEL_SIZE * (R.T @ [pixel[0] - K[0, 2], pixel[1] - K[1, 2], K[0, 0]])

    # ------------------------------------------------------------- drawing
    def load_object() -> None:
        """Build the selected object, sample points on it and put it into the 3D view."""
        nonlocal shape, points, normals, colors, point_size, target, selected
        shape = builders[gui_object.value]()
        a, b, c = (shape.vertices[shape.faces[:, i]] for i in range(3))
        face_normals = np.cross(b - a, c - a)
        area = np.linalg.norm(face_normals, axis=1)
        face_normals = face_normals / np.maximum(area, 1e-12)[:, None]

        count = int(np.clip(0.5 * area.sum() * POINTS_PER_M2, 20_000, MAX_POINTS))
        detail = np.broadcast_to(shape.detail, len(shape.faces))
        face, barycentric = sample_surface(shape.vertices, shape.faces, count, weights=detail)
        points = np.einsum("nk,nkd->nd", barycentric, shape.vertices[shape.faces[face]])
        normals = face_normals[face]
        # Typical distance between neighbouring samples: smaller where the sampling is denser.
        point_size = np.sqrt(0.5 * (area * detail).sum() / (count * detail[face]))
        towards_light = normals @ LIGHT
        shade = 0.7 + 0.3 * (np.clip(towards_light, 0.0, 1.0) if shape.closed else np.abs(towards_light))
        colors = shape.face_colors[face] * shade[:, None]

        target = shape.center
        if target is None:
            target = (shape.vertices.min(axis=0) + shape.vertices.max(axis=0)) / 2
        # Start with the point closest to the default camera position selected.
        front = int((a + b + c)[:, 2].argmax())
        if shape.center is not None:  # in a scene: a point on what the camera looks at
            front = int(np.linalg.norm((a + b + c) / 3 - (shape.center + [0.0, 0.4, 0.3]), axis=1).argmin())
        selected = (front, np.full(3, 1 / 3))

        server.scene.add_mesh_trimesh("/object", shape.trimesh())

    def draw(move_handle: bool = True) -> None:
        """Take the picture and update the camera in the 3D view."""
        nonlocal syncing
        with lock:
            eye, R, t, K = camera()
            f, k1 = gui_f.value, gui_k1.value
            distance = max(float(R[2] @ target + t[2]), 2 * NEAR)  # depth of the object's centre
            mode = PROJECTIONS.index(gui_projection.value)
            weak_depth = distance if mode == 1 else None  # the object's centre is at this depth

            # --- the image: project the sampled points, nearest point wins
            visible = np.ones(len(points), dtype=bool)
            if shape.closed:
                visible = np.einsum("nd,nd->n", normals, points - eye) < 0  # facing the camera
            X = np.concatenate([world_to_camera(points[visible], R, t), world_to_camera(grid_points, R, t)])
            rgb = np.concatenate([colors[visible], np.tile(GRID_COLOR, (len(grid_points), 1))])
            spacing = point_size[visible]
            drawable = X[:, 2] > NEAR
            if k1 < 0:
                # Far from the centre the distortion polynomial folds the image back
                # onto itself. A real lens does not do that: leave those points out.
                radius2 = ((X[:, :2] / np.where(drawable, X[:, 2], 1.0)[:, None]) ** 2).sum(axis=1)
                drawable &= 1.0 + 3.0 * k1 * radius2 > 0
            X, rgb = X[drawable], rgb[drawable]
            num_object = int(drawable[: visible.sum()].sum())  # the rest are floor points
            spacing = spacing[drawable[: visible.sum()]]

            pixels = project(X, K, k1, depth=weak_depth)
            # Draw every point large enough to leave no holes between neighbours:
            # near points cover more pixels than far ones.
            point_depth = distance if mode == 1 else X[:num_object, 2]
            size = np.clip(np.ceil(2.2 * spacing * f / point_depth), 1, 9).astype(np.int64)
            image = splat(pixels[:num_object], X[:num_object, 2], rgb[:num_object], WIDTH, HEIGHT, size)
            floor = splat(pixels[num_object:], X[num_object:, 2], rgb[num_object:], WIDTH, HEIGHT, 1)
            behind_object = (image == 1.0).all(axis=2)
            image[behind_object] = floor[behind_object]

            perspective = project(X[:num_object], K, k1)
            weak = project(X[:num_object], K, k1, depth=distance)
            difference = np.linalg.norm(perspective - weak, axis=1).mean() if num_object else 0.0
            if mode == 2:
                draw_dots(image, weak[:: max(1, num_object // 1500)], (0.9, 0.1, 0.1), 1)

            # --- the selected point, and the same point moved along its ray
            X_world = selected_point()
            moved_world = eye + gui_along.value * (X_world - eye)
            X_cam, moved_cam = world_to_camera(np.stack([X_world, moved_world]), R, t)
            point_pixel = moved_pixel = None
            if X_cam[2] > NEAR:
                point_pixel = project(X_cam[None], K, k1, depth=weak_depth)[0]
                moved_pixel = project(moved_cam[None], K, k1, depth=weak_depth)[0]
                draw_dots(image, moved_pixel, np.array(MOVED_COLOR) / 255.0, 9)
                for du, dv in [(12, 0), (0, 12)]:  # a cross on the selected point
                    draw_line(image, point_pixel - [du, dv], point_pixel + [du, dv], np.array(POINT_COLOR) / 255.0, 2)
            cross = np.array([gui_cx.value, gui_cy.value])  # a small mark on the principal point
            for du, dv in [(5, 0), (0, 5)]:
                draw_line(image, cross - [du, dv], cross + [du, dv], (0.0, 0.0, 0.0))
            image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            gui_image.image = image_u8

            # --- the camera in the 3D view: centre, image plane at distance f, and the image on it
            corners = np.array(
                [on_image_plane(p, eye, R, K) for p in [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]]
            )
            principal = on_image_plane((K[0, 2], K[1, 2]), eye, R, K)
            edges = [[eye, corner] for corner in corners]
            edges += [[corners[k], corners[(k + 1) % 4]] for k in range(4)]
            edges += [[eye, principal]]  # the optical axis
            server.scene.add_line_segments(
                "/camera/frustum", np.array(edges, dtype=np.float32), colors=CAMERA_COLOR, thickness=2.0, thickness_units="screen"
            )
            wxyz = tf.SO3.from_matrix(R.T).wxyz
            if move_handle:  # left alone while the user is dragging it
                syncing = True
                try:
                    handle.position, handle.wxyz = eye, wxyz
                finally:
                    syncing = False
            server.scene.add_image(
                "/camera/image",
                image_u8,
                render_width=WIDTH * PIXEL_SIZE,
                render_height=HEIGHT * PIXEL_SIZE,
                wxyz=wxyz,
                position=on_image_plane((WIDTH / 2, HEIGHT / 2), eye, R, K),
                cast_shadow=False,
            )

            # --- the ray through the selected point
            ray_end = eye + gui_along.max * (X_world - eye)
            server.scene.add_line_segments(
                "/ray/line", np.array([[eye, ray_end]], dtype=np.float32), colors=RAY_COLOR, thickness=3.0, thickness_units="screen"
            )
            server.scene.add_icosphere("/ray/point", radius=0.02, color=POINT_COLOR, position=X_world)
            server.scene.add_icosphere("/ray/moved", radius=0.02, color=MOVED_COLOR, position=moved_world)
            if X_cam[2] > NEAR:  # where the ray crosses the image plane (of an ideal pinhole camera)
                crossing = on_image_plane(project(X_cam[None], K)[0], eye, R, K)
                server.scene.add_icosphere("/ray/crossing", radius=0.006, color=POINT_COLOR, position=crossing)

            # --- the numbers
            fov = np.degrees(2 * np.arctan(np.array([WIDTH, HEIGHT]) / (2 * f)))
            lines = [
                matrix_text("K", K, 1),
                "",
                matrix_text("R", R, 3),
                "",
                matrix_text("t", t, 3),
                "",
                f"field of view   {fov[0]:.0f} x {fov[1]:.0f} deg",
                f"perspective vs weak perspective: {difference:.1f} px on average",
                "",
                "red point",
                f"  world   X   = ({X_world[0]:6.2f}, {X_world[1]:6.2f}, {X_world[2]:6.2f}) m",
                f"  camera  X_c = ({X_cam[0]:6.2f}, {X_cam[1]:6.2f}, {X_cam[2]:6.2f}) m",
            ]
            if point_pixel is not None:
                lines += [
                    f"  pixel       = ({point_pixel[0]:6.1f}, {point_pixel[1]:6.1f})",
                    f"blue point, at depth {moved_cam[2]:.2f} m",
                    f"  pixel       = ({moved_pixel[0]:6.1f}, {moved_pixel[1]:6.1f})",
                ]
            else:
                lines += ["  behind the camera: no image"]
            gui_numbers.content = "```\n" + "\n".join(lines) + "\n```"

    # ----------------------------------------------------------- callbacks
    def set_value(slider, value: float) -> None:
        nonlocal syncing
        syncing = True
        try:
            slider.value = float(np.clip(value, slider.min, slider.max))
        finally:
            syncing = False

    def on_change(_) -> None:
        if not syncing:
            draw()

    def on_orbit(_) -> None:
        """The orbit sliders put the camera back on its orbit, looking at the object."""
        nonlocal free_pose
        if syncing:
            return
        free_pose = None
        draw()

    def on_handle(_) -> None:
        """The camera was dragged or rotated in the 3D view: take its pose from the handle."""
        nonlocal free_pose, last_handle_use
        last_handle_use = time.time()
        if syncing:
            return
        eye = np.array(handle.position, dtype=np.float64)
        free_pose = (eye, tf.SO3(np.array(handle.wxyz)).as_matrix().T)
        # Keep the sliders in step with where the camera now is.
        offset = eye - target
        distance = float(np.linalg.norm(offset))
        set_value(gui_distance, distance)
        set_value(gui_azimuth, np.degrees(np.arctan2(offset[0], offset[2])))
        set_value(gui_elevation, np.degrees(np.arcsin(offset[1] / max(distance, 1e-9))))
        draw(move_handle=False)

    def on_distance(_) -> None:
        nonlocal free_pose
        if syncing:
            return
        free_pose = None
        if gui_dolly.value:
            set_value(gui_f, dolly_ratio * gui_distance.value)
        draw()

    def on_f(_) -> None:
        nonlocal free_pose
        if syncing:
            return
        if gui_dolly.value:
            free_pose = None
            set_value(gui_distance, gui_f.value / dolly_ratio)
        draw()

    def on_dolly(_) -> None:
        nonlocal dolly_ratio
        dolly_ratio = gui_f.value / gui_distance.value

    def on_play(_) -> None:
        """Move the camera away and back, changing f so that the object keeps its size."""
        nonlocal playing, free_pose
        if playing:
            return
        playing = True
        free_pose = None
        try:
            ratio = gui_f.value / gui_distance.value
            start = gui_distance.value
            near, far = max(gui_distance.min, gui_f.min / ratio), min(8.0, gui_f.max / ratio)
            path = np.concatenate([np.linspace(start, near, 15), np.linspace(near, far, 50), np.linspace(far, start, 30)])
            for distance in path:
                set_value(gui_distance, distance)
                set_value(gui_f, ratio * distance)
                draw()
                time.sleep(0.02)
        finally:
            playing = False

    def on_object(_) -> None:
        nonlocal free_pose, shown_object
        free_pose = None
        gui_pose.visible = gui_object.value == "SMPL body"
        with lock:
            load_object()
        if gui_object.value != shown_object:  # a new object: start from its own camera position
            shown_object = gui_object.value
            for slider, value in zip([gui_azimuth, gui_elevation, gui_distance], shape.view):
                set_value(slider, value)
            on_dolly(None)
        draw()

    def on_reset(_) -> None:
        nonlocal free_pose
        free_pose = None
        for slider, value in [
            (gui_azimuth, shape.view[0]),
            (gui_elevation, shape.view[1]),
            (gui_distance, shape.view[2]),
            (gui_f, 500.0),
            (gui_cx, WIDTH / 2),
            (gui_cy, HEIGHT / 2),
            (gui_k1, 0.0),
            (gui_along, 1.0),
        ]:
            set_value(slider, value)
        on_dolly(None)
        draw()

    def on_look(event) -> None:
        """Put the viewer where the camera is, with the same field of view."""
        eye, R, _, _ = camera()
        clients = [event.client] if event.client is not None else list(server.get_clients().values())
        for client in clients:
            with client.atomic():
                client.camera.up_direction = (0.0, 1.0, 0.0)
                client.camera.position = eye
                client.camera.look_at = eye + R[2]  # one metre along the viewing direction
                client.camera.fov = float(2 * np.arctan(HEIGHT / (2 * gui_f.value)))

    @server.scene.on_click()
    def on_scene_click(event) -> None:
        """Select the point of the object under the cursor."""
        nonlocal selected
        if time.time() - last_handle_use < 0.3:  # the click was on the camera handle
            return
        direction = np.asarray(event.ray_direction, dtype=np.float64)
        hit = ray_mesh_intersection(event.ray_origin, direction / np.linalg.norm(direction), shape.vertices, shape.faces)
        if hit is not None:
            selected = hit
            set_value(gui_along, 1.0)
            draw()

    for slider in [gui_cx, gui_cy, gui_k1, gui_projection, gui_along]:
        slider.on_update(on_change)
    gui_azimuth.on_update(on_orbit)
    gui_elevation.on_update(on_orbit)
    handle.on_update(on_handle)
    gui_distance.on_update(on_distance)
    gui_f.on_update(on_f)
    gui_dolly.on_update(on_dolly)
    gui_object.on_update(on_object)
    gui_pose.on_update(on_object)
    gui_play.on_click(on_play)
    gui_reset.on_click(on_reset)
    gui_look.on_click(on_look)

    on_object(None)
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
