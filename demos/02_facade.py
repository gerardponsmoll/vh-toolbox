"""Lecture 2: should the far ends of a long façade be drawn smaller than its centre?

A camera stands in front of a long row of identical columns. The end columns are
much farther from the camera than the middle one, yet in the image they are
exactly as tall. In a pinhole image the size of an object is f * H / Z, and Z
is the depth along the optical axis, not the distance to the camera. All
columns of a façade parallel to the image plane have the same depth.

Turn the camera and the depths differ: now the columns shrink towards one end.
Switch the image surface to a cylinder, as in a stitched panorama photo, and
the size follows the distance instead: the ends shrink and straight lines bend.

    python demos/02_facade.py
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import trimesh
import viser

from vh.camera import intrinsics, look_at, project, project_cylinder, world_to_camera
from vh.render import draw_line
from vh.scene import clip_to_image, draw_segments, ground, merge, sample, show_camera, solid, take_picture

WIDTH, HEIGHT = 640, 300  # image size in pixels
EYE_HEIGHT = 1.6  # metres
COLUMN_X = np.arange(-30.0, 30.01, 2.5)  # where the columns stand, on the line z = 0
COLUMN_HEIGHT, COLUMN_RADIUS = 4.0, 0.25
NUM_POINTS = 800_000

SURFACES = ["Flat image plane (pinhole camera)", "Cylinder (stitched panorama)"]
STONE, WALL, SELECTED = (0.88, 0.84, 0.72), (0.80, 0.74, 0.62), np.array([0.85, 0.2, 0.15])
DISTANCE_COLOR, DEPTH_COLOR = (240, 140, 0), (40, 100, 220)


def facade():
    """A long row of columns with a beam on top and a wall behind them, all along x."""
    parts = []
    for x in COLUMN_X:
        column = solid(trimesh.creation.cylinder(radius=COLUMN_RADIUS, height=COLUMN_HEIGHT, sections=32), STONE, (x, COLUMN_HEIGHT / 2, 0.0), z_to_y=True)
        column.detail = 3.0  # the columns are what we look at: sample them densely
        parts.append(column)
    length = COLUMN_X[-1] - COLUMN_X[0] + 1.5
    parts.append(solid(trimesh.creation.box((length, 0.6, 0.8)), STONE, (0.0, COLUMN_HEIGHT + 0.3, 0.0)))
    parts.append(solid(trimesh.creation.box((length, COLUMN_HEIGHT + 0.6, 0.2)), WALL, (0.0, (COLUMN_HEIGHT + 0.6) / 2, -1.2)))
    return merge(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    shape = facade()
    samples = sample(shape, NUM_POINTS)
    # Which column every sample belongs to (-1: none), to colour the selected one.
    near_column = np.abs(samples.points[:, None, 0] - COLUMN_X[None]) < COLUMN_RADIUS + 0.01
    near_column &= (np.abs(samples.points[:, 2]) < COLUMN_RADIUS + 0.01)[:, None]
    near_column &= (samples.points[:, 1] < COLUMN_HEIGHT - 0.01)[:, None]
    column_of = np.where(near_column.any(axis=1), near_column.argmax(axis=1), -1)

    # Ground lines: tiles of 2.5 m, lines along x (parallel to the façade) and along z.
    xs, zs = np.arange(-35.0, 35.01, 2.5), np.arange(-1.25, 30.01, 2.5)
    line_starts = np.array([[x, 0.0, zs[0]] for x in xs] + [[xs[0], 0.0, z] for z in zs])
    line_ends = np.array([[x, 0.0, zs[-1]] for x in xs] + [[xs[-1], 0.0, z] for z in zs])

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_mesh_trimesh("/facade", shape.trimesh())
    server.scene.add_grid("/ground", width=70.0, height=31.25, plane="xz", cell_size=2.5, section_size=2.5, position=(0.0, 0.0, 14.375))
    server.initial_camera.position = (7.0, 9.0, 16.0)
    server.initial_camera.look_at = (1.0, 1.0, 3.0)
    lock = threading.Lock()

    gui_image = server.gui.add_image(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), label="Camera image")
    with server.gui.add_folder("Camera"):
        gui_distance = server.gui.add_slider("Distance to the façade (m)", min=2.0, max=20.0, step=0.1, initial_value=6.0)
        gui_turn = server.gui.add_slider(
            "Turn the camera (deg)", min=-60.0, max=60.0, step=1.0, initial_value=0.0,
            hint="0: the image plane is parallel to the façade.",
        )
        gui_f = server.gui.add_slider("Focal length f (px)", min=60.0, max=800.0, step=5.0, initial_value=140.0)
        gui_surface = server.gui.add_dropdown("Image surface", SURFACES, initial_value=SURFACES[0])
    with server.gui.add_folder("Measure"):
        gui_column = server.gui.add_slider(
            "Column (red)", min=0, max=len(COLUMN_X) - 1, step=1, initial_value=len(COLUMN_X) // 2 + 4,
            hint="Its distance (orange) and depth (blue) are drawn in the 3D view.",
        )
        gui_measure = server.gui.add_checkbox("Show distance and depth", True)
    gui_numbers = server.gui.add_markdown("")

    def draw(_=None) -> None:
        with lock:
            turn = np.radians(gui_turn.value)
            eye = np.array([0.0, EYE_HEIGHT, gui_distance.value])
            forward = np.array([np.sin(turn), 0.0, -np.cos(turn)])
            R, t = look_at(eye, eye + forward)
            K = intrinsics(gui_f.value, WIDTH / 2, HEIGHT / 2)
            cylinder = gui_surface.value == SURFACES[1]
            chosen = int(gui_column.value)

            # --- the image
            colors = samples.colors.copy()
            colors[column_of == chosen] = SELECTED * (0.7 + 0.3 * samples.colors[column_of == chosen].mean(axis=1, keepdims=True))
            image, covered = take_picture(samples, eye, R, t, K, WIDTH, HEIGHT, cylinder=cylinder, colors=colors)
            lines = np.ones_like(image)
            draw_segments(lines, world_to_camera(line_starts, R, t), world_to_camera(line_ends, R, t), K, (0.6, 0.6, 0.6), cylinder, pieces=40)
            image[~covered] = lines[~covered]

            # Top and bottom of every column, and their height in the image.
            ends = np.array([[[x, 0.0, 0.0], [x, COLUMN_HEIGHT, 0.0]] for x in COLUMN_X]).reshape(-1, 3)
            X = world_to_camera(ends, R, t)
            pixels = project_cylinder(X, K) if cylinder else project(X, K)
            visible = (np.hypot(X[:, 0], X[:, 2]) if cylinder else X[:, 2]).reshape(-1, 2).min(axis=1) > 0.05
            heights = np.where(visible, pixels[0::2, 1] - pixels[1::2, 1], np.nan)
            # Two guide lines through the top and bottom of the middle column.
            middle = len(COLUMN_X) // 2
            if visible[middle]:
                for v in pixels[2 * middle : 2 * middle + 2, 1]:
                    for u in range(0, WIDTH, 12):
                        draw_line(image, (u, v), (u + 6, v), (0.85, 0.2, 0.15))
            # The vanishing point of the façade's horizontal lines, K R d with d along the row,
            # and the lines through the column tops and feet that meet there.
            direction = R @ [1.0, 0.0, 0.0]
            if not cylinder and abs(direction[2]) > 1e-3:
                direction = direction if direction[2] > 0 else -direction
                vanishing = (K @ direction)[:2] / direction[2]
                near_end = int(np.argmin(np.where(visible, X[0::2, 2], np.inf)))  # the nearest column
                for p in pixels[2 * near_end : 2 * near_end + 2]:
                    segment = clip_to_image(p, vanishing, WIDTH, HEIGHT)
                    if segment is not None:
                        step = (segment[1] - segment[0]) / 40
                        for k in range(40):  # dashed
                            draw_line(image, segment[0] + k * step, segment[0] + (k + 0.5) * step, (0.2, 0.4, 0.85))
                if 0 <= vanishing[0] < WIDTH and 0 <= vanishing[1] < HEIGHT:
                    for du, dv in [(8, 8), (8, -8)]:
                        draw_line(image, vanishing - [du, dv], vanishing + [du, dv], (0.2, 0.4, 0.85), 2)
            gui_image.image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            show_camera(server, eye, R, K, image, cylinder=cylinder)

            # --- distance and depth of the selected column and of the middle one, at eye height
            centres = np.stack([COLUMN_X, np.full_like(COLUMN_X, EYE_HEIGHT), np.zeros_like(COLUMN_X)], axis=1)
            distance = np.linalg.norm(centres - eye, axis=1)
            depth = (centres - eye) @ forward
            for name, k in [("chosen", chosen), ("middle", middle)]:
                show = gui_measure.value and not (name == "middle" and k == chosen)
                foot = eye + depth[k] * forward  # on the optical axis, at the column's depth
                segments = [[eye, centres[k]], [eye, foot], [foot, centres[k]]]
                colors3d = [DISTANCE_COLOR, DEPTH_COLOR, (150, 150, 150)]
                for i, (segment, color) in enumerate(zip(segments, colors3d)):
                    server.scene.add_line_segments(
                        f"/measure/{name}/line{i}", np.array([segment], dtype=np.float32), colors=color,
                        thickness=4.0 if i < 2 else 1.5, thickness_units="screen", visible=show,
                    )
                text = f"distance {distance[k]:.1f} m" if name == "chosen" else f"middle column: distance {distance[k]:.1f} m"
                server.scene.add_label(
                    f"/measure/{name}/distance", text,
                    position=eye + (0.5 if name == "chosen" else 0.8) * (centres[k] - eye) + [0.0, 0.25 if name == "chosen" else 0.6, 0.0],
                    visible=show,
                )
                server.scene.add_label(
                    f"/measure/{name}/depth", f"depth Z {depth[k]:.1f} m",
                    position=eye + 0.4 * (foot - eye) + [0.0, -0.35, 0.0], visible=show and name == "chosen",
                )
            # The plane of all points at the chosen column's depth: parallel to the image plane.
            right = np.cross(forward, [0.0, 1.0, 0.0])
            centre = eye + depth[chosen] * forward
            corners = np.array([centre + a * 34.0 * right + [0.0, y - EYE_HEIGHT, 0.0] for a, y in [(-1, 0), (1, 0), (1, 5), (-1, 5)]])
            server.scene.add_mesh_simple(
                "/measure/depth_plane", corners.astype(np.float32), np.array([[0, 1, 2], [0, 2, 3]]),
                color=DEPTH_COLOR, opacity=0.15, side="double", cast_shadow=False, receive_shadow=False, visible=gui_measure.value,
            )

            # --- the numbers
            size_rule = "f H / distance" if cylinder else "f H / Z"
            rows = ["| column | distance (m) | depth Z (m) | height in the image (px) |", "|---:|---:|---:|---:|"]
            for k in range(len(COLUMN_X)):
                height = "behind the camera" if np.isnan(heights[k]) else f"{heights[k]:.1f}"
                row = f"| {k} | {distance[k]:.2f} | {depth[k]:.2f} | {height} |"
                rows.append(row.replace("| ", "| **", 1).replace(" |", "** |", 1) if k == chosen else row)
            gui_numbers.content = "\n".join(
                [
                    f"Columns are {COLUMN_HEIGHT:.0f} m tall. Height in the image = **{size_rule}** "
                    f"(f = {gui_f.value:.0f} px). The dashed red lines mark the middle column. Turn the camera to see the "
                    f"vanishing point of the row (blue cross) where the dashed blue lines meet.",
                    "",
                    *rows,
                ]
            )

    for control in [gui_distance, gui_turn, gui_f, gui_surface, gui_column, gui_measure]:
        control.on_update(draw)
    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
