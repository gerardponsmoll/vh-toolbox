"""Lecture 2: the thin (Gaussian) lens, focus and blur.

A ball stands in front of a tree, as in a photograph with the subject in focus
and the background blurred. Rays from a point on each object fill the lens and
meet again at the distance z' given by Gauss's lens formula 1/z + 1/z' = 1/f.
The ball's rays meet on the sensor: it is sharp. The tree's rays meet in front
of the sensor and have spread out again when they reach it: the tree is
blurred. Move the sensor, or change f, to focus on the tree instead.

The lens is drawn much larger than a real one so that the rays can be seen.

    python demos/02_lens.py
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import trimesh
import viser

from vh.camera import intrinsics, look_at
from vh.lens import blur_diameter, disc_blur, image_distance
from vh.scene import merge, sample, solid, take_picture
from vh.viewer import configure

WIDTH, HEIGHT = 360, 240  # camera image size in pixels
SENSOR_W, SENSOR_H = 0.9, 0.6  # sensor size, in the units of the scene
PIXEL = SENSOR_W / WIDTH
APERTURE = 0.5  # lens diameter
FACING_LENS = (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0)  # turns an image plane to face +x
BALL_COLOR, TRUNK, LEAVES = (0.85, 0.25, 0.20), (0.45, 0.30, 0.18), (0.25, 0.55, 0.28)

# The camera looks along -x from the lens centre; the objects stand in front of it.
BALL_CENTER, BALL_RADIUS = np.array([-2.5, -0.12, 0.2]), 0.25
TREE_X, TREE_W, CROWN_Y, CROWN_RADIUS = -6.0, -0.35, 0.35, 0.6


def make_objects() -> list[dict]:
    """Each object: its mesh, points for the camera image, and the point whose rays are drawn
    (the point of the object nearest the lens)."""
    ball = solid(trimesh.creation.icosphere(subdivisions=3, radius=BALL_RADIUS), BALL_COLOR, BALL_CENTER)
    trunk = solid(trimesh.creation.cylinder(radius=0.1, height=0.9, sections=16), TRUNK, (TREE_X, -0.55, TREE_W), z_to_y=True)
    crown = solid(trimesh.creation.icosphere(subdivisions=3, radius=CROWN_RADIUS), LEAVES, (TREE_X, CROWN_Y, TREE_W))
    tree = merge([trunk, crown])
    return [
        dict(name="tree", shape=tree, samples=sample(tree, 150_000), color=LEAVES,
             point=np.array([TREE_X + CROWN_RADIUS, CROWN_Y, TREE_W])),
        dict(name="ball", shape=ball, samples=sample(ball, 150_000, seed=1), color=BALL_COLOR,
             point=BALL_CENTER + [BALL_RADIUS, 0.0, 0.0]),
    ]


def camera_image(objects: list[dict], s: float, f: float) -> np.ndarray:
    """The picture on the sensor, shown upright: each object seen through the lens
    centre (a pinhole at distance s), then blurred by its own disc, far to near."""
    R, t = look_at(np.zeros(3), [-1.0, 0.0, 0.0])
    K = intrinsics(s / PIXEL, WIDTH / 2, HEIGHT / 2)
    image = np.ones((HEIGHT, WIDTH, 3))
    for obj in objects:
        picture, covered = take_picture(obj["samples"], np.zeros(3), R, t, K, WIDTH, HEIGHT)
        alpha = covered[..., None].astype(np.float64)
        radius = blur_diameter(APERTURE, s, f, -obj["point"][0]) / PIXEL / 2
        layer = disc_blur(np.concatenate([picture * alpha, alpha], axis=2), radius)
        image = image * (1.0 - layer[..., 3:]) + layer[..., :3]
    return np.clip(image, 0.0, 1.0)


def rgb(color) -> tuple[int, int, int]:
    return tuple(int(255 * c) for c in color)


def where(z: float, s: float, f: float) -> str:
    """Where the rays from depth z meet, in words."""
    if z <= f:
        return "never meet (closer than f)"
    z_image = float(image_distance(f, z))
    if blur_diameter(APERTURE, s, f, z) < 2 * PIXEL:  # less than two pixels: we see it as sharp
        return f"meet **on the sensor** (z' = {z_image:.2f}): sharp"
    side = "behind" if z_image > s else "in front of"
    return f"meet **{side} the sensor** (z' = {z_image:.2f}): blurred"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    objects = make_objects()
    server = viser.ViserServer(port=args.port)
    configure(server)
    server.scene.set_up_direction("+y")
    # Seen from the side, a little behind the lens.
    server.initial_camera.position = (2.0, 0.9, 5.0)
    server.initial_camera.look_at = (-0.6, 0.0, 0.0)
    for obj in objects:
        server.scene.add_mesh_trimesh(f"/objects/{obj['name']}", obj["shape"].trimesh())
        server.scene.add_icosphere(f"/points/{obj['name']}", radius=0.03, color=(20, 20, 20), position=obj["point"])
    server.scene.add_line_segments("/axis", np.array([[[-6.5, 0, 0], [3.0, 0, 0]]]), colors=(170, 170, 170),
                                   thickness=1.0, thickness_units="screen")
    server.scene.add_icosphere("/lens", radius=1.0, color=(150, 200, 235), opacity=0.5, subdivisions=4,
                               scale=(0.04, APERTURE / 2, APERTURE / 2))
    server.scene.add_label("/labels/lens", "lens", position=(0.0, APERTURE / 2 + 0.12, 0.0))

    lock = threading.Lock()
    gui_image = server.gui.add_image(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), label="Camera image")
    gui_s = server.gui.add_slider("Sensor distance s", min=0.8, max=2.5, step=0.01,
                                  initial_value=round(float(image_distance(1.0, -objects[1]["point"][0])), 2))
    gui_f = server.gui.add_slider("Focal length f", min=0.6, max=1.6, step=0.01, initial_value=1.0)
    gui_text = server.gui.add_markdown("")

    def draw(_=None) -> None:
        with lock:
            s, f = gui_s.value, gui_f.value
            image = camera_image(objects, s, f)
            gui_image.image = (image * 255).astype(np.uint8)

            # The sensor, with the picture on it (upside down, as on a real sensor).
            server.scene.add_image("/sensor/image", (image[:, ::-1] * 255).astype(np.uint8), SENSOR_W, SENSOR_H,
                                   wxyz=FACING_LENS, position=(s, 0.0, 0.0))
            corners = np.array([[s, -1, -1], [s, -1, 1], [s, 1, 1], [s, 1, -1]]) * [1, SENSOR_H / 2, SENSOR_W / 2]
            server.scene.add_line_segments("/sensor/frame", np.stack([corners, np.roll(corners, -1, 0)], axis=1),
                                           colors=(40, 40, 40), thickness=3.0, thickness_units="screen")
            server.scene.add_label("/labels/sensor", "sensor", position=(s, SENSOR_H / 2 + 0.1, 0.0))
            for side, x in (("front", -f), ("back", f)):
                server.scene.add_icosphere(f"/focal_points/{side}", radius=0.025, color=(40, 40, 40), position=(x, 0.0, 0.0))
                server.scene.add_label(f"/labels/f_{side}", "F", position=(x, -0.08, 0.0))

            # Rays from the marked point of each object through the top, centre and
            # bottom of the lens. Behind the lens they head for the point's sharp
            # image P' (at distance z') and are drawn to just past the sensor.
            segments, colors = [], []
            for obj in objects:
                P = obj["point"]
                z = -P[0]
                if z <= f:
                    server.scene.add_icosphere(f"/images/{obj['name']}", radius=0.022, color=rgb(obj["color"]), visible=False)
                    continue
                z_image = float(image_distance(f, z))
                P_image = np.array([z_image, -P[1] * z_image / z, -P[2] * z_image / z])
                x_end = max(s, z_image) + 0.15
                for top in (APERTURE / 2, 0.0, -APERTURE / 2):
                    L = np.array([0.0, top, 0.0])
                    segments += [[P, L], [L, L + (P_image - L) * x_end / z_image]]
                    colors += [rgb(obj["color"])] * 2
                server.scene.add_icosphere(f"/images/{obj['name']}", radius=0.022, color=rgb(obj["color"]), position=P_image)
            server.scene.add_line_segments(
                "/rays", np.array(segments, dtype=np.float32), colors=np.repeat(np.array(colors)[:, None], 2, axis=1),
                thickness=2.5, thickness_units="screen",
            )

            gui_text.content = "\n\n".join(
                [f"Gauss: 1/z + 1/z' = 1/f. Sensor at s = {s:.2f}."]
                + [f"{obj['name'].capitalize()} (z = {-obj['point'][0]:.2f}): its rays {where(-obj['point'][0], s, f)}"
                   for obj in reversed(objects)]
            )

    gui_s.on_update(draw)
    gui_f.on_update(draw)
    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
