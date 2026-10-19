"""Lecture 2: why telephoto pictures look flat.

Two people stand one behind the other. The camera moves back while zooming in,
so that the front person keeps the same size in the image. Their image sizes
are f * H / Z1 and f * H / Z2, so their ratio is Z1 / Z2: far away this ratio
goes to 1 and the two look as if they stood in the same plane. This is weak
perspective, the limit of the pinhole camera for distant objects. The trees
and the house behind them grow for the same reason.

    python demos/02_telephoto.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from vh import SMPLModel
from vh.camera import intrinsics, look_at, project, world_to_camera
from vh.render import draw_line
from vh.scene import body, draw_segments, merge, moved, sample, show_camera, solid, stick_figure, take_picture

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

WIDTH, HEIGHT = 480, 360  # image size in pixels
SENSOR_WIDTH_MM = 36.0  # a full-frame camera, to turn f in pixels into millimetres
EYE_HEIGHT = 1.2  # metres
FRONT_SIZE = 0.6  # the front person fills this fraction of the image height
POINTS_PER_PERSON = 120_000

FRONT_COLOR, BACK_COLOR = np.array([0.35, 0.55, 0.85]), np.array([0.85, 0.42, 0.35])
TRUNK, LEAVES, WALL, ROOF = (0.45, 0.30, 0.18), (0.25, 0.55, 0.28), (0.93, 0.90, 0.82), (0.72, 0.25, 0.20)


def background():
    """A row of trees and a house behind the two people."""
    parts = []
    for x in np.arange(-9.0, 9.1, 3.0):
        parts.append(solid(trimesh.creation.cylinder(radius=0.12, height=1.5, sections=12), TRUNK, (x, 0.75, -12.0), z_to_y=True))
        parts.append(solid(trimesh.creation.icosphere(subdivisions=2, radius=1.1), LEAVES, (x, 2.4, -12.0)))
    parts.append(solid(trimesh.creation.box((6.0, 3.5, 3.0)), WALL, (0.0, 1.75, -24.0)))
    w, h, d = 3.5, 2.0, 1.8
    roof = np.array([[-w, 0, -d], [w, 0, -d], [0, h, -d], [-w, 0, d], [w, 0, d], [0, h, d]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], [1, 2, 5], [1, 5, 4]])
    parts.append(solid(trimesh.Trimesh(roof, faces, process=False), ROOF, (0.0, 3.5, -24.0)))
    return merge(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model_paths = sorted(args.model_dir.glob("smpl_*.npz"))
    if model_paths:
        model = SMPLModel.load(next((p for p in model_paths if p.stem == "smpl_neutral"), model_paths[0]))
        make_person = lambda color: body(model, "Arms down", color)  # noqa: E731
    else:
        print(f"No SMPL model in {args.model_dir}: the people are simple figures (see README).")
        make_person = stick_figure
    people = [make_person(FRONT_COLOR), make_person(BACK_COLOR)]
    person_height = float(people[0].vertices[:, 1].max())
    scenery = background()
    scenery_samples = sample(scenery, 400_000)

    # Ground lines every metre.
    xs, zs = np.arange(-15.0, 15.01, 1.0), np.arange(-30.0, 120.01, 1.0)
    line_starts = np.array([[x, 0.0, zs[0]] for x in xs] + [[xs[0], 0.0, z] for z in zs])
    line_ends = np.array([[x, 0.0, zs[-1]] for x in xs] + [[xs[-1], 0.0, z] for z in zs])

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_mesh_trimesh("/background", scenery.trimesh())
    server.scene.add_grid("/ground", width=30.0, height=150.0, plane="xz", cell_size=1.0, section_size=5.0, position=(0.0, 0.0, 45.0))
    server.initial_camera.position = (-14.0, 9.0, 16.0)
    server.initial_camera.look_at = (0.0, 1.0, 1.0)
    lock = threading.Lock()
    syncing = False
    playing = False

    gui_image = server.gui.add_image(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), label="Camera image")
    with server.gui.add_folder("Camera"):
        gui_distance = server.gui.add_slider(
            "Distance to the front person (m)", min=1.5, max=100.0, step=0.5, initial_value=2.0,
        )
        gui_f = server.gui.add_slider("Focal length f (px)", min=50.0, max=20000.0, step=10.0, initial_value=300.0)
        gui_keep = server.gui.add_checkbox(
            "Keep the front person's size", True, hint="Zoom in as the camera moves back (f / distance stays fixed).",
        )
        gui_play = server.gui.add_button("Play", hint="Move back to 100 m while zooming in, then return.")
    with server.gui.add_folder("People"):
        gui_gap = server.gui.add_slider("Gap between them (m)", min=0.5, max=5.0, step=0.1, initial_value=2.0)
        gui_rays = server.gui.add_checkbox("Show rays to heads and feet", True)
    gui_numbers = server.gui.add_markdown("")

    def kept_f(distance: float) -> float:
        """f that makes the front person FRONT_SIZE of the image height at this distance."""
        return FRONT_SIZE * HEIGHT * distance / person_height

    state = {}

    def place_people() -> None:
        """Front person at z = 0, back person `gap` metres behind, a bit to the side."""
        gap = gui_gap.value
        placed = [moved(people[0], [-0.3, 0.0, 0.0]), moved(people[1], [0.35, 0.0, -gap])]
        state["placed"] = placed
        state["samples"] = [sample(p, POINTS_PER_PERSON, seed=k) for k, p in enumerate(placed)]
        for k, p in enumerate(placed):
            server.scene.add_mesh_trimesh(f"/person_{k}", p.trimesh())

    def draw(_=None) -> None:
        with lock:
            distance, f, gap = gui_distance.value, gui_f.value, gui_gap.value
            eye = np.array([0.0, EYE_HEIGHT, distance])
            R, t = look_at(eye, [0.0, EYE_HEIGHT - 0.3, -gap / 2])
            K = intrinsics(f, WIDTH / 2, HEIGHT / 2)

            # --- the image: ground lines, then the scenery and the people over them
            image = np.ones((HEIGHT, WIDTH, 3))
            draw_segments(image, world_to_camera(line_starts, R, t), world_to_camera(line_ends, R, t), K, (0.75, 0.75, 0.75))
            for samples in [scenery_samples, state["samples"][1], state["samples"][0]]:
                picture, covered = take_picture(samples, eye, R, t, K, WIDTH, HEIGHT)
                image[covered] = picture[covered]  # drawn back to front: the scenery, the back person, the front person

            # Height of each person in the image, from all their vertices, and a bar for it on the right.
            heights, depths = [], []
            for k, person in enumerate(state["placed"]):
                X = world_to_camera(person.vertices, R, t)
                v = project(X, K)[:, 1]
                heights.append(v.max() - v.min())
                depths.append(float(X[:, 2].mean()))
                u = WIDTH - 24 + 10 * k
                bottom = HEIGHT - 10
                color = (FRONT_COLOR, BACK_COLOR)[k]
                for du in range(6):
                    draw_line(image, (u + du, bottom), (u + du, bottom - min(heights[-1], HEIGHT - 20)), color)
            gui_image.image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            show_camera(server, eye, R, K, image)

            # --- rays from the camera centre to the top and bottom of each person
            segments, colors = [], []
            for k, person in enumerate(state["placed"]):
                centre = person.vertices.mean(axis=0)
                for y in (0.0, person_height):
                    segments.append([eye, [centre[0], y, centre[2]]])
                    colors.append((np.array((FRONT_COLOR, BACK_COLOR)[k]) * 255).astype(int))
            server.scene.add_line_segments(
                "/rays", np.array(segments, dtype=np.float32), colors=np.repeat(np.array(colors)[:, None], 2, axis=1),
                thickness=2.0, thickness_units="screen", visible=gui_rays.value,
            )

            # --- the numbers
            f_mm = f * SENSOR_WIDTH_MM / WIDTH
            fov = np.degrees(2 * np.arctan(WIDTH / (2 * f)))
            gui_numbers.content = "\n".join(
                [
                    "| | depth Z (m) | height in the image (px) |",
                    "|---|---:|---:|",
                    f"| front (blue) | {depths[0]:.1f} | {heights[0]:.0f} |",
                    f"| back (red) | {depths[1]:.1f} | {heights[1]:.0f} |",
                    "",
                    f"Size ratio back / front = **{heights[1] / heights[0]:.3f}**, "
                    f"depth ratio Z_front / Z_back = {depths[0] / depths[1]:.3f}",
                    "",
                    f"f = {f:.0f} px, like a **{f_mm:.0f} mm** lens on a full-frame camera; "
                    f"horizontal field of view {fov:.1f} deg.",
                ]
            )

    def set_value(slider, value: float) -> None:
        nonlocal syncing
        syncing = True
        try:
            slider.value = float(np.clip(value, slider.min, slider.max))
        finally:
            syncing = False

    def on_distance(_) -> None:
        if syncing:
            return
        if gui_keep.value:
            set_value(gui_f, kept_f(gui_distance.value))
        draw()

    def on_f(_) -> None:
        if syncing:
            return
        if gui_keep.value:  # zooming moves the camera, as in a dolly zoom
            set_value(gui_distance, gui_f.value * person_height / (FRONT_SIZE * HEIGHT))
        draw()

    def on_keep(_) -> None:
        if gui_keep.value:
            set_value(gui_f, kept_f(gui_distance.value))
            draw()

    def on_gap(_) -> None:
        with lock:
            place_people()
        draw()

    def on_play(_) -> None:
        nonlocal playing
        if playing:
            return
        playing = True
        try:
            start = gui_distance.value
            path = np.concatenate([np.geomspace(start, 100.0, 80), np.geomspace(100.0, start, 40)])
            for distance in path:
                set_value(gui_distance, distance)
                set_value(gui_f, kept_f(distance))
                draw()
                time.sleep(0.03)
        finally:
            playing = False

    gui_distance.on_update(on_distance)
    gui_f.on_update(on_f)
    gui_keep.on_update(on_keep)
    gui_gap.on_update(on_gap)
    gui_rays.on_update(draw)
    gui_play.on_click(on_play)

    place_people()
    set_value(gui_f, kept_f(gui_distance.value))
    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
