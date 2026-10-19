"""Lecture 2.2: rotations, and the many ways to write them down.

A small airplane is turned in five ways, one page each:

    1. Order matters        the same three rotations in a different order
    2. Rotation matrix      the columns of R are the airplane's axes
    3. Euler angles         a gimbal, and the gimbal lock
    4. Interpolation        Euler angles, matrices and quaternions (SLERP) from A to B
    5. Axis-angle           the exponential map and the Rodrigues formula

Every page lists the same rotation as a matrix, Euler angles, a quaternion and
an axis-angle vector (the functions are in vh/rotations.py).

    python demos/02_rotations.py
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import trimesh
import viser
import viser.transforms as tf

from vh.rotations import (
    AXES,
    axis_rotation,
    euler_to_matrix,
    exp_map,
    hat,
    log_map,
    matrix_to_euler_xyz,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_to_matrix,
    slerp,
)
from vh.scene import merge, solid

PAGES = ["1. Order matters", "2. Rotation matrix", "3. Euler angles and gimbal lock", "4. Interpolation", "5. Axis-angle"]
AXIS_COLORS = {"x": (220, 50, 50), "y": (40, 160, 60), "z": (50, 90, 220)}
CENTER = np.array([0.0, 1.2, 0.0])
CONVENTIONS = ["x-y-z", "z-y-x", "z-x-z", "y-x-y"]


def airplane():
    """An airplane flying along +z, with its wings along x and its fin up (+y). Every part has its own colour."""
    parts = [
        solid(trimesh.creation.cylinder(radius=0.07, height=0.8, sections=24), (0.92, 0.92, 0.92)),
        solid(trimesh.creation.cone(radius=0.07, height=0.22, sections=24), (0.85, 0.2, 0.15), (0.0, 0.0, 0.4)),
        solid(trimesh.creation.box((0.55, 0.025, 0.22)), (0.25, 0.65, 0.3), (0.32, 0.0, 0.05)),  # left wing (+x)
        solid(trimesh.creation.box((0.55, 0.025, 0.22)), (0.95, 0.6, 0.15), (-0.32, 0.0, 0.05)),  # right wing
        solid(trimesh.creation.box((0.4, 0.02, 0.1)), (0.7, 0.7, 0.75), (0.0, 0.0, -0.36)),
        solid(trimesh.creation.box((0.02, 0.2, 0.12)), (0.25, 0.45, 0.85), (0.0, 0.1, -0.36)),  # fin
    ]
    return merge(parts)


def rotation_text(R: np.ndarray) -> str:
    """The same rotation in four representations."""
    lines = [("R = " if k == 0 else "    ") + "[" + " ".join(f"{v:6.3f}" for v in row) + "]" for k, row in enumerate(R)]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-6 or not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        return "\n".join(lines + ["", f"det R = {det:.3f}, R^T R != I: this is not a rotation."])
    euler = np.degrees(matrix_to_euler_xyz(R))
    q = quat_from_matrix(R)
    r = log_map(R)
    angle = np.linalg.norm(r)
    axis = r / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
    lines += [
        "",
        "Euler x-y-z, intrinsic (deg)",
        f"    ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})",
        "quaternion (w, x, y, z), or minus it",
        f"    ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})",
        "axis-angle",
        f"    axis ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})",
        f"    angle {np.degrees(angle):.1f} deg",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    plane = airplane()
    plane_mesh = plane.trimesh()

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/floor", width=6.0, height=6.0, plane="xz", cell_size=0.5, section_size=1.0)
    server.initial_camera.position = (2.2, 2.4, 3.6)
    server.initial_camera.look_at = (0.0, 1.1, 0.0)
    lock = threading.RLock()

    gui_page = server.gui.add_dropdown("Page", PAGES, initial_value=PAGES[0])
    state = {"folder": None, "numbers": None, "draw": lambda: None}

    # ------------------------------------------------------------ drawing helpers
    def show_plane(name: str, M: np.ndarray, position, scale: float = 1.0) -> None:
        """The airplane transformed by M (any 3 x 3 matrix, so also non-rotations) and moved to position."""
        mesh = plane_mesh.copy()
        mesh.vertices = scale * plane_mesh.vertices @ M.T + position
        server.scene.add_mesh_trimesh(f"/page/{name}", mesh)

    def show_ghost(name: str, M: np.ndarray, position, color, scale: float = 1.0) -> None:
        vertices = scale * plane.vertices @ M.T + position
        server.scene.add_mesh_simple(f"/page/{name}", vertices.astype(np.float32), plane.faces, color=color, opacity=0.35, cast_shadow=False)

    def show_axes(name: str, M: np.ndarray, position, length: float = 0.7) -> None:
        """The columns of M as arrows: where the airplane's x, y and z axes point."""
        points = np.stack([np.stack([position, position + length * M[:, k]]) for k in range(3)])
        server.scene.add_arrows(
            f"/page/{name}", points.astype(np.float32), colors=np.array([AXIS_COLORS[a] for a in "xyz"]),
            shaft_radius=0.012, head_radius=0.03, head_length=0.06,
        )

    def numbers(text: str, R: np.ndarray | None) -> None:
        body = [text]
        if R is not None:
            body += ["```", rotation_text(R), "```"]
        state["numbers"].content = "\n\n".join(body)

    # ------------------------------------------------------------ 1. order matters
    def page_order(folder) -> None:
        sequences = [("y", "z", "x"), ("z", "y", "x")]
        positions = [CENTER + [-1.3, 0.0, 0.0], CENTER + [1.3, 0.0, 0.0]]
        with folder:
            gui_steps = server.gui.add_slider("Rotations done", min=0.0, max=3.0, step=0.01, initial_value=0.0)
            gui_play = server.gui.add_button("Play")
        for name, sequence, position in zip("ab", sequences, positions):
            server.scene.add_label(f"/page/label_{name}", f"{sequence[0]}, then {sequence[1]}, then {sequence[2]}",
                                   position=position + [0.0, 0.75, 0.0], anchor="bottom-center")

        def rotation(sequence, steps: float) -> np.ndarray:
            R = np.eye(3)
            for k, axis in enumerate(sequence):
                amount = np.clip(steps - k, 0.0, 1.0)
                R = axis_rotation(axis, amount * np.pi / 2) @ R  # about the fixed world axis
            return R

        def draw() -> None:
            Rs = [rotation(sequence, gui_steps.value) for sequence in sequences]
            for name, R, position in zip("ab", Rs, positions):
                show_plane(f"plane_{name}", R, position)
                show_axes(f"axes_{name}", np.eye(3), position + [0.0, -0.9, 0.0], length=0.3)
            same = "the same" if np.allclose(Rs[0], Rs[1]) else "different"
            numbers(
                "Each airplane turns 90 deg about three world axes, in the order written above it. "
                f"After {gui_steps.value:.2f} of the three rotations the two airplanes are **{same}**. "
                "In 3D the order of rotations matters: R_x R_z R_y != R_x R_y R_z. "
                "(The small arrows show the world axes: x red, y green, z blue.)",
                None,
            )

        def on_play(_) -> None:
            for steps in np.linspace(0.0, 3.0, 90):
                gui_steps.value = float(steps)
                time.sleep(0.03)

        gui_steps.on_update(lambda _: redraw())
        gui_play.on_click(on_play)
        state["draw"] = draw

    # ------------------------------------------------------------ 2. rotation matrix
    def page_matrix(folder) -> None:
        with folder:
            gui_mirror = server.gui.add_checkbox("Mirror the airplane", False, hint="Flip the x axis: still orthogonal, but det = -1.")
            gui_reset = server.gui.add_button("Reset rotation")
        handle = server.scene.add_transform_controls("/page/handle", scale=0.9, disable_axes=True, disable_sliders=True, position=CENTER)
        handle.wxyz = tf.SO3.from_matrix(euler_to_matrix(np.radians([20.0, 35.0, 10.0]))).wxyz

        def draw() -> None:
            R = tf.SO3(np.array(handle.wxyz)).as_matrix()
            if gui_mirror.value:
                R = R @ np.diag([-1.0, 1.0, 1.0])
            show_plane("plane", R, CENTER)
            show_axes("axes", R, CENTER)
            numbers(
                "Drag the rings to rotate the airplane. The **columns of R** are the airplane's own axes "
                "written in world coordinates: column 1 is the red arrow (towards the left wing), "
                "column 2 the green one (up through the fin), column 3 the blue one (out of the nose).",
                R,
            )

        def on_reset(_) -> None:
            handle.wxyz = (1.0, 0.0, 0.0, 0.0)
            redraw()

        handle.on_update(lambda _: redraw())
        gui_mirror.on_update(lambda _: redraw())
        gui_reset.on_click(on_reset)
        state["draw"] = draw

    # ------------------------------------------------------------ 3. Euler angles
    def page_euler(folder) -> None:
        with folder:
            gui_convention = server.gui.add_dropdown("Axes", CONVENTIONS, initial_value=CONVENTIONS[0])
            gui_extrinsic = server.gui.add_checkbox(
                "About the world axes (extrinsic)", False, hint="Off: each rotation is about the axes already turned by the ones before (intrinsic)."
            )
            gui_angles = [server.gui.add_slider(f"Angle {k + 1} (deg)", min=-180.0, max=180.0, step=1.0, initial_value=v)
                          for k, v in enumerate([30.0, 20.0, 40.0])]
            gui_lock = server.gui.add_button("Go to the gimbal lock")
        radii = [1.0, 0.86, 0.72]
        rings = [trimesh.creation.torus(r, 0.025, major_sections=64, minor_sections=8) for r in radii]

        def draw() -> None:
            order = gui_convention.value.replace("-", "")
            angles = np.radians([s.value for s in gui_angles])
            R = euler_to_matrix(angles, order, intrinsic=not gui_extrinsic.value)
            # The gimbal turns about the moving axes. An extrinsic sequence is the
            # intrinsic one in reverse order, so the gimbal is nested the other way round.
            g_order, g_angles = (order, angles) if not gui_extrinsic.value else (order[::-1], angles[::-1])
            frame = np.eye(3)
            turn_axes = []
            for k in range(3):
                axis = np.eye(3)[AXES[g_order[k]]]
                turn_axes.append(frame @ axis)
                frame = frame @ axis_rotation(g_order[k], g_angles[k])
                other = np.eye(3)[AXES[g_order[k + 1] if k < 2 else g_order[1]]]
                normal = np.cross(axis, other)
                normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-9 else np.eye(3)[(AXES[g_order[k]] + 1) % 3]
                # Rest orientation of ring k: its plane holds its own axis (the pins) and the next ring's pins.
                rest = np.stack([axis, np.cross(normal, axis), normal], axis=1)
                M = frame @ rest
                vertices = rings[k].vertices @ M.T + CENTER
                server.scene.add_mesh_simple(f"/page/ring_{k}", vertices.astype(np.float32), rings[k].faces, color=AXIS_COLORS[g_order[k]])
                pin = frame @ axis
                outer = radii[k - 1] if k > 0 else radii[0] + 0.25
                pins = [[CENTER + s * radii[k] * pin, CENTER + s * outer * pin] for s in (-1.0, 1.0)]
                server.scene.add_line_segments(f"/page/pins_{k}", np.array(pins, dtype=np.float32), colors=AXIS_COLORS[g_order[k]],
                                               thickness=5.0, thickness_units="screen")
            show_plane("plane", R, CENTER, scale=0.95)
            arrows = np.stack([np.stack([CENTER, CENTER + 1.3 * a]) for a in turn_axes])
            server.scene.add_arrows("/page/turn_axes", arrows.astype(np.float32), colors=np.array([AXIS_COLORS[a] for a in g_order]),
                                    shaft_radius=0.01, head_radius=0.03, head_length=0.06)
            volume = abs(np.linalg.det(np.stack(turn_axes)))
            locked = "**Gimbal lock**: two rings turn about the same axis, so one direction of rotation is lost. " if volume < 0.02 else ""
            sequence = " ".join(f"R_{a}" for a in order) if not gui_extrinsic.value else " ".join(f"R_{a}" for a in order[::-1])
            numbers(
                f"R = {sequence}. The arrows are the three axes the rings turn about right now; "
                f"they span a volume of {volume:.2f} (1 when they are perpendicular). {locked}",
                R,
            )

        def on_lock(_) -> None:
            order = gui_convention.value.replace("-", "")
            gui_angles[1].value = 0.0 if order[0] == order[2] else 90.0

        for control in [gui_convention, gui_extrinsic, *gui_angles]:
            control.on_update(lambda _: redraw())
        gui_lock.on_click(on_lock)
        state["draw"] = draw

    # ------------------------------------------------------------ 4. interpolation
    def page_interpolation(folder) -> None:
        rng = np.random.default_rng(3)
        end = {"R": euler_to_matrix(np.radians([150.0, 60.0, -120.0]))}
        rows = [("Euler angles, interpolated linearly", 2.2, (220, 120, 40)),
                ("Matrices, interpolated linearly", 1.2, (150, 80, 200)),
                ("Quaternions, SLERP", 0.2, (40, 140, 220))]
        with folder:
            gui_t = server.gui.add_slider("t", min=0.0, max=1.0, step=0.01, initial_value=0.5)
            gui_play = server.gui.add_button("Play")
            gui_new = server.gui.add_button("New end orientation")
        for name, height, _ in rows:
            server.scene.add_label(f"/page/label_{height}", name, position=(-3.4, height + 0.45, 0.0))

        def interpolated(method: int, t: float) -> np.ndarray:
            R1 = end["R"]
            if method == 0:
                return euler_to_matrix(t * matrix_to_euler_xyz(R1))
            if method == 1:
                return (1 - t) * np.eye(3) + t * R1
            return quat_to_matrix(slerp(np.array([1.0, 0.0, 0.0, 0.0]), quat_from_matrix(R1), t))

        def draw() -> None:
            t = gui_t.value
            speeds = []
            for method, (name, height, color) in enumerate(rows):
                for k, s in enumerate(np.linspace(0.0, 1.0, 7)):
                    show_ghost(f"ghost_{method}_{k}", interpolated(method, s), (-2.7 + 5.4 * s, height, 0.0), color, scale=0.6)
                show_plane(f"plane_{method}", interpolated(method, t), (-2.7 + 5.4 * t, height, 0.9), scale=0.6)
                samples = [interpolated(method, s) for s in np.linspace(0.0, 1.0, 41)]
                if method == 1:
                    speeds.append(f"| {name} | det R goes down to {min(np.linalg.det(R) for R in samples):.2f}: not rotations |")
                    continue
                steps = [np.degrees(np.linalg.norm(log_map(b @ a.T))) for a, b in zip(samples[:-1], samples[1:])]
                total = np.degrees(np.linalg.norm(log_map(end["R"])))
                speeds.append(f"| {name} | {min(steps) * 40:.0f} to {max(steps) * 40:.0f} deg per unit of t (shortest: {total:.0f}) |")
            numbers(
                "All three rows go from the same start (left) to the same end (right). The front airplane is at t.\n\n"
                "| | turning speed |\n|---|---|\n" + "\n".join(speeds),
                interpolated(2, t),
            )

        def on_play(_) -> None:
            for t in np.linspace(0.0, 1.0, 60):
                gui_t.value = float(t)
                time.sleep(0.04)

        def on_new(_) -> None:
            end["R"] = exp_map(rng.normal(size=3) * 1.4)
            redraw()

        gui_t.on_update(lambda _: redraw())
        gui_play.on_click(on_play)
        gui_new.on_click(on_new)
        state["draw"] = draw

    # ------------------------------------------------------------ 5. axis-angle
    def page_axis_angle(folder) -> None:
        with folder:
            gui_azimuth = server.gui.add_slider("Axis azimuth (deg)", min=-180.0, max=180.0, step=1.0, initial_value=30.0)
            gui_elevation = server.gui.add_slider("Axis elevation (deg)", min=-90.0, max=90.0, step=1.0, initial_value=50.0)
            gui_angle = server.gui.add_slider("Angle theta (deg)", min=-360.0, max=720.0, step=1.0, initial_value=90.0)
            gui_play = server.gui.add_button("Play: turn twice around")
        nose = np.array([0.0, 0.0, 0.62])

        def draw() -> None:
            a, e = np.radians(gui_azimuth.value), np.radians(gui_elevation.value)
            w = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
            theta = np.radians(gui_angle.value)
            R = exp_map(theta * w)
            show_plane("plane", R, CENTER)
            server.scene.add_arrows("/page/axis", np.array([[CENTER - 1.0 * w, CENTER + 1.0 * w]], dtype=np.float32),
                                    colors=(30, 30, 30), shaft_radius=0.012, head_radius=0.035, head_length=0.08)
            circle = np.array([exp_map(phi * w) @ nose for phi in np.linspace(0.0, 2 * np.pi, 97)]) + CENTER
            server.scene.add_line_segments("/page/circle", np.stack([circle[:-1], circle[1:]], axis=1).astype(np.float32),
                                           colors=(170, 170, 170), thickness=1.5, thickness_units="screen")
            swept = np.array([exp_map(phi * w) @ nose for phi in np.linspace(0.0, theta, 97)]) + CENTER
            server.scene.add_line_segments("/page/swept", np.stack([swept[:-1], swept[1:]], axis=1).astype(np.float32),
                                           colors=(220, 50, 40), thickness=4.0, thickness_units="screen")
            K = hat(w)
            terms = [("I", np.eye(3)), ("sin(theta) [w]x", np.sin(theta) * K), ("(1 - cos(theta)) [w]x^2", (1 - np.cos(theta)) * K @ K)]
            q = quat_from_axis_angle(theta * w)
            text = "\n\n".join(name + "\n" + "\n".join("    [" + " ".join(f"{v:6.2f}" for v in row) + "]" for row in M) for name, M in terms)
            numbers(
                "R = exp(theta [w]x) = I + sin(theta) [w]x + (1 - cos(theta)) [w]x^2 (Rodrigues). "
                "The red arc is the path of the nose. The quaternion of this axis and angle is "
                f"(cos(theta/2), sin(theta/2) w) = **({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f})**: "
                "after one full turn it is minus where it started, after two it is back.\n\n```\n" + text + "\n```",
                R,
            )

        def on_play(_) -> None:
            for angle in np.linspace(0.0, 720.0, 120):
                gui_angle.value = float(angle)
                time.sleep(0.03)

        for control in [gui_azimuth, gui_elevation, gui_angle]:
            control.on_update(lambda _: redraw())
        gui_play.on_click(on_play)
        state["draw"] = draw

    # ------------------------------------------------------------ pages
    builders = [page_order, page_matrix, page_euler, page_interpolation, page_axis_angle]

    def redraw() -> None:
        with lock:
            state["draw"]()

    def on_page(_) -> None:
        with lock:
            if state["folder"] is not None:
                state["folder"].remove()
                state["numbers"].remove()
            server.scene.remove_by_name("/page")
            state["folder"] = server.gui.add_folder(gui_page.value)
            builders[PAGES.index(gui_page.value)](state["folder"])
            state["numbers"] = server.gui.add_markdown("")
            state["draw"]()

    gui_page.on_update(on_page)
    on_page(None)
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
