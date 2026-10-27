"""Lecture 2.2: a kinematic chain, the product of exponentials and inverse kinematics.

An arm with four revolute joints. Each joint is a twist xi_i (its axis, drawn
in its colour), and the pose of the hand is the product of exponentials

    G_sb(theta) = exp(theta_1 xi_1^) exp(theta_2 xi_2^) exp(theta_3 xi_3^) exp(theta_4 xi_4^) G_sb(0).

Move the joint sliders (forward kinematics), or drag the target and let the arm
reach for it (inverse kinematics with the articulated Jacobian). The coloured
arrows at the hand are the columns of the Jacobian: how the hand moves when only
one joint turns.

    python demos/02_kinematic_chain.py
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import trimesh
import viser
import viser.transforms as tf

from vh.kinematics import damped_least_squares, forward_kinematics, point_jacobian, revolute_twist
from vh.scene import merge, solid
from vh.viewer import configure

# The arm in its rest pose (all angles 0): a column on the floor, then three segments along x.
SHOULDER_HEIGHT = 0.5
JOINT_POINTS = np.array([[0.0, 0.0, 0.0], [0.0, SHOULDER_HEIGHT, 0.0], [0.8, SHOULDER_HEIGHT, 0.0], [1.4, SHOULDER_HEIGHT, 0.0]])
JOINT_AXES = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
JOINT_NAMES = ["base (turns about y)", "shoulder", "elbow", "wrist"]
TIP = np.array([1.8, SHOULDER_HEIGHT, 0.0])  # the end effector: the point we want to place
TWISTS = [revolute_twist(w, q) for w, q in zip(JOINT_AXES, JOINT_POINTS)]

COLORS = np.array([[220, 60, 50], [240, 150, 0], [40, 160, 70], [50, 100, 220]])
SEGMENT_COLOR = (0.75, 0.75, 0.78)
ARROW_SCALE = 0.3  # metres per radian of joint motion, for the Jacobian arrows


def segments() -> list:
    """The rigid parts, in the rest pose. Part i moves with joints 1..i."""
    gray = SEGMENT_COLOR
    column = solid(trimesh.creation.cylinder(radius=0.07, height=SHOULDER_HEIGHT, sections=24), gray, (0.0, SHOULDER_HEIGHT / 2, 0.0), z_to_y=True)
    parts = [column]
    for start, end in zip(JOINT_POINTS[1:, 0], np.r_[JOINT_POINTS[2:, 0], TIP[0]]):
        parts.append(solid(trimesh.creation.box((end - start - 0.1, 0.08, 0.08)), gray, ((start + end) / 2, SHOULDER_HEIGHT, 0.0)))
    # A joint is a cylinder along its axis. It belongs to the part before it.
    for k in range(1, 4):
        hinge = solid(trimesh.creation.cylinder(radius=0.06, height=0.16, sections=24), COLORS[k] / 255, JOINT_POINTS[k])
        parts[k - 1] = merge([parts[k - 1], hinge])
    hand = solid(trimesh.creation.icosphere(subdivisions=2, radius=0.05), (0.2, 0.2, 0.2), TIP)
    parts[3] = merge([parts[3], hand])
    return parts


def matrix_text(name: str, M: np.ndarray, digits: int = 2) -> str:
    rows = ["  ".join(f"{value:7.{digits}f}" for value in row) for row in np.atleast_2d(M)]
    pad = " " * (len(name) + 3)
    return "\n".join((f"{name} = " if k == 0 else pad) + row for k, row in enumerate(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = viser.ViserServer(port=args.port)
    configure(server)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/floor", width=4.0, height=4.0, plane="xz", cell_size=0.25, section_size=1.0)
    server.initial_camera.position = (1.6, 1.3, 2.2)
    server.initial_camera.look_at = (0.8, 0.6, 0.0)
    lock = threading.RLock()
    syncing = False

    parts = segments()
    part_handles = [server.scene.add_mesh_trimesh(f"/arm/part_{k}", part.trimesh()) for k, part in enumerate(parts)]
    rest = merge(parts)
    ghost = server.scene.add_mesh_simple(
        "/rest/arm", rest.vertices.astype(np.float32), rest.faces, color=(150, 150, 160), opacity=0.25, visible=False
    )
    target = server.scene.add_transform_controls("/target", scale=0.25, disable_rotations=True, position=(1.2, 1.0, 0.4))
    server.scene.add_icosphere("/target/ball", radius=0.04, color=(230, 30, 30))

    with server.gui.add_folder("Joint angles (forward kinematics)"):
        gui_angles = [server.gui.add_slider(name, min=-180.0, max=180.0, step=1.0, initial_value=0.0) for name in JOINT_NAMES]
        gui_reset = server.gui.add_button("Rest pose")
    with server.gui.add_folder("Reach the red target (inverse kinematics)"):
        gui_follow = server.gui.add_checkbox("Follow the target", False, hint="Run IK steps whenever the target moves.")
        gui_step = server.gui.add_button("One IK step", hint="Linearise with the Jacobian and take one damped least-squares step.")
        gui_damping = server.gui.add_slider(
            "Damping", min=0.0, max=1.0, step=0.01, initial_value=0.1, hint="0 is Gauss-Newton. More damping: smaller, safer steps."
        )
    with server.gui.add_folder("Show"):
        gui_jacobian = server.gui.add_checkbox("Jacobian columns at the hand", True)
        gui_rest = server.gui.add_checkbox("Rest pose and its twists", False, hint="The twists xi_i in the product of exponentials.")
    gui_numbers = server.gui.add_markdown("")

    def thetas() -> np.ndarray:
        return np.radians([s.value for s in gui_angles])

    def tip_position(motions) -> np.ndarray:
        return (motions[-1] @ np.r_[TIP, 1.0])[:3]

    def draw() -> None:
        theta = thetas()
        motions = forward_kinematics(TWISTS, theta)
        for handle, G in zip(part_handles, motions):
            handle.wxyz, handle.position = tf.SO3.from_matrix(G[:3, :3]).wxyz, G[:3, 3]
        tip = tip_position(motions)

        # Current joint axes: twist i moved by the joints before it.
        before = [np.eye(4)] + motions[:-1]
        axes = []
        for k in range(4):
            q = (before[k] @ np.r_[JOINT_POINTS[k], 1.0])[:3]
            w = before[k][:3, :3] @ JOINT_AXES[k]
            axes.append([q - 0.35 * w, q + 0.35 * w])
        server.scene.add_line_segments("/arm/axes", np.array(axes, dtype=np.float32), colors=np.repeat(COLORS[:, None], 2, axis=1), thickness=4.0, thickness_units="screen")
        rest_axes = [[q - 0.35 * w, q + 0.35 * w] for q, w in zip(JOINT_POINTS, JOINT_AXES)]
        server.scene.add_line_segments(
            "/rest/axes", np.array(rest_axes, dtype=np.float32), colors=np.repeat(COLORS[:, None], 2, axis=1),
            thickness=2.0, thickness_units="screen", visible=gui_rest.value,
        )
        ghost.visible = gui_rest.value

        J = point_jacobian(TWISTS, theta, tip)
        arrows = np.stack([np.stack([tip, tip + ARROW_SCALE * J[:, k]]) for k in range(4)])
        server.scene.add_arrows(
            "/arm/jacobian", arrows.astype(np.float32), colors=COLORS, shaft_radius=0.008, head_radius=0.02, head_length=0.04,
            visible=gui_jacobian.value,
        )
        goal = np.asarray(target.position, dtype=np.float64)
        server.scene.add_line_segments(
            "/error", np.array([[tip, goal]], dtype=np.float32), colors=(230, 30, 30), thickness=1.5, thickness_units="screen"
        )

        singular = np.linalg.svd(J, compute_uv=False)
        G_sb = motions[-1].copy()
        G_sb[:3, 3] = tip  # G_sb(0) only moves the hand frame to the tip, without rotation
        gui_numbers.content = "\n".join(
            [
                "```",
                matrix_text("G_sb", G_sb),
                "",
                matrix_text("J   ", J),
                "```",
                f"Distance to the target: **{np.linalg.norm(goal - tip):.3f} m**. "
                f"Singular values of J: {', '.join(f'{s:.2f}' for s in singular)}. "
                "A singular value near 0 means there is a direction the hand cannot move in "
                "(try stretching the arm out straight).",
            ]
        )

    def set_angles(theta) -> None:
        nonlocal syncing
        syncing = True
        try:
            for slider, value in zip(gui_angles, np.degrees(theta)):
                slider.value = float((value + 180.0) % 360.0 - 180.0)
        finally:
            syncing = False

    def ik_step() -> float:
        """One damped least-squares step towards the target. Returns the remaining distance."""
        theta = thetas()
        tip = tip_position(forward_kinematics(TWISTS, theta))
        error = np.asarray(target.position, dtype=np.float64) - tip
        step = damped_least_squares(point_jacobian(TWISTS, theta, tip), error, gui_damping.value)
        step = step * min(1.0, 0.5 / max(np.abs(step).max(), 1e-12))  # at most ~30 degrees per step
        set_angles(theta + step)
        return float(np.linalg.norm(error))

    def on_angle(_) -> None:
        if not syncing:
            with lock:
                draw()

    def on_target(_) -> None:
        with lock:
            if gui_follow.value:
                for _ in range(10):
                    if ik_step() < 1e-3:
                        break
            draw()

    def on_step(_) -> None:
        with lock:
            ik_step()
            draw()

    def on_reset(_) -> None:
        with lock:
            set_angles(np.zeros(4))
            draw()

    def on_show(_) -> None:
        with lock:
            draw()

    for slider in gui_angles:
        slider.on_update(on_angle)
    target.on_update(on_target)
    gui_step.on_click(on_step)
    gui_reset.on_click(on_reset)
    for control in [gui_jacobian, gui_rest, gui_follow]:
        control.on_update(on_show)

    set_angles(np.radians([20.0, 40.0, -60.0, -20.0]))
    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
