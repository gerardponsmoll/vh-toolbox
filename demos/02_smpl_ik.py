"""Lecture 2.2: inverse kinematics on SMPL.

Drag the red target and a hand (or foot, or the head) of the SMPL body reaches
for it. Every joint of the chain is a ball joint with three degrees of freedom.
Turning joint k by a small rotation delta about the world axes moves the end
effector p by delta x (p - j_k), so the Jacobian has three columns per joint,
[e_x x (p - j_k), e_y x (p - j_k), e_z x (p - j_k)]: twists through the joint.

SMPL stores each joint's rotation relative to its parent (axis-angle), so a
rotation found in world coordinates is brought into the parent's frame:
R_k <- R_parent^T exp(delta) R_parent R_k. The panel lists the result.

    python demos/02_smpl_ik.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser

from vh import JOINT_NAMES, SMPLModel, forward, joint_transforms, rodrigues
from vh.kinematics import damped_least_squares
from vh.rotations import exp_map, hat, log_map

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
EFFECTORS = {"Right hand": 23, "Left hand": 22, "Right foot": 11, "Left foot": 10, "Head": 15}
SKIN = (222, 184, 160)
CHAIN_COLOR, BONE_COLOR, TARGET_COLOR = (240, 140, 0), (90, 90, 90), (230, 30, 30)


def chain_to_root(parents: np.ndarray, effector: int) -> list[int]:
    """The joints that move the effector: its parent, grandparent, ..., up to the pelvis.

    The pelvis (the root) is left out: turning it would turn the whole body."""
    chain, k = [], parents[effector]
    while k > 0:
        chain.append(int(k))
        k = parents[k]
    return chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    paths = sorted(args.model_dir.glob("smpl_*.npz"))
    if not paths:
        raise SystemExit(f"No SMPL model in {args.model_dir}: see the README to prepare one.")
    model = SMPLModel.load(next((p for p in paths if p.stem == "smpl_neutral"), paths[0]))
    parents = model.parents
    joints_rest = forward(model).joints_rest
    lift = -forward(model).vertices[:, 1].min()  # stand the body on the floor

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/floor", width=4.0, height=4.0, plane="xz", cell_size=0.25, section_size=1.0)
    server.initial_camera.position = (1.2, 1.4, 2.8)
    server.initial_camera.look_at = (0.0, 1.0, 0.0)
    lock = threading.RLock()
    syncing = False  # True while the target is moved from code

    pose = np.zeros((model.num_joints, 3))
    body = server.scene.add_mesh_simple("/body", forward(model).vertices.astype(np.float32), model.faces, color=SKIN, opacity=0.85)
    # Limits: dragging along an arrow that points at the viewer could otherwise throw the target far away.
    target = server.scene.add_transform_controls(
        "/target", scale=0.2, disable_rotations=True, translation_limits=((-1.2, 1.2), (0.0, 2.4), (-1.2, 1.2))
    )
    server.scene.add_icosphere("/target/ball", radius=0.03, color=TARGET_COLOR)

    with server.gui.add_folder("Inverse kinematics"):
        gui_effector = server.gui.add_dropdown("End effector", list(EFFECTORS), initial_value="Right hand")
        gui_length = server.gui.add_slider(
            "Joints in the chain", min=1, max=7, step=1, initial_value=3, hint="Counted from the effector towards the pelvis, which stays fixed."
        )
        gui_follow = server.gui.add_checkbox("Follow the target", True)
        gui_step = server.gui.add_button("One IK step")
        gui_damping = server.gui.add_slider("Damping", min=0.0, max=1.0, step=0.01, initial_value=0.1)
        gui_reset = server.gui.add_button("Reset pose")
    gui_numbers = server.gui.add_markdown("")

    def chain() -> list[int]:
        full = chain_to_root(parents, EFFECTORS[gui_effector.value])
        return full[: int(gui_length.value)]

    def kinematics():
        G = joint_transforms(rodrigues(pose), joints_rest, parents)
        G[:, 1, 3] += lift
        return G

    def ik_step() -> float:
        """One damped least-squares step. Returns the distance to the target before the step."""
        G = kinematics()
        joints = G[:, :3, 3]
        p = joints[EFFECTORS[gui_effector.value]]
        error = np.asarray(target.position, dtype=np.float64) - p
        links = chain()
        # Three columns per joint: rotating about the world x, y or z axis through the joint.
        J = np.concatenate([-hat(p - joints[k]) for k in links], axis=1)  # delta x r = -r x delta = -hat(r) delta
        step = damped_least_squares(J, error, gui_damping.value).reshape(-1, 3)
        step = step * min(1.0, 0.3 / max(np.linalg.norm(step, axis=1).max(), 1e-12))
        for k, delta in zip(links, step):
            # The rotation delta is in world coordinates; SMPL wants the joint's rotation relative to its parent.
            R_parent = G[parents[k], :3, :3] if parents[k] >= 0 else np.eye(3)
            R_local = R_parent.T @ exp_map(delta) @ R_parent @ exp_map(pose[k])
            pose[k] = log_map(R_local)
        return float(np.linalg.norm(error))

    def draw() -> None:
        out = forward(model, pose=pose)
        body.vertices = (out.vertices + [0.0, lift, 0.0]).astype(np.float32)
        joints = out.joints + [0.0, lift, 0.0]
        effector, links = EFFECTORS[gui_effector.value], chain()

        bones = [[joints[k], joints[parents[k]]] for k in range(1, model.num_joints)]
        server.scene.add_line_segments("/skeleton", np.array(bones, dtype=np.float32), colors=BONE_COLOR, thickness=2.0, thickness_units="screen")
        # The lever arm of every chain joint: the effector moves by delta x (p - j_k).
        levers = [[joints[k], joints[effector]] for k in links]
        server.scene.add_line_segments("/chain", np.array(levers, dtype=np.float32), colors=CHAIN_COLOR, thickness=3.0, thickness_units="screen")
        server.scene.add_point_cloud("/chain_joints", joints[links].astype(np.float32), colors=CHAIN_COLOR, point_size=0.04, point_shape="circle")
        goal = np.asarray(target.position, dtype=np.float64)
        server.scene.add_line_segments(
            "/error", np.array([[joints[effector], goal]], dtype=np.float32), colors=TARGET_COLOR, thickness=1.5, thickness_units="screen"
        )

        rows = ["| joint | axis-angle relative to the parent (deg) | angle (deg) |", "|---|---|---:|"]
        for k in links:
            r = np.degrees(pose[k])
            rows.append(f"| {JOINT_NAMES[k]} | ({r[0]:6.1f}, {r[1]:6.1f}, {r[2]:6.1f}) | {np.linalg.norm(r):.1f} |")
        gui_numbers.content = "\n".join(
            [
                f"Distance to the target: **{np.linalg.norm(goal - joints[effector]):.3f} m**. "
                f"The Jacobian is 3 x {3 * len(links)}: three columns for every joint in the chain (orange).",
                "",
                *rows,
            ]
        )

    def place_target() -> None:
        """Put the target on the effector, a little in front of it."""
        nonlocal syncing
        joints = kinematics()[:, :3, 3]
        syncing = True
        try:
            target.position = joints[EFFECTORS[gui_effector.value]] + [0.0, 0.0, 0.15]
        finally:
            syncing = False

    def on_target(_) -> None:
        if syncing:
            return
        with lock:
            if gui_follow.value:
                for _ in range(15):
                    if ik_step() < 1e-3:
                        break
            draw()

    def on_step(_) -> None:
        with lock:
            ik_step()
            draw()

    def on_effector(_) -> None:
        with lock:
            gui_length.max = len(chain_to_root(parents, EFFECTORS[gui_effector.value]))
            place_target()
            draw()

    def on_reset(_) -> None:
        with lock:
            pose[:] = 0.0
            place_target()
            draw()

    def on_change(_) -> None:
        with lock:
            draw()

    target.on_update(on_target)
    gui_step.on_click(on_step)
    gui_effector.on_update(on_effector)
    gui_reset.on_click(on_reset)
    gui_length.on_update(on_change)

    pose[16], pose[17] = [0.0, 0.0, -1.2], [0.0, 0.0, 1.2]  # arms down
    place_target()
    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
