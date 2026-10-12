"""Lecture 1: interactive SMPL viewer.

Move the sliders to change body shape (beta) and pose (theta), and use the
"Display" options to look at each step of the model separately.

    python demos/01_smpl_viewer.py
"""

from __future__ import annotations

import argparse
import colorsys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from vh import JOINT_NAMES, SMPLModel, forward, rigid_weights

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

STAGES = [
    "1. Template",
    "2. + shape blendshapes",
    "3. + pose blendshapes",
    "4. + skinning (final)",
]
COLORS = [
    "Plain",
    "Skinning weights: all joints",
    "Skinning weights: selected joint",
    "Pose blendshape offset",
]
SKIN = (222, 184, 160)
MAX_OFFSET = 0.02  # metres; pose blendshape offsets at or above this are fully red


def joint_palette(num_joints: int) -> np.ndarray:
    hues = (np.arange(num_joints) * 0.618034) % 1.0  # golden ratio: neighbours differ
    return np.array([colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues])


def ramp(t: np.ndarray) -> np.ndarray:
    """Scalars in [0, 1] -> light grey to red."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    return (1.0 - t) * np.array([0.85, 0.85, 0.85]) + t * np.array([0.85, 0.1, 0.1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    paths = {p.stem.removeprefix("smpl_"): p for p in sorted(args.model_dir.glob("smpl_*.npz"))}
    if not paths:
        raise SystemExit(
            f"No models in {args.model_dir}. Download SMPL and run scripts/prepare_smpl.py first (see README)."
        )
    models = {name: SMPLModel.load(path) for name, path in paths.items()}
    first = "neutral" if "neutral" in models else next(iter(models))
    num_joints = models[first].num_joints
    num_betas = min(10, models[first].num_betas)
    joint_names = (JOINT_NAMES + [f"joint_{k}" for k in range(len(JOINT_NAMES), num_joints)])[:num_joints]
    palette = joint_palette(num_joints)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    floor = float(models[first].v_template[:, 1].min())
    server.scene.add_grid("/floor", width=4.0, height=4.0, plane="xz", position=(0.0, floor, 0.0))

    pose = np.zeros((num_joints, 3))
    lock = threading.Lock()
    syncing = False  # True while we set sliders from code, so their callbacks stay quiet

    # ---------------------------------------------------------------- GUI
    with server.gui.add_folder("Display"):
        gui_model = server.gui.add_dropdown("Model", list(models), initial_value=first)
        gui_stage = server.gui.add_dropdown("Show", STAGES, initial_value=STAGES[-1])
        gui_blend = server.gui.add_checkbox(
            "Pose blendshapes", True, hint="Turn off to see linear blend skinning on its own."
        )
        gui_skinning = server.gui.add_checkbox(
            "Blend skinning",
            True,
            hint="Turn off to attach each vertex to one joint only (weight 1, all others 0).",
        )
        gui_color = server.gui.add_dropdown("Colour", COLORS, initial_value=COLORS[0])
        gui_skeleton = server.gui.add_checkbox("Skeleton", False)
        gui_wireframe = server.gui.add_checkbox("Wireframe", False)

    with server.gui.add_folder("Shape (beta)"):
        gui_betas = [
            server.gui.add_slider(f"beta {i}", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
            for i in range(num_betas)
        ]
        gui_shape_reset = server.gui.add_button("Reset shape")
        gui_shape_random = server.gui.add_button("Random shape")

    with server.gui.add_folder("Pose (theta)"):
        gui_joint = server.gui.add_dropdown("Joint", joint_names, initial_value=joint_names[0])
        gui_axis = [
            server.gui.add_slider(
                label,
                min=-3.14,
                max=3.14,
                step=0.01,
                initial_value=0.0,
                hint="Axis-angle component, in radians.",
            )
            for label in ("x", "y", "z")
        ]
        gui_pose_reset = server.gui.add_button("Reset pose")
        gui_pose_random = server.gui.add_button("Random pose")

    # ------------------------------------------------------------- drawing
    mesh_handle = None
    mesh_key = None  # what kind of mesh node is currently in the scene

    def draw() -> None:
        nonlocal mesh_handle, mesh_key
        with lock:
            model = models[gui_model.value]
            betas = np.array([g.value for g in gui_betas])
            selected = joint_names.index(gui_joint.value)
            stage = STAGES.index(gui_stage.value)
            out = forward(
                model,
                betas,
                pose,
                pose_blendshapes=gui_blend.value,
                blend_skinning=gui_skinning.value,
            )
            weights = model.weights if gui_skinning.value else rigid_weights(model.weights)

            rest_joints = model.J_regressor @ model.v_template
            vertices, joints = [
                (model.v_template, rest_joints),
                (out.v_shaped, out.joints_rest),
                (out.v_posed, out.joints_rest),
                (out.vertices, out.joints),
            ][stage]

            color_mode = gui_color.value
            if color_mode == COLORS[1]:
                colors = weights @ palette
            elif color_mode == COLORS[2]:
                colors = ramp(weights[:, selected])
            elif color_mode == COLORS[3]:
                colors = ramp(np.linalg.norm(out.v_posed - out.v_shaped, axis=1) / MAX_OFFSET)
            else:
                colors = None

            # A plain mesh can be moved in place. Per-vertex colours need a
            # trimesh node, which is replaced on every update.
            key = (gui_model.value, colors is None, gui_wireframe.value, gui_skeleton.value)
            if colors is None and key == mesh_key:
                mesh_handle.vertices = vertices.astype(np.float32)
            elif colors is None:
                mesh_handle = server.scene.add_mesh_simple(
                    "/body",
                    vertices.astype(np.float32),
                    model.faces,
                    color=SKIN,
                    wireframe=gui_wireframe.value,
                    opacity=0.5 if gui_skeleton.value else None,
                )
            else:
                mesh = trimesh.Trimesh(vertices, model.faces, process=False)
                alpha = 0.5 if gui_skeleton.value else 1.0
                rgba = np.concatenate([colors, np.full((len(colors), 1), alpha)], axis=1)
                mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
                mesh_handle = server.scene.add_mesh_trimesh("/body", mesh)
            mesh_key = key

            has_parent = model.parents >= 0
            bones = np.stack([joints[model.parents[has_parent]], joints[has_parent]], axis=1)
            joint_colors = np.tile(np.array([[40, 40, 40]], dtype=np.uint8), (num_joints, 1))
            joint_colors[selected] = (230, 30, 30)
            server.scene.add_line_segments(
                "/skeleton/bones",
                bones.astype(np.float32),
                colors=(40, 40, 40),
                thickness=3.0,
                thickness_units="screen",
                visible=gui_skeleton.value,
            )
            server.scene.add_point_cloud(
                "/skeleton/joints",
                joints.astype(np.float32),
                colors=joint_colors,
                point_size=0.025,
                point_shape="circle",
                precision="float32",
                visible=gui_skeleton.value,
            )

    # ----------------------------------------------------------- callbacks
    def set_sliders(handles, values) -> None:
        nonlocal syncing
        syncing = True
        try:
            for handle, value in zip(handles, values):
                handle.value = float(value)
        finally:
            syncing = False

    def on_display(_) -> None:
        if not syncing:
            draw()

    def on_axis(_) -> None:
        if syncing:
            return
        pose[joint_names.index(gui_joint.value)] = [g.value for g in gui_axis]
        draw()

    def on_joint(_) -> None:
        set_sliders(gui_axis, pose[joint_names.index(gui_joint.value)])
        draw()

    def on_shape_reset(_) -> None:
        set_sliders(gui_betas, np.zeros(num_betas))
        draw()

    def on_shape_random(_) -> None:
        set_sliders(gui_betas, np.clip(np.random.randn(num_betas), -3.0, 3.0).round(1))
        draw()

    def on_pose_reset(_) -> None:
        pose[:] = 0.0
        on_joint(None)

    def on_pose_random(_) -> None:
        pose[1:] = np.random.randn(num_joints - 1, 3) * 0.2
        on_joint(None)

    for handle in [gui_model, gui_stage, gui_blend, gui_skinning, gui_color, gui_skeleton, gui_wireframe, *gui_betas]:
        handle.on_update(on_display)
    for handle in gui_axis:
        handle.on_update(on_axis)
    gui_joint.on_update(on_joint)
    gui_shape_reset.on_click(on_shape_reset)
    gui_shape_random.on_click(on_shape_random)
    gui_pose_reset.on_click(on_pose_reset)
    gui_pose_random.on_click(on_pose_random)

    draw()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
