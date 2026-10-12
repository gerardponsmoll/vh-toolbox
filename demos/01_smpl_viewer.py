"""Lecture 1: interactive SMPL viewer.

Move the sliders to change body shape (beta) and pose (theta), and use the
"Display" options to look at each step of the model separately. Click a body
part (or a joint, with the skeleton on) to select its joint, then drag the
rings to rotate it.

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
import viser.transforms as tf

from vh import JOINT_NAMES, SMPLModel, forward, rigid_weights
from vh.geometry import ray_mesh_intersection

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
JOINT_COLOR = (40, 40, 40)
SELECTED_COLOR = (230, 30, 30)
SEE_THROUGH = 0.35  # mesh opacity while the skeleton is shown
JOINT_RADIUS = 0.02  # metres, size of the joint spheres
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
        gui_skeleton = server.gui.add_checkbox(
            "Skeleton", False, hint="Show joints and bones. Click a joint to select it."
        )
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
                min=-3.15,
                max=3.15,
                step=0.01,
                initial_value=0.0,
                hint="Axis-angle component, in radians.",
            )
            for label in ("x", "y", "z")
        ]
        gui_gizmo = server.gui.add_checkbox(
            "Rotation handle", True, hint="Drag the rings on the selected joint to rotate it."
        )
        gui_pose_reset = server.gui.add_button("Reset pose")
        gui_pose_random = server.gui.add_button("Random pose")

    # ------------------------------------------------------------- drawing
    mesh_handle = None
    mesh_key = None  # what kind of mesh node is currently in the scene
    world_rotations = np.tile(np.eye(3), (num_joints, 1, 1))  # of each joint, from the last draw
    shown_vertices = models[first].v_template  # the mesh as currently displayed
    shown_joints = np.zeros((num_joints, 3))  # and its joints
    last_handle_use = 0.0  # time of the last interaction with the rotation handle

    # Clicks are handled for the whole scene rather than per object, because
    # viser draws a highlight around objects that take clicks themselves.
    @server.scene.on_click()
    def on_scene_click(event) -> None:
        """Select the joint under the cursor.

        With the skeleton shown, a click on a joint sphere selects that joint.
        Otherwise the click selects the joint that this point of the body is
        mostly attached to, according to the skinning weights.
        """
        if time.time() - last_handle_use < 0.3:  # the click was on the rotation handle
            return
        origin = np.asarray(event.ray_origin, dtype=np.float64)
        direction = np.asarray(event.ray_direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)

        if gui_skeleton.value:
            to_joints = shown_joints - origin
            along = to_joints @ direction
            distance = np.linalg.norm(to_joints - along[:, None] * direction, axis=1)
            along[(distance > 1.5 * JOINT_RADIUS) | (along <= 0)] = np.inf
            if np.isfinite(along).any():
                gui_joint.value = joint_names[int(along.argmin())]
                return

        model = models[gui_model.value]
        hit = ray_mesh_intersection(origin, direction, shown_vertices, model.faces)
        if hit is not None:
            face, barycentric = hit
            weights_at_click = barycentric @ model.weights[model.faces[face]]
            gui_joint.value = joint_names[int(weights_at_click.argmax())]

    joint_handles = [
        server.scene.add_icosphere(f"/skeleton/joints/{k}", radius=JOINT_RADIUS, color=JOINT_COLOR, visible=False)
        for k in range(num_joints)
    ]
    gizmo = server.scene.add_transform_controls(
        "/rotation_handle",
        scale=0.25,
        disable_axes=True,
        disable_sliders=True,
        depth_test=False,
        visible=False,
    )

    def draw(move_gizmo: bool = True) -> None:
        nonlocal mesh_handle, mesh_key, world_rotations, shown_vertices, shown_joints, syncing
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
                    opacity=SEE_THROUGH if gui_skeleton.value else None,
                )
            else:
                mesh = trimesh.Trimesh(vertices, model.faces, process=False)
                alpha = SEE_THROUGH if gui_skeleton.value else 1.0
                rgba = np.concatenate([colors, np.full((len(colors), 1), alpha)], axis=1)
                mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
                mesh_handle = server.scene.add_mesh_trimesh("/body", mesh)
            mesh_key = key
            shown_vertices, shown_joints = vertices, joints

            has_parent = model.parents >= 0
            bones = np.stack([joints[model.parents[has_parent]], joints[has_parent]], axis=1)
            server.scene.add_line_segments(
                "/skeleton/bones",
                bones.astype(np.float32),
                colors=JOINT_COLOR,
                thickness=3.0,
                thickness_units="screen",
                visible=gui_skeleton.value,
            )
            for k, handle in enumerate(joint_handles):
                handle.position = joints[k]
                handle.color = SELECTED_COLOR if k == selected else JOINT_COLOR
                handle.visible = gui_skeleton.value

            # The rotation handle sits on the selected joint, aligned with the
            # joint's frame. It is left alone while the user is dragging it.
            world_rotations = out.transforms[:, :3, :3]
            if move_gizmo:
                syncing = True
                try:
                    gizmo.position = out.joints[selected]
                    gizmo.wxyz = tf.SO3.from_matrix(world_rotations[selected]).wxyz
                    gizmo.visible = gui_gizmo.value and stage == len(STAGES) - 1
                finally:
                    syncing = False

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

    def on_gizmo(_) -> None:
        """Turn the handle's world rotation into the joint's rotation relative to its parent."""
        nonlocal last_handle_use
        last_handle_use = time.time()
        if syncing:
            return
        k = joint_names.index(gui_joint.value)
        parent = models[gui_model.value].parents[k]
        parent_rotation = np.eye(3) if parent < 0 else world_rotations[parent]
        local_rotation = parent_rotation.T @ tf.SO3(np.array(gizmo.wxyz)).as_matrix()
        pose[k] = tf.SO3.from_matrix(local_rotation).log()
        set_sliders(gui_axis, pose[k])
        draw(move_gizmo=False)

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

    for handle in [
        gui_model,
        gui_stage,
        gui_blend,
        gui_skinning,
        gui_color,
        gui_skeleton,
        gui_wireframe,
        gui_gizmo,
        *gui_betas,
    ]:
        handle.on_update(on_display)
    for handle in gui_axis:
        handle.on_update(on_axis)
    gui_joint.on_update(on_joint)
    gizmo.on_update(on_gizmo)
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
