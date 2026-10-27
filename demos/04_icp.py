"""Lecture 4.1: Iterative Closest Point (ICP) in 3D.

The orange body (source X) has to be aligned to the blue scan (target Y): the same
body, rotated, scaled and moved, sampled at other points, with noise, outliers or a
missing arm if you like. ICP repeats two easy steps:

    1. correspondences: pair every target point y_i with its closest source point,
    2. Procrustes: the s, R, t that bring the paired points closest (vh/icp.py).

Trimming drops the worst pairs before step 2, which keeps outliers from pulling.

    python demos/04_icp.py
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser

from vh import SMPLModel
from vh.icp import icp_step, transform
from vh.rotations import exp_map, log_map
from vh.scene import BODY_POSES, body, sample, stick_figure

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_POINTS = 1500
SOURCE_COLOR, TARGET_COLOR, OUTLIER_COLOR, LINK_COLOR = (235, 120, 30), (40, 110, 220), (150, 150, 150), (170, 170, 170)
INITS = ["Align the centroids", "R = I, t = 0"]
MAX_ITERATIONS = 80


def load_shape(model_dir: Path):
    """An SMPL body reaching forwards (asymmetric, so the alignment is unique), or a stick figure."""
    paths = sorted(model_dir.glob("smpl_*.npz"))
    if paths:
        model = SMPLModel.load(next((p for p in paths if p.stem == "smpl_neutral"), paths[0]))
        return body(model, BODY_POSES[0]), True
    return stick_figure(), False


def interpolate(f0, f1, u: float, center: np.ndarray):
    """In between two transforms, about the body's centre so it travels instead of swinging:
    scale and the centre's position linearly, rotation along the shortest arc."""
    (s0, R0, _), (s1, R1, _) = f0, f1
    c0, c1 = transform(center[None], *f0)[0], transform(center[None], *f1)[0]
    s, R, c = (1 - u) * s0 + u * s1, R0 @ exp_map(u * log_map(R0.T @ R1)), (1 - u) * c0 + u * c1
    return s, R, c - s * R @ center


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    shape, has_smpl = load_shape(args.model_dir)
    X = sample(shape, NUM_POINTS, seed=1).points

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/floor", width=8.0, height=8.0, plane="xz", cell_size=0.25, section_size=1.0)
    server.initial_camera.position = (0.6, 1.9, 4.6)
    server.initial_camera.look_at = (0.6, 0.9, 0.0)
    lock = threading.RLock()

    state = {"seed": 4, "f": None, "j": 0, "rms": [], "pairs": None, "done": False, "running": False}
    target = {"Y": None, "outlier": None, "truth": None}
    source_cloud = server.scene.add_point_cloud("/source", X.astype(np.float32), colors=SOURCE_COLOR, point_size=0.012, point_shape="circle")

    with server.gui.add_folder("Iterate"):
        gui_run = server.gui.add_button("Run")
        gui_step = server.gui.add_button("Step", hint="One iteration: correspondences, then Procrustes.")
        gui_reset = server.gui.add_button("Reset")
        gui_plot = server.gui.add_uplot(
            (np.zeros(1), np.ones(1)),
            ({"label": "iteration j"}, {"label": "RMS of the kept pairs", "stroke": "#2870dc", "width": 2}),
            scales={"x": {"time": False}, "y": {"distr": 3}},
            aspect=2.0,
        )
    gui_numbers = server.gui.add_markdown("")
    with server.gui.add_folder("Target scan"):
        gui_new = server.gui.add_button("New random target")
        gui_noise = server.gui.add_slider("Noise sigma (m)", min=0.0, max=0.04, step=0.002, initial_value=0.01)
        gui_outliers = server.gui.add_checkbox("Add 15% outliers", False)
        gui_partial = server.gui.add_checkbox("Partial scan (left arm missing)", False)
    with server.gui.add_folder("ICP options"):
        gui_init = server.gui.add_dropdown("Start from", INITS, initial_value=INITS[0])
        gui_trim = server.gui.add_slider("Trim the worst pairs (%)", min=0, max=40, step=5, initial_value=0)
        gui_links = server.gui.add_checkbox("Show correspondences", True)
    if not has_smpl:
        server.gui.add_markdown("_No SMPL model installed: the stick figure is round, so ICP cannot tell how far to turn it._")

    def make_target() -> None:
        rng = np.random.default_rng(state["seed"] * 7919 + 5)
        axis = rng.normal(size=3)
        R = exp_map(axis / np.linalg.norm(axis) * np.radians(10 + 15 * rng.random()))
        s, t = 0.8 + 0.4 * rng.random(), np.array([1.0 + 0.6 * rng.random(), 0.8 * rng.random() - 0.4, 0.5 * rng.random() - 0.25])
        points = sample(shape, NUM_POINTS, seed=100 + state["seed"]).points  # other points on the same body
        if gui_partial.value:
            points = points[~((points[:, 0] > 0.15) & (points[:, 1] > 1.0))]  # the left arm and shoulder
        Y = transform(points, s, R, t) + gui_noise.value * rng.normal(size=points.shape)
        outlier = np.zeros(len(Y), dtype=bool)
        if gui_outliers.value:
            lo, hi = Y.min(axis=0) - 0.3, Y.max(axis=0) + 0.3
            extra = lo + (hi - lo) * rng.random((int(round(0.15 * len(Y))), 3))
            Y, outlier = np.concatenate([Y, extra]), np.concatenate([outlier, np.ones(len(extra), dtype=bool)])
        target.update(Y=Y, outlier=outlier, truth=(s, R, t))
        server.scene.add_point_cloud("/target", Y[~outlier].astype(np.float32), colors=TARGET_COLOR, point_size=0.012, point_shape="circle")
        server.scene.add_point_cloud("/outliers", Y[outlier].astype(np.float32) if outlier.any() else np.zeros((1, 3), np.float32),
                                     colors=OUTLIER_COLOR, point_size=0.012, point_shape="circle", visible=bool(outlier.any()))

    def errors(f):
        s_true, R_true, t_true = target["truth"]
        rotation = np.degrees(np.linalg.norm(log_map(R_true.T @ f[1])))
        chest = np.array([[0.0, 1.3, 0.05]])
        position = np.linalg.norm(transform(chest, *f) - transform(chest, s_true, R_true, t_true))
        return rotation, position, s_true

    def draw(f) -> None:
        moved = transform(X, *f)
        source_cloud.points = moved.astype(np.float32)
        pairs = state["pairs"]
        if gui_links.value and pairs is not None:
            every = pairs[::6]
            segments = np.stack([moved[every[:, 0]], target["Y"][every[:, 1]]], axis=1)
            server.scene.add_line_segments("/links", segments.astype(np.float32), colors=LINK_COLOR, thickness=1.0, thickness_units="screen")
        else:
            server.scene.remove_by_name("/links")

    def report() -> None:
        f, rms = state["f"], state["rms"]
        rotation, position, s_true = errors(f)
        lines = [
            f"| iteration j | {state['j']} |", "|---|---:|",
            f"| RMS of the kept pairs | {rms[-1]:.4f} m |" if rms else "| RMS of the kept pairs | - |",
            f"| scale s (true) | {f[0]:.3f} ({s_true:.3f}) |",
            f"| rotation error | {rotation:.2f} deg |",
            f"| position error (chest) | {100 * position:.1f} cm |",
        ]
        if state["done"]:
            head = f"Stopped after {MAX_ITERATIONS} iterations" if state["j"] >= MAX_ITERATIONS else f"Converged after {state['j']} iterations"
            if rotation < 3:
                verdict = f"{head}: the rotation is within {rotation:.1f} deg of the truth."
            elif gui_outliers.value and gui_trim.value == 0:
                verdict = f"{head}, but the rotation is off by {rotation:.0f} deg: the outliers pull the estimate. Trim the worst pairs."
            else:
                verdict = f"{head}, but the rotation is off by {rotation:.0f} deg: ICP has settled in a local minimum."
            lines = [f"**{verdict}**", ""] + lines
        gui_numbers.content = "\n".join(lines)
        if rms:
            gui_plot.data = (np.arange(len(rms), dtype=np.float64), np.maximum(np.array(rms), 1e-5))
        gui_run.label = "Pause" if state["running"] else "Run"

    def reset() -> None:
        state["running"] = False
        Y = target["Y"]
        offset = Y.mean(axis=0) - X.mean(axis=0) if gui_init.value == INITS[0] else np.zeros(3)
        state.update(f=(1.0, np.eye(3), offset), j=0, rms=[], pairs=None, done=False)
        gui_plot.data = (np.zeros(1), np.ones(1))
        draw(state["f"])
        report()

    def iterate(frames: int) -> None:
        """One ICP iteration, with the source gliding to its new place over a few frames."""
        f_new, rms, pairs = icp_step(X, target["Y"], *state["f"], trim=gui_trim.value / 100)
        state["rms"].append(rms)
        state["pairs"] = pairs
        if len(state["rms"]) > 1 and abs(state["rms"][-2] - rms) < 1e-5 or state["j"] >= MAX_ITERATIONS:
            state["done"], state["running"] = True, False
            draw(state["f"])
            report()
            return
        f_old = state["f"]
        for u in np.linspace(0.0, 1.0, frames + 1)[1:]:
            draw(interpolate(f_old, f_new, float(u), X.mean(axis=0)))
            time.sleep(1.0 / 60.0)
        state["f"] = f_new
        state["j"] += 1
        report()

    def run_loop() -> None:
        while True:
            with lock:
                if not state["running"] or state["done"]:
                    break
                iterate(frames=6)

    def on_run(_) -> None:
        with lock:
            if state["done"]:
                return
            state["running"] = not state["running"]
            report()
        if state["running"]:
            threading.Thread(target=run_loop, daemon=True).start()

    def on_step(_) -> None:
        with lock:
            state["running"] = False
            if not state["done"]:
                iterate(frames=14)

    def on_reset(_) -> None:
        with lock:
            reset()

    def on_new_target(_) -> None:
        with lock:
            state["running"] = False
            if _ is not None and _.target is gui_new:
                state["seed"] += 1
            make_target()
            reset()

    def on_links(_) -> None:
        with lock:
            draw(state["f"])

    gui_run.on_click(on_run)
    gui_step.on_click(on_step)
    gui_reset.on_click(on_reset)
    gui_new.on_click(on_new_target)
    for control in [gui_noise, gui_outliers, gui_partial]:
        control.on_update(on_new_target)
    for control in [gui_init, gui_trim]:
        control.on_update(on_reset)
    gui_links.on_update(on_links)

    make_target()
    reset()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
