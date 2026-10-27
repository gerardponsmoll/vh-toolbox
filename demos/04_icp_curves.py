"""Lecture 4.1: ICP step by step, on the two curves of the slides.

Iterative Closest Point alternates two problems that are easy on their own:

    1. with f fixed, find the closest points (grey lines),
    2. with the pairs fixed, solve for the best f = {s, R, t} with Procrustes (dashed green),

and then applies the new f. Both steps lower the same error, so it never increases,
but where ICP ends depends on where it starts (the initial rotation theta_0).

Step 2 can also take gradient steps on E instead of the closed-form Procrustes solution,
as one does for a general deformation model: then the step size alpha matters.

    python demos/04_icp_curves.py
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import viser

from vh.icp import correspondences, gradient_steps, icp_step, procrustes, transform

CURVES = Path(__file__).resolve().parent / "data" / "icp_curves.json"
SOURCE_COLOR, TARGET_COLOR, START_COLOR, NEXT_COLOR, LINK_COLOR = (235, 120, 30), (40, 110, 220), (150, 150, 150), (40, 160, 70), (150, 150, 150)
INITS = ["Align the centroids", "R = I, t = 0"]
DIRECTIONS = ["Target -> source (lecture)", "Source -> target"]
PHASES = ["Find correspondences", "Solve Procrustes", "Apply the transform"]
SOLVERS = ["Procrustes (closed form)", "Gradient descent"]
ALIGNED_RMS = 0.03
MAX_ITERATIONS = 120


def rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def angle(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def segments(points: np.ndarray, dashed: bool = False) -> np.ndarray:
    """A closed curve as line segments in the z = 0 plane; dashed keeps every other one."""
    p = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
    seg = np.stack([p, np.roll(p, -1, axis=0)], axis=1)
    return (seg[::2] if dashed else seg).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    curves = json.loads(CURVES.read_text())
    X, Y = np.array(curves["X"]), np.array(curves["Y"])
    mx, my = X.mean(axis=0), Y.mean(axis=0)

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.set_up_direction("+y")
    server.scene.add_line_segments("/target", segments(Y), colors=TARGET_COLOR, thickness=4.0, thickness_units="screen")
    server.scene.add_label("/target/name", "target Y", position=(float(Y[:, 0].mean()), float(Y[:, 1].min()) - 0.3, 0.0))
    lock = threading.RLock()
    state = {"f": None, "j": 0, "phase": 0, "pairs": None, "next": None, "rms": [], "done": False, "running": False, "message": ""}

    with server.gui.add_folder("Iterate"):
        gui_step = server.gui.add_button(PHASES[0])
        gui_run = server.gui.add_button("Run")
        gui_reset = server.gui.add_button("Reset")
        gui_plot = server.gui.add_uplot(
            (np.zeros(1), np.ones(1), np.ones(1)),
            ({"label": "iteration j"}, {"label": "RMS", "stroke": "#2870dc", "width": 2},
             {"label": "Procrustes-ICP, same start", "stroke": "#888888", "width": 1.5, "dash": [6, 4]}),
            scales={"x": {"time": False}, "y": {"distr": 3}},
            aspect=2.0,
        )
    gui_numbers = server.gui.add_markdown("")
    with server.gui.add_folder("Initialisation f0"):
        gui_init = server.gui.add_dropdown("Start from", INITS, initial_value=INITS[0])
        gui_theta = server.gui.add_slider("Initial rotation theta_0 (deg)", min=-180, max=180, step=5, initial_value=40)
    with server.gui.add_folder("Pairs and model"):
        gui_direction = server.gui.add_dropdown("Pairs", DIRECTIONS, initial_value=DIRECTIONS[0])
        gui_scale = server.gui.add_checkbox("Estimate scale s", True)
    with server.gui.add_folder("Step 2: solve with"):
        gui_solver = server.gui.add_dropdown("Solver", SOLVERS, initial_value=SOLVERS[0])
        gui_alpha = server.gui.add_slider("Step size alpha", min=0.005, max=2.0, step=0.005, initial_value=0.2)
        gui_steps = server.gui.add_slider("Gradient steps per iteration", min=1, max=20, step=1, initial_value=5)
    server.gui.add_markdown("Dashed grey: f0(X). Dashed green: the solution of step 2 before it is applied. "
                            "Gradient descent: E is averaged over the pairs, so alpha does not depend on their number.")
    reference = {"rms": np.ones(1)}  # Procrustes-ICP from the same start, to compare gradient descent with

    def start_transform():
        """Rotate by theta_0 about the source centroid, then keep it in place or move it onto the target centroid."""
        R = rotation(np.radians(gui_theta.value))
        base = my if gui_init.value == INITS[0] else mx
        return 1.0, R, base - R @ mx

    def between(f0, f1, u: float):
        """Interpolate about the source centroid, so the curve travels instead of swinging."""
        (s0, R0, _), (s1, R1, _) = f0, f1
        c0, c1 = transform(mx[None], *f0)[0], transform(mx[None], *f1)[0]
        d = np.arctan2(np.sin(angle(R1) - angle(R0)), np.cos(angle(R1) - angle(R0)))
        s, R, c = (1 - u) * s0 + u * s1, rotation(angle(R0) + u * d), (1 - u) * c0 + u * c1
        return s, R, c - s * R @ mx

    def draw(f) -> None:
        FX = transform(X, *f)
        server.scene.add_line_segments("/start", segments(transform(X, *start_transform()), dashed=True),
                                       colors=START_COLOR, thickness=1.5, thickness_units="screen")
        server.scene.add_line_segments("/source", segments(FX), colors=SOURCE_COLOR, thickness=4.0, thickness_units="screen")
        server.scene.add_label("/source/name", f"f{state['j']}(X)", position=(float(FX[:, 0].mean()), float(FX[:, 1].max()) + 0.25, 0.0))
        if state["phase"] >= 1 and state["pairs"] is not None:
            p = state["pairs"]
            links = np.stack([np.c_[FX[p[:, 0]], np.zeros(len(p))], np.c_[Y[p[:, 1]], np.zeros(len(p))]], axis=1)
            server.scene.add_line_segments("/links", links.astype(np.float32), colors=LINK_COLOR, thickness=1.0, thickness_units="screen")
        else:
            server.scene.remove_by_name("/links")
        if state["phase"] == 2 and state["next"] is not None:
            server.scene.add_line_segments("/next", segments(transform(X, *state["next"]), dashed=True),
                                           colors=NEXT_COLOR, thickness=2.5, thickness_units="screen")
        else:
            server.scene.remove_by_name("/next")

    def report() -> None:
        s, R, t = state["f"]
        rms = state["rms"]
        rows = ["| | |", "|---|---:|", f"| iteration j | {state['j']} |",
                f"| s, theta | {s:.3f}, {np.degrees(angle(R)):.1f} deg |", f"| t | ({t[0]:.2f}, {t[1]:.2f}) |"]
        if rms:
            rows.append(f"| RMS | {rms[-1]:.4f} |")
        gui_numbers.content = "\n".join(([f"**{state['message']}**", ""] if state["message"] else []) + rows)
        if rms:
            ref = reference["rms"][np.minimum(np.arange(len(rms)), len(reference["rms"]) - 1)]
            if gui_solver.value == SOLVERS[0]:
                ref = np.full(len(rms), np.nan)
            gui_plot.data = (np.arange(len(rms), dtype=np.float64), np.clip(np.array(rms), 1e-5, 1e3), ref)
        names = [PHASES[0], "Solve Procrustes" if gui_solver.value == SOLVERS[0] else "Gradient steps", PHASES[2]]
        gui_step.label = "Done" if state["done"] else names[min(state["phase"], 2)]
        gui_run.label = "Pause" if state["running"] else "Run"

    def finish_message() -> str:
        rms, s = state["rms"][-1], state["f"][0]
        if not np.isfinite(rms) or rms > 7:
            return (f"Diverged at iteration {state['j']}: alpha is too large for the curvature of E, "
                    "so each step overshoots the minimum.")
        if gui_solver.value == SOLVERS[1] and rms > 1.5 * reference["rms"][-1] + 2e-4:
            return (f"Stopped after {state['j']} iterations with RMS {rms:.3f}: with this alpha the gradient steps are "
                    "too small to keep up with Procrustes-ICP from the same start.")
        if gui_direction.value == DIRECTIONS[1] and gui_scale.value and s < 0.5:
            return (f"Converged after {state['j']} iterations, but the source has shrunk to s = {s:.2f}. When every source "
                    "point picks its closest target point, a smaller source always fits better: pair from the target to the source.")
        if rms < ALIGNED_RMS:
            return f"Converged after {state['j']} iterations with RMS {rms:.4f}: the shapes are aligned."
        return (f"Converged after {state['j']} iterations, but RMS {rms:.3f} is a local minimum: "
                "the shapes are not aligned. Try another theta_0.")

    def step(frames: int) -> None:
        """Advance one phase: correspondences, Procrustes, or apply (animated)."""
        if state["done"]:
            return
        if state["phase"] == 0:
            pairs, sq_dist = correspondences(X, Y, *state["f"], target_to_source=gui_direction.value == DIRECTIONS[0])
            state["pairs"], state["phase"] = pairs, 1
            state["rms"].append(float(np.sqrt(sq_dist.mean())))
            r = state["rms"]
            diverged = not np.isfinite(r[-1]) or r[-1] > 7
            if diverged or (len(r) > 1 and abs(r[-2] - r[-1]) < 1e-5) or state["j"] >= MAX_ITERATIONS:
                state["done"], state["running"], state["phase"] = True, False, 0
                state["message"] = finish_message()
        elif state["phase"] == 1:
            p = state["pairs"]
            if gui_solver.value == SOLVERS[0]:
                state["next"] = procrustes(X[p[:, 0]], Y[p[:, 1]], scale=gui_scale.value)
            else:
                state["next"] = gradient_steps(X, Y, p, *state["f"], alpha=gui_alpha.value, steps=int(gui_steps.value),
                                               scale=gui_scale.value)
            state["phase"] = 2
        else:
            f0, f1 = state["f"], state["next"]
            state["phase"] = 3  # hide the helper lines while the curve moves
            for u in np.linspace(0.0, 1.0, frames + 1)[1:]:
                draw(between(f0, f1, float(u)))
                time.sleep(1.0 / 60.0)
            state.update(f=f1, next=None, pairs=None, phase=0, j=state["j"] + 1)
        draw(state["f"])
        report()

    def frame(client) -> None:
        """Look straight at the curves (the target and the start), leaving room for the panel on the right."""
        points = np.concatenate([Y, transform(X, *start_transform())])
        lo, hi = points.min(axis=0), points.max(axis=0)
        width, height = hi - lo
        fov = client.camera.fov
        dist = 1.35 * max(height, width / (0.7 * client.camera.aspect)) / (2 * np.tan(fov / 2))
        center = (lo + hi) / 2 + [0.15 * width / 0.7, 0.0]  # the panel covers about 30% on the right
        with client.atomic():
            client.camera.up_direction = (0.0, 1.0, 0.0)
            client.camera.position = (center[0], center[1], dist)
            client.camera.look_at = (center[0], center[1], 0.0)

    server.on_client_connect(lambda client: frame(client))

    def reset() -> None:
        for client in server.get_clients().values():
            frame(client)
        state.update(f=start_transform(), j=0, phase=0, pairs=None, next=None, rms=[], done=False, running=False, message="")
        f, ref = start_transform(), []
        for _ in range(MAX_ITERATIONS):
            f, rms, _ = icp_step(X, Y, *f, scale=gui_scale.value, target_to_source=gui_direction.value == DIRECTIONS[0])
            ref.append(rms)
            if len(ref) > 1 and abs(ref[-2] - ref[-1]) < 1e-5:
                break
        reference["rms"] = np.array(ref)
        gui_plot.data = (np.zeros(1), np.ones(1), np.ones(1))
        draw(state["f"])
        report()

    def run_loop() -> None:
        while True:
            with lock:
                if not state["running"] or state["done"]:
                    break
                step(frames=12)
                pause = 0.04 if state["phase"] == 1 else 0.16
            time.sleep(pause)

    def on_step(_) -> None:
        with lock:
            state["running"] = False
            step(frames=24)

    def on_run(_) -> None:
        with lock:
            if state["done"]:
                return
            state["running"] = not state["running"]
            report()
        if state["running"]:
            threading.Thread(target=run_loop, daemon=True).start()

    def on_reset(_) -> None:
        with lock:
            reset()

    gui_step.on_click(on_step)
    gui_run.on_click(on_run)
    gui_reset.on_click(on_reset)
    for control in [gui_init, gui_theta, gui_direction, gui_scale, gui_solver, gui_alpha, gui_steps]:
        control.on_update(on_reset)

    reset()
    print(f"Open http://localhost:{args.port} in your browser. Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
