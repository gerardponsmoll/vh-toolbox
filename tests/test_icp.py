import numpy as np

from vh.icp import closest_points, icp_step, procrustes, transform
from vh.rotations import exp_map

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3)) * [0.3, 0.9, 0.15]  # an elongated, asymmetric cloud
X[:40, 0] += 0.6  # an "arm" on one side
R_TRUE = exp_map([0.2, -0.4, 0.3])
S_TRUE, T_TRUE = 1.3, np.array([0.5, -0.2, 1.0])


def test_procrustes_recovers_a_similarity_transform():
    s, R, t = procrustes(X, transform(X, S_TRUE, R_TRUE, T_TRUE))
    assert np.isclose(s, S_TRUE) and np.allclose(R, R_TRUE) and np.allclose(t, T_TRUE)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_procrustes_without_scale_and_never_a_mirror():
    mirror = X * [-1.0, 1.0, 1.0]
    s, R, _ = procrustes(X, mirror, scale=False)
    assert s == 1.0 and np.isclose(np.linalg.det(R), 1.0)


def test_closest_points_matches_brute_force():
    Q, P = rng.normal(size=(50, 3)), rng.normal(size=(200, 3))
    index, sq = closest_points(Q, P, chunk=16)
    full = ((Q[:, None] - P[None]) ** 2).sum(-1)
    assert np.array_equal(index, full.argmin(1)) and np.allclose(sq, full.min(1))


def test_icp_converges_from_a_small_rotation():
    Y = transform(X, S_TRUE, R_TRUE, T_TRUE)
    f = (1.0, np.eye(3), Y.mean(0) - X.mean(0))
    errors = []
    for _ in range(60):
        f, rms, _ = icp_step(X, Y, *f)
        errors.append(rms)
    assert errors[-1] < 1e-6 and np.allclose(f[1], R_TRUE, atol=1e-5)
    assert all(b <= a + 1e-7 for a, b in zip(errors, errors[1:]))  # the error never increases


def test_procrustes_and_icp_in_2d():
    theta = 0.5
    R2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    P = X[:, :2]
    Q = transform(P, 0.7, R2, np.array([1.0, -2.0]))
    s, R, t = procrustes(P, Q)
    assert np.isclose(s, 0.7) and np.allclose(R, R2) and np.allclose(t, [1.0, -2.0])


def test_the_slide_curves_align_from_40_degrees_and_shrink_with_source_to_target_pairs():
    import json
    from pathlib import Path

    curves = json.loads((Path(__file__).resolve().parent.parent / "demos" / "data" / "icp_curves.json").read_text())
    P, Q = np.array(curves["X"]), np.array(curves["Y"])

    def run(theta, target_to_source):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        f = (1.0, R, Q.mean(0) - R @ P.mean(0))
        for _ in range(150):
            f, rms, _ = icp_step(P, Q, *f, target_to_source=target_to_source)
        return f, rms

    f, rms = run(np.radians(40), True)
    assert rms < 0.03 and np.isclose(f[0], 0.59, atol=0.01)  # aligned
    f, _ = run(np.radians(150), False)
    assert f[0] < 0.5  # every source point finds a closer target point when the source gets smaller


def test_gradient_steps_reach_the_procrustes_solution():
    from vh.icp import gradient_steps

    Y = transform(X, S_TRUE, R_TRUE, T_TRUE)
    pairs = np.stack([np.arange(len(X))] * 2, axis=1)  # correct pairs: only the solve step is tested
    f = gradient_steps(X, Y, pairs, 1.0, np.eye(3), Y.mean(0) - X.mean(0), alpha=0.2, steps=500)
    assert np.isclose(f[0], S_TRUE, atol=1e-4) and np.allclose(f[1], R_TRUE, atol=1e-4) and np.allclose(f[2], T_TRUE, atol=1e-3)
    P, Q = X[:, :2], transform(X[:, :2], 0.8, np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([1.0, 2.0]))
    f = gradient_steps(P, Q, pairs, 1.0, np.eye(2), np.zeros(2), alpha=0.2, steps=800)
    assert np.isclose(f[0], 0.8, atol=1e-4) and np.allclose(f[1], [[0.0, -1.0], [1.0, 0.0]], atol=1e-4)
