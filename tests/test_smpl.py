import numpy as np

from vh import forward, rigid_weights, rodrigues


def test_rodrigues_gives_rotations():
    R = rodrigues(np.random.default_rng(0).normal(size=(50, 3)))
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)


def test_rodrigues_known_values():
    assert np.allclose(rodrigues(np.zeros(3))[0], np.eye(3))
    quarter_turn_z = rodrigues(np.array([0.0, 0.0, np.pi / 2]))[0]
    assert np.allclose(quarter_turn_z @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_zero_parameters_give_template(model):
    out = forward(model)
    assert np.allclose(out.vertices, model.v_template)
    assert np.allclose(out.joints, model.J_regressor @ model.v_template)


def test_zero_pose_gives_shaped_body(model):
    betas = np.random.default_rng(1).normal(size=model.num_betas)
    out = forward(model, betas)
    assert not np.allclose(out.v_shaped, model.v_template)
    assert np.allclose(out.vertices, out.v_shaped)
    assert np.allclose(out.joints, out.joints_rest)


def test_root_rotation_is_rigid(model):
    """Global orientation rotates the body about the pelvis without deforming it."""
    rng = np.random.default_rng(2)
    betas = rng.normal(size=model.num_betas)
    pose = rng.normal(size=(model.num_joints, 3)) * 0.3
    pose[0] = 0.0
    reference = forward(model, betas, pose)

    pose[0] = [0.3, -1.1, 0.5]
    rotated = forward(model, betas, pose)
    R = rodrigues(pose[0])[0]
    pelvis = reference.joints_rest[0]
    assert np.allclose(rotated.vertices, (reference.vertices - pelvis) @ R.T + pelvis)
    assert np.allclose(rotated.joints, (reference.joints - pelvis) @ R.T + pelvis)


def test_joints_follow_forward_kinematics(model):
    """Bone lengths do not change with pose."""
    pose = np.random.default_rng(3).normal(size=(model.num_joints, 3))
    out = forward(model, pose=pose)
    child = np.arange(1, model.num_joints)
    parent = model.parents[1:]
    posed = np.linalg.norm(out.joints[child] - out.joints[parent], axis=1)
    rest = np.linalg.norm(out.joints_rest[child] - out.joints_rest[parent], axis=1)
    assert np.allclose(posed, rest)


def test_pose_blendshapes_can_be_disabled(model):
    pose = np.random.default_rng(4).normal(size=(model.num_joints, 3)) * 0.5
    on = forward(model, pose=pose)
    off = forward(model, pose=pose, pose_blendshapes=False)
    assert np.allclose(off.v_posed, off.v_shaped)
    assert not np.allclose(on.vertices, off.vertices)
    assert np.allclose(on.joints, off.joints)


def test_translation(model):
    trans = np.array([1.0, 2.0, 3.0])
    assert np.allclose(forward(model, trans=trans).vertices, model.v_template + trans)


def test_rigid_skinning_moves_each_part_rigidly(model):
    """Without blending, a vertex follows exactly one joint transform."""
    pose = np.random.default_rng(5).normal(size=(model.num_joints, 3)) * 0.5
    out = forward(model, pose=pose, pose_blendshapes=False, blend_skinning=False)
    blended = forward(model, pose=pose, pose_blendshapes=False)
    assert not np.allclose(out.vertices, blended.vertices)
    assert np.allclose(out.joints, blended.joints)

    owner = model.weights.argmax(axis=1)
    assert np.allclose(rigid_weights(model.weights).sum(axis=1), 1.0)
    for k in np.unique(owner):
        part = owner == k
        R, t = out.transforms[k, :3, :3], out.transforms[k, :3, 3]
        expected = (out.v_shaped[part] - out.joints_rest[k]) @ R.T + t
        assert np.allclose(out.vertices[part], expected)
