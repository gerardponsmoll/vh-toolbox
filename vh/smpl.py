"""SMPL in plain NumPy, written to be read alongside the lecture slides.

The model is a function M(beta, theta) -> vertices, built in four steps:

    1. shape blendshapes   T_shaped = T_bar + B_S(beta)
    2. joint regression    J        = J_regressor @ T_shaped
    3. pose blendshapes    T_posed  = T_shaped + B_P(theta)
    4. skinning            v        = LBS(T_posed, J, theta, W)

Reference: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", 2015.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


@dataclass
class SMPLModel:
    """The learned parameters of SMPL. V vertices, K joints, B shape components."""

    v_template: np.ndarray  # (V, 3)      mean body in the rest (T) pose
    shapedirs: np.ndarray  # (V, 3, B)    shape blendshapes
    posedirs: np.ndarray  # (V, 3, 9(K-1)) pose blendshapes
    J_regressor: np.ndarray  # (K, V)     joints as a linear function of vertices
    weights: np.ndarray  # (V, K)         skinning weights, rows sum to 1
    parents: np.ndarray  # (K,)           parent of each joint, -1 for the root
    faces: np.ndarray  # (F, 3)           triangles

    @property
    def num_joints(self) -> int:
        return len(self.parents)

    @property
    def num_betas(self) -> int:
        return self.shapedirs.shape[2]

    @classmethod
    def load(cls, path: str | Path) -> "SMPLModel":
        """Load a model converted by scripts/prepare_smpl.py."""
        data = np.load(path)
        return cls(
            v_template=data["v_template"].astype(np.float64),
            shapedirs=data["shapedirs"].astype(np.float64),
            posedirs=data["posedirs"].astype(np.float64),
            J_regressor=data["J_regressor"].astype(np.float64),
            weights=data["weights"].astype(np.float64),
            parents=data["parents"].astype(np.int64),
            faces=data["faces"].astype(np.int64),
        )


@dataclass
class SMPLOutput:
    """Every intermediate result of the forward pass, so each step can be shown."""

    v_shaped: np.ndarray  # (V, 3)    after shape blendshapes, rest pose
    v_posed: np.ndarray  # (V, 3)     after pose blendshapes, still rest pose
    vertices: np.ndarray  # (V, 3)    after skinning
    joints_rest: np.ndarray  # (K, 3) joints of the shaped body in the rest pose
    joints: np.ndarray  # (K, 3)      joints after posing
    transforms: np.ndarray  # (K, 4, 4) world transform of each joint


def rodrigues(r: np.ndarray) -> np.ndarray:
    """Axis-angle vectors (N, 3) -> rotation matrices (N, 3, 3).

    The direction of r is the rotation axis and its length is the angle:
    R = I + sin(a) K + (1 - cos(a)) K^2, with K the cross-product matrix of the axis.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
    angle = np.linalg.norm(r, axis=1, keepdims=True)
    axis = r / np.maximum(angle, 1e-12)

    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

    sin = np.sin(angle)[:, :, None]
    cos = np.cos(angle)[:, :, None]
    return np.eye(3) + sin * K + (1.0 - cos) * (K @ K)


def rigid_weights(weights: np.ndarray) -> np.ndarray:
    """Skinning weights (V, K) with all the weight moved to each vertex's main joint."""
    rigid = np.zeros_like(weights)
    rigid[np.arange(len(weights)), weights.argmax(axis=1)] = 1.0
    return rigid


def joint_transforms(R: np.ndarray, joints_rest: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Forward kinematics: the world transform (K, 4, 4) of every joint.

    Each local transform rotates by R[k] about the joint, expressed as a
    rotation plus the offset of the joint from its parent, and is chained
    down the tree: G[k] = G[parent] @ local[k].
    """
    G = np.zeros((len(parents), 4, 4))
    for k, p in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = R[k]
        local[:3, 3] = joints_rest[k] if p < 0 else joints_rest[k] - joints_rest[p]
        G[k] = local if p < 0 else G[p] @ local
    return G


def forward(
    model: SMPLModel,
    betas: np.ndarray | None = None,
    pose: np.ndarray | None = None,
    trans: np.ndarray | None = None,
    pose_blendshapes: bool = True,
    blend_skinning: bool = True,
) -> SMPLOutput:
    """Evaluate SMPL.

    betas: (B,) shape coefficients, in standard deviations of the training population.
    pose:  (K, 3) or (3K,) axis-angle rotation of each joint relative to its parent.
           Joint 0 is the global orientation of the body.
    trans: (3,) global translation.
    pose_blendshapes: set to False to see what skinning alone produces.
    blend_skinning: set to False to attach each vertex rigidly to a single joint.
    """
    K = model.num_joints
    betas = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
    pose = np.zeros((K, 3)) if pose is None else np.asarray(pose, dtype=np.float64).reshape(K, 3)
    trans = np.zeros(3) if trans is None else np.asarray(trans, dtype=np.float64)

    # 1. Shape blendshapes: a linear offset from the mean body.
    v_shaped = model.v_template + model.shapedirs[:, :, : len(betas)] @ betas

    # 2. Joint locations depend on shape: a tall person has joints further apart.
    joints_rest = model.J_regressor @ v_shaped

    # 3. Pose blendshapes: corrective offsets, linear in the elements of the
    #    rotation matrices (minus identity, so the rest pose gives zero offset).
    #    The root is excluded: global orientation must not deform the body.
    R = rodrigues(pose)
    if pose_blendshapes:
        pose_feature = (R[1:] - np.eye(3)).ravel()
        v_posed = v_shaped + model.posedirs @ pose_feature
    else:
        v_posed = v_shaped

    # 4a. Forward kinematics: chain the local transforms down the tree.
    G = joint_transforms(R, joints_rest, model.parents)
    joints = G[:, :3, 3]

    # 4b. The vertices are given in the rest pose, so first undo the rest pose:
    #     A[k] maps a rest-pose point attached to joint k to its posed position.
    A = G.copy()
    A[:, :3, 3] -= np.einsum("kij,kj->ki", G[:, :3, :3], joints_rest)

    # 4c. Linear blend skinning: each vertex is moved by a weighted average of
    #     the joint transforms. With rigid weights the average has a single
    #     term, so each body part moves as a rigid piece.
    weights = model.weights if blend_skinning else rigid_weights(model.weights)
    T = np.einsum("vk,kij->vij", weights, A)
    vertices = np.einsum("vij,vj->vi", T[:, :3, :3], v_posed) + T[:, :3, 3]

    return SMPLOutput(
        v_shaped=v_shaped,
        v_posed=v_posed,
        vertices=vertices + trans,
        joints_rest=joints_rest,
        joints=joints + trans,
        transforms=G,
    )
