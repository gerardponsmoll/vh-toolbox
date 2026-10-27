# Virtual Humans Toolbox

**Author:** Gerard Pons-Moll, University of Tübingen

Interactive demos for the Virtual Humans lecture. Each demo opens a 3D viewer in
your browser with controls for the quantities discussed in the lecture, and the
code behind it is short enough to read and modify.

The purpose is to teach the material while you can see the code: every Python demo runs
on a short file in `vh/` that follows the notation of the slides, so you can read how a
method works, change it, and watch what happens.

| Lecture | Demo | What you can explore |
| --- | --- | --- |
| 1 | `demos/01_smpl_viewer.py` | SMPL body shape and pose, and each step of the model |
| 2 | `demos/02_camera.py` | The pinhole camera: extrinsics, intrinsics, perspective and weak perspective |
| 2 | `demos/02_facade.py` | Depth is not distance: a long façade, and panoramas |
| 2 | `demos/02_telephoto.py` | Telephoto compression: two people seen from far away |
| 2 | `demos/02_lens.py` | The thin lens: focus and blur |
| 2.2 | `demos/02_rotations.py` | Ways to write a rotation: order, matrices, Euler angles and gimbal lock, interpolation, axis-angle |
| 2.2 | `demos/02_kinematic_chain.py` | A kinematic chain: product of exponentials, the Jacobian, inverse kinematics |
| 2.2 | `demos/02_smpl_ik.py` | Inverse kinematics on the SMPL body |
| 4.1 | `demos/04_icp_curves.py` | ICP step by step on the curves of the slides, with Procrustes or gradient descent |
| 4.1 | `demos/04_icp.py` | ICP in 3D: aligning a body to a noisy, partial scan with outliers |
| 3.1 | `browser/03_1_surface_representations.html` | Implicit and explicit surfaces, CSG, level sets, meshes, Bézier curves and patches, subdivision |
| 3.2 | `browser/03_2_procrustes.html` | Similarity transforms, the SVD, Procrustes alignment |
| 4.1 | `browser/04_1_icp.html` | Closest points, ICP step by step, local minima, k-d trees, ICP in 3D |
| 4.2 | `browser/04_2_body_model.html` | Exponential map, skinning artefacts, pose and shape blend shapes, a toy SMPL |

`01_smpl_viewer` and `02_smpl_ik` need the SMPL model (see below). `02_camera` and
`02_telephoto` use it when it is there and show simpler figures otherwise. The other
demos run without it.

The `browser/` demos need no installation at all: see [Browser demos](#browser-demos).

## Setup

You need Python 3.10 or newer. Everything is installed into a virtual
environment inside this folder, so nothing else on your machine is touched.

```bash
git clone https://github.com/gerardponsmoll/vh-toolbox.git
cd vh-toolbox

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

If you use [uv](https://docs.astral.sh/uv/), `uv venv && uv pip install -e .` does the same.

To open a demo in your browser:

```bash
python demo.py                  # list the demos
python demo.py camera           # open one, by name or number
python demo.py camera --fresh   # start it again from the beginning
python demo.py --stop           # stop all demos
```

Each demo keeps running on its own port until you stop it, so opening it again is instant.
You can also run a script directly, e.g. `python demos/02_camera.py`, and open
http://localhost:8080.

## Get the SMPL model

The SMPL model files are licensed and are not part of this repository.

1. Register at <https://smpl.is.tue.mpg.de> and accept the licence.
2. Under *Downloads*, get "version 1.1.0 for Python 2.7" and unzip it.
3. Convert the files:

```bash
python scripts/prepare_smpl.py path/to/SMPL_python_v.1.1.0
```

This writes `models/smpl_neutral.npz`, `smpl_female.npz` and `smpl_male.npz`.
The `models/` folder is ignored by git: do not share these files.

## Lecture 1: SMPL viewer

```bash
python demos/01_smpl_viewer.py
```

Then open <http://localhost:8080>. Drag to rotate the view, scroll to zoom,
right-drag to pan.

- **Shape (beta)**: the first 10 shape coefficients, in standard deviations.
- **Pose (theta)**: pick a joint, then set its rotation as an axis-angle vector.
  `pelvis` is the global orientation of the body. You can also click a body part
  to select its joint and drag the coloured rings to rotate it; the sliders follow.
- **Display › Show**: stop the model after each step, from the template to the final mesh.
- **Display › Pose blendshapes**: turn the pose-dependent corrections off.
- **Display › Blend skinning**: turn off to attach each vertex to a single joint
  (weight 1, all others 0), so every body part rotates as a rigid piece.
- **Display › Colour**: skinning weights, or the size of the pose blendshape offsets.
- **Display › Skeleton**: joints and bones, with the body made see-through.
  Click a joint to select it; the selected joint is red.

### Things to try

1. Which shape coefficient changes height the most? Which changes weight?
   Do any of them change only one of the two?
2. Turn on the skeleton and move `beta 0`. Why do the joints move?
3. Bend `left_elbow` by about 2 radians around y, then switch *Pose blendshapes*
   off and on. What does linear blend skinning get wrong, and where?
4. Bend `left_knee` and switch *Blend skinning* off. What happens at the joint,
   and why does blending the transforms of two joints fix it?
5. Set *Colour* to "Pose blendshape offset" and bend different joints.
   Which vertices does each joint affect?
6. Set *Colour* to "Skinning weights: selected joint" and step through the
   joints. Which vertices are influenced by more than one joint?
7. Rotate `pelvis`. Does the pose blendshape offset change? Find the line in
   `vh/smpl.py` that explains your answer.

## Lecture 2: the pinhole camera

```bash
python demos/02_camera.py
python demos/02_camera.py --mesh path/to/your_mesh.obj     # any mesh, y axis up
```

A camera looks at an object. The 3D view shows the camera centre, its image
plane (drawn at distance f from the centre) and the object. The panel shows the
image this camera takes. That image is not drawn by the viewer: every point of
the object goes through `project()` in `vh/camera.py`, so what you see is the
formula from the slides. This demo also runs without the SMPL files (the scene
is then shown without the person).

- **Object**: a scene with depth (two rows of columns leading to a house, trees, and
  the SMPL body in the middle), the SMPL body alone, a cube, a checkerboard, or your own mesh.
- **Extrinsics**: the sliders orbit the camera around the object, always looking at its centre.
  You can also grab the camera in the 3D view: drag the arrows to move it and the
  rings to rotate it. The image and the numbers follow while you drag.
- **Intrinsics**: focal length and principal point in pixels, and radial lens distortion.
- **Projection**: perspective, weak perspective, or both on top of each other.
- **Dolly zoom**: ties focal length to distance, so the object keeps its size in the image.
- **A point and its ray**: click the object to pick a point (red). Its ray, its
  coordinates in the world and in the camera, and its pixel are shown. The
  slider moves a second point (blue) along the same ray.
- **Look through the camera**: moves the 3D view to the camera, to compare with the image.

### Things to try

1. Increase the focal length. What happens to the image plane in the 3D view,
   and why does the object get larger in the image?
2. Press *Play* under *Dolly zoom* and watch the person, the columns and the
   house. The person keeps their size: what changes? Can you tell from one image how far away the
   camera was?
3. The two rows of columns are parallel in 3D: are they parallel in the image?
   Where do they meet? Then select the cube and look at its edges. Move the camera far away with *Keep the object's size* on. What happens
   to them?
4. Set *Projection* to "Both" and move the camera close to the body, with the
   arm reaching towards it. Where is weak perspective wrong, and why there?
   How far away must the camera be for the difference to drop below one pixel?
5. Click a point and move the blue point along its ray. Does its pixel change?
   Now switch to weak perspective and try again.
6. Move the principal point. How does the image change? Compare with rotating the camera slightly.
7. Select the checkerboard and change the lens distortion. Which lines stay straight?
8. In `vh/camera.py`, which single line makes the projection nonlinear?

### Depth is not distance: the long façade

```bash
python demos/02_facade.py
```

Should the far ends of a long façade be drawn smaller than its centre? A camera
stands in front of a row of identical columns. The 3D view shows, for the red
column and the middle one, the distance to the camera (orange) and the depth Z
along the optical axis (blue), with the plane of equal depth. The table lists
both for every column, and the height of each column in the image.

1. The columns at the edge of the image are about twice as far away as the middle one. Why are they just as tall?
2. Turn the camera. Which columns shrink now, and what happened to their depth? Where do the
   lines through the column tops and feet meet (the blue cross), and how is that point K R d?
3. Switch the image surface to a cylinder, as in a panorama stitched from photos.
   Which rule gives the column heights now? Why do the straight beam and floor lines bend?
4. Make f small and look at the outer columns on the flat image plane. Why are they
   wider than the middle one, although they are as tall?

### Telephoto compression: two people in one plane

```bash
python demos/02_telephoto.py
```

Two people stand one behind the other. *Play* moves the camera back to 100 m
while zooming in, so the front person keeps their size. The bars on the right of
the image show both heights.

1. Write the ratio of the two image heights in terms of their depths. What does it go to far away?
2. What focal length in millimetres does a full-frame camera need to take the last picture?
3. Why do the trees and the house grow while the people stay the same size?
4. At which distance is the difference between the two people below 5 %? Compare with
   weak perspective in `demos/02_camera.py`.

### The thin lens: focus and blur

```bash
python demos/02_lens.py
```

A ball stands in front of a tree, as in a portrait photograph. From the marked point
on each object, rays go through the top, centre and bottom of the lens and meet again
at the distance z' given by Gauss's formula 1/z + 1/z' = 1/f. At the start the ball
is in focus: its rays meet on the sensor. The tree's rays meet in front of the
sensor and have spread out again when they reach it, so the tree is blurred. The
lens is drawn much larger than a real one so that the rays can be seen.

1. Move *Sensor distance s* until the tree is sharp. What happens to the ball, and
   where do its rays meet now?
2. Go back to the start and change *Focal length f* instead. Which f makes the tree
   sharp? Check your answer with the lens formula.
3. Do the objects change size in the image when you refocus? Why?
4. The ray through the centre of the lens is not bent. Why does that mean the lens
   projects like a pinhole at distance s?
5. Move the sensor closer than f. Can anything be sharp? Why not?

## Lecture 2.2: rotations and kinematic chains

### Ways to write a rotation

```bash
python demos/02_rotations.py
```

A small airplane, turned in five ways (choose the page at the top). Every page
also lists the same rotation as a matrix, Euler angles, a quaternion and an
axis-angle vector, computed with `vh/rotations.py`.

1. *Order matters*: press *Play*. Both airplanes turn 90 degrees about the same
   three axes. Why do they end up different?
2. *Rotation matrix*: drag the rings. Which column of R belongs to which arrow?
   Tick *Mirror*: what happens to det R, and why is this not a rotation?
3. *Euler angles*: press *Go to the gimbal lock* and move angle 1 and angle 3. Why do
   they now do the same thing? Look at the Euler angles in the readout: are they the ones you set?
4. *Interpolation*: press *Play*. Which row turns at constant speed? What happens
   to the airplane in the matrix row halfway?
5. *Axis-angle*: press *Play* and watch the quaternion. Where is it after one full turn,
   and after two?

### A kinematic chain and inverse kinematics

```bash
python demos/02_kinematic_chain.py
```

An arm with four revolute joints. Its pose is the product of exponentials of the
joint twists (`vh/kinematics.py`). Drag the red target and press *One IK step*, or
tick *Follow the target*.

1. The coloured arrows at the hand are the columns of the Jacobian. Turn one joint a
   little with its slider: does the hand move along that joint's arrow?
2. Tick *Rest pose and its twists*. Why are the twists in the product of exponentials
   the axes at the rest pose, and not the current ones?
3. Put the target out of reach so that the arm stretches out straight. What happens
   to the smallest singular value of J, and to the IK steps with damping 0?

### Inverse kinematics on SMPL

```bash
python demos/02_smpl_ik.py
```

Drag the red target and the hand of the SMPL body follows. Every joint of the chain
is a ball joint with three columns in the Jacobian. The result is stored the way SMPL
stores poses: the rotation of each joint relative to its parent, as axis-angle.

1. Change *Joints in the chain*. How does the motion change with 1, 3 and 7 joints?
   (The pelvis is never in the chain: turning it would turn the whole body.)
2. Pull the target through the body or bend the elbow backwards. What would you add
   to stop poses that a person cannot make?

## Lecture 4.1: Iterative Closest Point

```bash
python demos/04_icp_curves.py
python demos/04_icp.py
```

Both demos run on `vh/icp.py`: Procrustes, closest points, one ICP step, and gradient
steps as an alternative to Procrustes, in about 100 lines. The browser page
`browser/04_1_icp.html` shows the same ideas and more; these demos are the version whose
code you can change.

1. Curves: press *Find correspondences*, *Solve Procrustes* and *Apply the transform* in
   turn, then *Run*. Set the initial rotation to 0 degrees: why does ICP stop in the wrong place?
2. Switch the pairs to *Source -> target* with scale on. Why does the source shrink?
3. Solve with *Gradient descent*. Find a step size that keeps up with Procrustes, one that
   is too slow, and one that diverges.
4. 3D: add outliers, run, then trim 15% of the pairs. What changes, and why?
5. In `vh/icp.py`, change how the pairs are chosen or weighted, and run the demos again.

## Browser demos

```bash
open browser/03_2_procrustes.html     # or double-click the file
```

Each file in `browser/` is a complete page: download it (or the whole repository),
double-click it, and it opens in your browser. The JavaScript is inside the file, in its
`<script type="module">` block. The pages load [three.js](https://threejs.org) from the
internet, so you need a connection. Every page has several sections, listed at the top,
one per idea from the slides. The body model page uses a procedural toy body, not SMPL data.

## The code

- `vh/smpl.py`: the whole SMPL forward pass in about 100 lines of NumPy.
- `vh/camera.py`: the pinhole camera: `look_at`, `project`, `project_cylinder`, `pixel_ray`.
- `vh/render.py`: draws an image from projected points (nearest point wins).
- `vh/scene.py`: coloured meshes for the camera demos, and taking a picture of them.
- `vh/lens.py`: the thin lens: where a point is sharp, the size of its blur disc, and disc blur.
- `vh/rotations.py`: rotation matrices, Euler angles, quaternions, axis-angle, SLERP.
- `vh/kinematics.py`: twists, the product of exponentials and the articulated Jacobian.
- `vh/icp.py`: Procrustes alignment, closest points, ICP and gradient-based ICP.
- `vh/geometry.py`: geometric helpers (ray-mesh intersection).
- `demos/`: one viewer per lecture.
- `scripts/prepare_smpl.py`: converts the official model files.
- `tests/`: run with `pip install -e ".[dev]" && pytest`.

```python
import numpy as np
from vh import SMPLModel, forward

model = SMPLModel.load("models/smpl_neutral.npz")
pose = np.zeros((24, 3))
pose[18] = [0.0, -1.5, 0.0]      # bend the left elbow
out = forward(model, betas=np.zeros(10), pose=pose)
print(out.vertices.shape)        # (6890, 3)
```

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`):
you are free to use, modify and share it for research, teaching and anything else,
as long as the copyright notice is kept.

The SMPL model is not part of this repository and is not covered by that licence.
It has its own licence, which you accept when you download it from
<https://smpl.is.tue.mpg.de>.
