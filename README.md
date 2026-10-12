# Virtual Humans Toolbox

**Author:** Gerard Pons-Moll, University of Tübingen

Interactive demos for the Virtual Humans lecture. Each demo opens a 3D viewer in
your browser with controls for the quantities discussed in the lecture, and the
code behind it is short enough to read and modify.

| Lecture | Demo | What you can explore |
| --- | --- | --- |
| 1 | `demos/01_smpl_viewer.py` | SMPL body shape and pose, and each step of the model |

## Setup

You need Python 3.10 or newer. Everything is installed into a virtual
environment inside this folder, so nothing else on your machine is touched.

```bash
git clone <repository-url>
cd vh-toolbox

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

If you use [uv](https://docs.astral.sh/uv/), `uv venv && uv pip install -e .` does the same.

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
  `pelvis` is the global orientation of the body.
- **Display › Show**: stop the model after each step, from the template to the final mesh.
- **Display › Pose blendshapes**: turn the pose-dependent corrections off.
- **Display › Blend skinning**: turn off to attach each vertex to a single joint
  (weight 1, all others 0), so every body part rotates as a rigid piece.
- **Display › Colour**: skinning weights, or the size of the pose blendshape offsets.
- **Display › Skeleton**: joints and bones; the selected joint is red.

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

## The code

- `vh/smpl.py`: the whole SMPL forward pass in about 100 lines of NumPy.
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
