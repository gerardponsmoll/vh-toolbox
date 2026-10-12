"""Convert the official SMPL model files into the .npz format used by this toolbox.

The official .pkl files were pickled with Python 2 and contain chumpy and scipy
objects. chumpy no longer installs on current Python, so instead of importing it
we unpickle those objects into placeholders and pull the arrays out of them.

Usage:
    python scripts/prepare_smpl.py path/to/SMPL_python_v.1.1.0
    python scripts/prepare_smpl.py path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "models"
KEYS = ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"]


class _Placeholder:
    """Stands in for a chumpy or scipy object; keeps whatever state was pickled."""

    def __init__(self, *args, **kwargs):
        self.state = {}

    def __setstate__(self, state):
        self.state = state if isinstance(state, dict) else {"value": state}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        package = module.split(".")[0]
        if package in ("chumpy", "scipy"):
            return type(name, (_Placeholder,), {})
        harmless = (
            module in ("copy_reg", "copyreg")
            and name == "_reconstructor"  # how Python 2 pickled class instances
            or module in ("__builtin__", "builtins")
            and name in ("object", "set", "frozenset", "dict", "list", "tuple")
        )
        if package == "numpy" or harmless:
            return super().find_class(module, name)
        # A pickle can run arbitrary code when loaded, so refuse anything else.
        raise pickle.UnpicklingError(f"unexpected object {module}.{name}")


def to_array(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if not isinstance(obj, _Placeholder):
        return np.asarray(obj)

    state = getattr(obj, "state", None) or vars(obj)
    if "indptr" in state:  # scipy.sparse.csc_matrix
        shape = state.get("_shape", state.get("shape"))
        dense = np.zeros(shape)
        for col in range(shape[1]):
            rows = slice(state["indptr"][col], state["indptr"][col + 1])
            dense[state["indices"][rows], col] = state["data"][rows]
        return dense
    if "x" in state:  # chumpy.Ch keeps its value in .x
        return to_array(state["x"])
    arrays = [v for v in state.values() if isinstance(v, np.ndarray)]
    if len(arrays) == 1:
        return arrays[0]
    raise ValueError(f"Cannot extract an array from {type(obj).__name__} with fields {list(state)}")


def load_raw(path: Path) -> dict:
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    with open(path, "rb") as f:
        return _Unpickler(f, encoding="latin1").load()


def gender_of(path: Path) -> str | None:
    name = path.stem.lower()
    for gender, pattern in [
        ("neutral", r"neutral"),
        ("female", r"female|(^|_)f(_|$)"),
        ("male", r"male|(^|_)m(_|$)"),
    ]:
        if re.search(pattern, name):
            return gender
    return None


def convert(path: Path, out_dir: Path = OUT_DIR) -> Path | None:
    gender = gender_of(path)
    if gender is None:
        return None
    raw = load_raw(path)
    if not all(k in raw for k in KEYS):
        return None

    kintree = to_array(raw["kintree_table"]).astype(np.int64)
    parents = kintree[0].copy()
    parents[0] = -1  # the file stores 2^32 - 1 for "no parent"

    v_template = to_array(raw["v_template"])
    posedirs = to_array(raw["posedirs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"smpl_{gender}.npz"
    np.savez_compressed(
        out,
        v_template=v_template.astype(np.float32),
        shapedirs=to_array(raw["shapedirs"]).astype(np.float32),
        posedirs=posedirs.reshape(len(v_template), 3, -1).astype(np.float32),
        J_regressor=to_array(raw["J_regressor"]).astype(np.float32),
        weights=to_array(raw["weights"]).astype(np.float32),
        parents=parents,
        faces=to_array(raw["f"]).astype(np.int32),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="SMPL .pkl/.npz files, or folders containing them")
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in (".pkl", ".npz"))
        elif path.exists():
            files.append(path)
        else:
            print(f"Not found: {path}")

    written = []
    for file in files:
        try:
            out = convert(file)
        except Exception as error:  # keep going: a folder may hold unrelated pickles
            print(f"Skipped {file.name}: {error}")
            continue
        if out is not None:
            print(f"{file.name} -> {out.relative_to(OUT_DIR.parent)}")
            written.append(out)

    if not written:
        print("No SMPL model files found. Expected names like basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
