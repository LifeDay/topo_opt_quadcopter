"""Binary STL export for the remote viewer (viewer/viewer.html polls viewer/latest.stl).

Writes are atomic: the file is written to <name>.tmp and renamed over the target, so the
viewer never loads a half-written file.

    uv run python -m topo_opt_quadcopter.stl_export runs/.../file057.vtk
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyvista as pv

from .beso_runner import REPO_ROOT

VIEWER_DIR = REPO_ROOT / "viewer"

_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def write_binary_stl(path: Path, points: np.ndarray, triangles: np.ndarray) -> None:
    tri = np.asarray(points, dtype=np.float64)[np.asarray(triangles)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(tri), dtype=_RECORD)
    records["normal"], records["vertices"] = normals, tri
    with open(path, "wb") as fh:
        fh.write(b"topo_opt_quadcopter".ljust(80, b"\0"))
        fh.write(np.uint32(len(tri)).tobytes())
        fh.write(records.tobytes())


def surface_triangles(data: pv.DataSet) -> tuple[np.ndarray, np.ndarray]:
    """Outer surface of any pyvista mesh (volume or shell) as (points, triangles)."""
    surf = data.extract_surface(algorithm="dataset_surface").triangulate().clean()
    return np.asarray(surf.points), surf.faces.reshape(-1, 4)[:, 1:]


def beso_solid(vtk_path: Path) -> pv.UnstructuredGrid:
    """Elements BESO kept (element_states == 1) from one of its per-iteration .vtk files."""
    return pv.read(vtk_path).threshold(0.5, scalars="element_states")


def export_stl(data: pv.DataSet | tuple[np.ndarray, np.ndarray], path: Path = VIEWER_DIR / "latest.stl") -> Path:
    """Atomically write data (a pyvista mesh, or (points, triangles)) as binary STL."""
    points, triangles = surface_triangles(data) if isinstance(data, pv.DataSet) else data
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    write_binary_stl(tmp, points, triangles)
    os.replace(tmp, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh", type=Path, help="BESO .vtk (solid elements only) or any pyvista-readable mesh")
    parser.add_argument("-o", "--out", type=Path, default=VIEWER_DIR / "latest.stl")
    args = parser.parse_args()
    is_beso = args.mesh.suffix == ".vtk" and "element_states" in pv.read(args.mesh).cell_data
    data = beso_solid(args.mesh) if is_beso else pv.read(args.mesh)
    print(export_stl(data, args.out))


if __name__ == "__main__":
    main()
