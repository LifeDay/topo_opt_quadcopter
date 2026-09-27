"""Stage 1: build123d cantilever → STEP → gmsh (fragment, sets, C3D10) → .inp → ccx.

Pass: region volumes match the CAD, BC sets are non-empty and on the right faces, and a
solid-beam static run completes.
"""

import sys

import numpy as np

from topo_opt_quadcopter.beso_runner import REPO_ROOT
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.geometry import Cantilever, expected_region_volumes, export_named_step
from topo_opt_quadcopter.inp_writer import PETG, model_data, static_step, write_inp
from topo_opt_quadcopter.mesh import BoxSelect, mesh_step

OUT = REPO_ROOT / "runs" / "stage1"
TIP_LOAD = -10.0  # N in z


def build(name: str, beam: Cantilever, size: float):
    solids = beam.solids()
    step = export_named_step(solids, OUT / f"{name}.step")
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, beam.length))}
    mesh = mesh_step(step, bcs, size=size, save_msh=OUT / f"{name}.msh")
    return solids, mesh


def check(name: str, beam: Cantilever, solids, mesh, vol_tol: float) -> bool:
    ok = True
    expected = expected_region_volumes(solids)
    print(f"\n[{name}] {len(mesh.node_ids)} nodes, {len(mesh.connectivity)} {mesh.elem_type}")
    print(f"  {'region':<14}{'elements':>9}{'CAD':>11}{'gmsh CAD':>11}{'mesh':>11}{'err':>9}")
    for region, vol in expected.items():
        mesh_vol = mesh.element_volumes(region).sum()
        err = mesh_vol / vol - 1
        ok &= abs(err) < vol_tol and abs(mesh.cad_volumes[region] / vol - 1) < 1e-6
        print(f"  {region:<14}{len(mesh.elsets[region]):>9}{vol:>11.1f}"
              f"{mesh.cad_volumes[region]:>11.1f}{mesh_vol:>11.1f}{err:>9.2e}")

    index = {n: i for i, n in enumerate(mesh.node_ids)}
    for set_name, x in [("FIXED", 0.0), ("LOAD", beam.length)]:
        xs = mesh.coords[[index[n] for n in mesh.nsets[set_name]], 0]
        area = mesh.surface_area(set_name)
        on_face = np.allclose(xs, x)
        area_ok = abs(area / beam.area - 1) < 1e-9
        ok &= on_face and area_ok
        print(f"  {set_name}: {len(xs)} nodes, {len(mesh.surfaces[set_name])} faces, "
              f"all at x={x}: {on_face}, face area {area:.3f} (expected {beam.area:.3f})")
    return ok


def solve(name: str, mesh) -> bool:
    model = model_data(mesh, PETG, couplings=["LOAD"])
    inp = write_inp(OUT / f"{name}.inp", model,
                    [static_step(["FIXED"], loads=[("REF_LOAD", 3, TIP_LOAD)], print_nsets=["LOAD"])])
    run = run_ccx(inp)
    uz = run.dat.get("displacements", "LOAD")[:, 3].mean()
    rf = run.dat.get("total force", "FIXED")[0]
    print(f"  ccx: {run.wall_s:.1f} s, mean tip uz = {uz:.5f} mm, root reaction = {rf}")
    return abs(rf[2] + TIP_LOAD) < 1e-6 * abs(TIP_LOAD)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    plain = Cantilever()
    solids, mesh = build("plain", plain, size=2.5)
    ok &= check("plain", plain, solids, mesh, vol_tol=1e-9)
    ok &= solve("plain", mesh)

    holed = Cantilever(hole_diameter=8.0, boss_diameter=14.0)
    solids, mesh = build("holed", holed, size=2.5)
    # C3D10 corners sit on the curved hole; the straight-sided volume is slightly under.
    ok &= check("holed", holed, solids, mesh, vol_tol=5e-3)
    ok &= solve("holed", mesh)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
