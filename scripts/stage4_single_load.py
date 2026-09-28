"""Stage 4: single-load BESO (stiffness) on a cantilever with a keep-out hole and keep-in boss.

Cases vary the mesh size (fixed filter radius in mm), the filter radius, and the ccx thread
count (to check whether run-to-run differences come from the solver's threading).

Checks per case: converges to the mass goal; keep-in and pad elements stay solid; no
elements inside the hole; the solid is one connected piece; tip displacement vs the solid
block; BESO's own time vs ccx's. Across meshes: overlap (IoU) of the final designs,
sampled on a 1 mm grid over the design region.

    uv run python scripts/stage4_single_load.py [case ...] [--live]
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from topo_opt_quadcopter.beso_runner import REPO_ROOT, Domain, domains_conf, run_beso
from topo_opt_quadcopter.geometry import Cantilever, export_named_step
from topo_opt_quadcopter.inp_writer import PETG, model_data, static_step, write_inp
from topo_opt_quadcopter.mesh import BoxSelect, mesh_step
from topo_opt_quadcopter.stl_export import VIEWER_DIR, beso_solid

OUT = REPO_ROOT / "runs" / "stage4"
BEAM = Cantilever(length=120.0, width=20.0, height=60.0, hole_diameter=10.0, boss_diameter=20.0)
TIP_LOAD = -100.0  # N in z; the stiffness optimum does not depend on the magnitude
MASS_GOAL = 0.3
BASE = "h2.5_r6"


@dataclass
class Case:
    size: float
    filter_range: float | str  # mm, or "auto" (2 × mean element size)
    threads: int = 8


CASES = {
    "h2.5_r6": Case(2.5, 6.0),
    "h2.5_r6_repeat": Case(2.5, 6.0),
    "h2.5_r6_t1": Case(2.5, 6.0, threads=1),
    "h3.0_r6": Case(3.0, 6.0),
    "h2.0_r6": Case(2.0, 6.0),
    "h2.5_rauto": Case(2.5, "auto"),
    "h2.5_r10": Case(2.5, 10.0),
}

CONF = """
mass_goal_ratio = {mass_goal}
filter_list = [["simple", {filter_range!r}]]
optimization_base = "stiffness"
displacement_graph = [["LOAD", "total"]]
save_iteration_results = 1
save_resulting_format = "vtk"
"""


def build_inp(size: float):
    path = OUT / f"cantilever_h{size}.inp"
    step = export_named_step(BEAM.solids(), OUT / "cantilever.step")
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, BEAM.length))}
    mesh = mesh_step(step, bcs, size=size)
    model = model_data(mesh, PETG, couplings=["LOAD"])
    write_inp(path, model, [static_step(["FIXED"], loads=[("REF_LOAD", 3, TIP_LOAD)])])
    return path, mesh


def beso_table(log_path) -> dict[str, np.ndarray]:
    """Columns of the iteration table in BESO's <inp>.log, keyed by header name."""
    lines = log_path.read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l.lstrip().startswith("i ") and "mass" in l)
    names = lines[head].split()
    rows = []
    for line in lines[head + 1:]:
        parts = line.split()
        if len(parts) != len(names) or not parts[0].isdigit():
            break
        rows.append([float(p) for p in parts])
    return dict(zip(names, np.array(rows).T))


def design_grid(step: float = 1.0) -> np.ndarray:
    """1 mm grid over the design region only (outside the pads and the boss)."""
    b = BEAM
    xs = np.arange(b.pad_length + step / 2, b.length - b.pad_length, step)
    ys = np.arange(-b.width / 2 + step / 2, b.width / 2, step)
    zs = np.arange(-b.height / 2 + step / 2, b.height / 2, step)
    p = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    r = np.hypot(p[:, 0] - b.length / 2, p[:, 2])
    return p[r > b.boss_diameter / 2]


def solid_mask(solid: pv.UnstructuredGrid, points: np.ndarray) -> np.ndarray:
    return solid.linear_copy().find_containing_cell(points) >= 0


def check_regions(vtk_path) -> dict:
    """Region checks on BESO's final state, by element centroid."""
    grid = pv.read(vtk_path)
    states = np.asarray(grid.cell_data["element_states"])
    c = grid.cell_centers().points
    b = BEAM
    r = np.hypot(c[:, 0] - b.length / 2, c[:, 2])
    pads = (c[:, 0] < b.pad_length) | (c[:, 0] > b.length - b.pad_length)
    boss = r < b.boss_diameter / 2
    solid = beso_solid(vtk_path)
    regions = solid.connectivity(extraction_mode="all")
    return {
        "pads_and_boss_solid": bool(states[pads | boss].min() == 1),
        "elements_in_hole": int((r < b.hole_diameter / 2 - 0.5).sum()),
        "pieces": len(np.unique(regions.cell_data["RegionId"])),
        "solid": solid,
    }


def run_case(name: str, case: Case, live: bool) -> dict:
    inp, mesh = build_inp(case.size)
    conf = domains_conf([Domain("design", True), Domain("fixed_root", False),
                         Domain("load_tip", False), Domain("keep_in_boss", False)], PETG)
    conf += CONF.format(mass_goal=MASS_GOAL, filter_range=case.filter_range)
    run = run_beso(OUT / name, inp, conf, threads=case.threads,
                   live_stl=VIEWER_DIR / "latest.stl" if live else None)
    table = beso_table(run.run_dir / inp.with_suffix(".log").name)
    iters = len(table["i"])
    mass = table["mass"] / table["mass"][0]
    disp = table["LOAD(u_total)"]
    vtk = max(run.run_dir.glob("file*.vtk"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    regions = check_regions(vtk)
    return {
        "case": name, "h_mm": case.size, "filter": case.filter_range, "threads": case.threads,
        "nodes": len(mesh.node_ids), "design_elems": len(mesh.elsets["design"]),
        "iterations": iters, "final_mass": round(float(mass[-1]), 4),
        "tip_u0_mm": round(float(disp[0]), 5), "tip_u_mm": round(float(disp[-1]), 5),
        "u_ratio": round(float(disp[-1] / disp[0]), 3),
        "pads_and_boss_solid": regions["pads_and_boss_solid"],
        "elements_in_hole": regions["elements_in_hole"], "pieces": regions["pieces"],
        "wall_s": round(run.wall_s, 1), "ccx_s": round(sum(run.ccx_s), 1),
        "beso_s_per_iter": round(run.overhead_s / iters, 2),
        "ccx_s_per_iter": round(sum(run.ccx_s) / iters, 2),
        "_solid": regions["solid"], "_mass": mass, "_disp": disp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help=f"default: all of {list(CASES)}")
    parser.add_argument("--live", action="store_true", help="export each iteration to viewer/latest.stl")
    args = parser.parse_args()
    unknown = set(args.cases) - set(CASES)
    if unknown:
        parser.error(f"unknown cases {sorted(unknown)}")
    args.cases = args.cases or list(CASES)
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.cases:
        print(f"[{name}] {CASES[name]}", flush=True)
        res = run_case(name, CASES[name], args.live)
        print({k: v for k, v in res.items() if not k.startswith("_")}, flush=True)
        results.append(res)

    grid = design_grid()
    base = next((r for r in results if r["case"] == BASE), None)
    if base is not None:
        base_mask = solid_mask(base["_solid"], grid)
        for r in results:
            mask = solid_mask(r["_solid"], grid)
            r["iou_vs_base"] = round(float((mask & base_mask).sum() / (mask | base_mask).sum()), 3)

    ok = True
    for r in results:
        passed = (abs(r["final_mass"] - MASS_GOAL) < 0.02 and r["pads_and_boss_solid"]
                  and r["elements_in_hole"] == 0 and r["pieces"] == 1)
        r["pass"] = passed
        ok &= passed

    fields = [k for k in results[0] if not k.startswith("_")]
    fields += [k for k in ("iou_vs_base", "pass") if k not in fields]
    csv_path = OUT / "stage4_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        np.savetxt(OUT / r["case"] / "history.csv", np.column_stack([r["_mass"], r["_disp"]]),
                   delimiter=",", header="mass_ratio,tip_u_mm", comments="")
    print(f"\nwrote {csv_path}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
