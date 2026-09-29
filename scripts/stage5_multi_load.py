"""Stage 5: several load cases (Fz, Fy, torsion Mx) on the stage 4 cantilever.

Loads go on the tip coupling (moments through *DISTRIBUTING dofs 1-6). Their magnitudes are
balanced so each gives the solid block the same compliance as Fz = -100 N; otherwise the
weak-axis Fy would dominate BESO's envelope (each element takes its highest energy density
over the steps) just because of the units chosen.

Cases: each load alone; the three as separate steps (the envelope); the envelope with Fy × 5;
and the three applied together in one step, which is a single, different load case.

Each final design is re-solved under each load alone (void elements at 1e-6 E, as in BESO)
and its compliance divided by that of the design optimized for that load alone. Pass: the
usual stage 4 checks per case, and the envelope design has the lowest worst-case ratio.

    uv run python scripts/stage5_multi_load.py [case ...] [--size MM] [--live] [--report-only]
"""

import argparse
import csv
import dataclasses
import json
import math
import re
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from stage4_single_load import BEAM, beso_table, check_regions
from topo_opt_quadcopter.beso_runner import REPO_ROOT, Domain, domains_conf, run_beso
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.geometry import export_named_step
from topo_opt_quadcopter.inp_writer import PETG, model_data, static_step, void_material, write_inp
from topo_opt_quadcopter.mesh import BoxSelect, FEMesh, mesh_step
from topo_opt_quadcopter.stl_export import VIEWER_DIR, beso_solid

FILTER = 6.0
MASS_GOAL = 0.3
# Reference loads on REF_LOAD (dof, value): N, or N·mm for the moment. fz is kept as is;
# fy and mx are rescaled to match its solid-block compliance.
UNIT = {"fz": (3, -100.0), "fy": (2, -100.0), "mx": (4, 1000.0)}
SINGLE = ["fz", "fy", "mx"]
# case → steps; each step maps a load to a multiple of its balanced magnitude
CASES = {
    "fz": [{"fz": 1}],
    "fy": [{"fy": 1}],
    "mx": [{"mx": 1}],
    "env": [{"fz": 1}, {"fy": 1}, {"mx": 1}],
    "env_fy5": [{"fz": 1}, {"fy": 5}, {"mx": 1}],
    "simul": [{"fz": 1, "fy": 1, "mx": 1}],
}

CONF = """
mass_goal_ratio = {mass_goal}
filter_list = [["simple", {filter_range!r}]]
optimization_base = "stiffness"
displacement_graph = [["LOAD", "total"]]
save_iteration_results = 1
save_resulting_format = "vtk"
"""


def build_mesh(size: float, out) -> FEMesh:
    step = export_named_step(BEAM.solids(), out / "cantilever.step")
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, BEAM.length))}
    return mesh_step(step, bcs, size=size)


def load_step(loads: dict[str, float], print_load: bool = False) -> str:
    """One *STATIC step with the given loads (name → value on its REF_LOAD dof)."""
    return static_step(["FIXED"], loads=[("REF_LOAD", UNIT[k][0], v) for k, v in loads.items()],
                       print_nsets=["LOAD"] if print_load else [])


def tip_compliance(dat, step: int, mesh: FEMesh, load: str, value: float) -> float:
    """F·u (or M·θ) at the tip from the LOAD face nodes: mean displacement for a force,
    least-squares twist about x for the moment."""
    u = dat.get("displacements", "LOAD", step)
    if load != "mx":
        return value * float(u[:, UNIT[load][0]].mean())
    xyz = mesh.coords[np.searchsorted(mesh.node_ids, u[:, 0].astype(int))]
    a = np.r_[-xyz[:, 2], xyz[:, 1]]
    theta = float(a @ np.r_[u[:, 2], u[:, 3]] / (a @ a))
    return value * theta


def evaluate(mesh: FEMesh, solid_ids: np.ndarray, magnitudes: dict[str, float], path) -> dict[str, float]:
    """Compliance under each load alone, with the elements not in solid_ids set to void."""
    all_ids = np.concatenate(list(mesh.elsets.values()))
    void_ids = np.setdiff1d(all_ids, solid_ids)
    elsets = {"solid": np.sort(solid_ids)} | ({"void": void_ids} if len(void_ids) else {})
    model = model_data(dataclasses.replace(mesh, elsets=elsets), PETG, couplings=["LOAD"], dofs="1,6",
                       elset_materials={"void": void_material(PETG)})
    write_inp(path, model, [load_step({k: magnitudes[k]}, print_load=True) for k in SINGLE])
    dat = run_ccx(path, threads=8).dat
    return {k: tip_compliance(dat, i, mesh, k, magnitudes[k]) for i, k in enumerate(SINGLE)}


def balance(mesh: FEMesh, out) -> tuple[dict[str, float], dict[str, float]]:
    """Magnitudes giving equal solid-block compliance, and that compliance per load."""
    c = evaluate(mesh, np.concatenate(list(mesh.elsets.values())), {k: v for k, (_, v) in UNIT.items()},
                 out / "solid_unit.inp")
    mags = {k: float(f"{v * math.sqrt(c['fz'] / c[k]):.3g}") for k, (_, v) in UNIT.items()}
    solid = {k: c[k] * (mags[k] / UNIT[k][1]) ** 2 for k in SINGLE}
    return mags, solid


def final_vtk(run_dir):
    return max(run_dir.glob("file*.vtk"), key=lambda p: int(re.sub(r"\D", "", p.stem)))


def solid_element_ids(mesh: FEMesh, vtk_path) -> np.ndarray:
    """Element ids BESO kept, matched to the vtk cells by the mean of all their nodes (not
    the parametric centre, which moves when midside nodes sit on a curved surface). The vtk
    stores float32 points, hence the loose tolerance."""
    grid = pv.read(vtk_path)
    states = np.asarray(grid.cell_data["element_states"])
    cells = grid.cells.reshape(grid.n_cells, -1)[:, 1:]  # all C3D10: [10, n0, ..., n9] per cell
    ids = np.concatenate(list(mesh.elsets.values()))
    corners = range(len(mesh.connectivity[int(ids[0])]))
    centroids = mesh._corner_coords(ids, corners).mean(axis=1)
    dist, idx = cKDTree(grid.points[cells].mean(axis=1)).query(centroids)
    if dist.max() > 1e-3 or len(np.unique(idx)) != len(ids):
        raise ValueError(f"vtk cells do not match the mesh elements (max distance {dist.max():.2g})")
    return ids[states[idx] == 1]


def run_case(name: str, mesh: FEMesh, inp_path, mags: dict[str, float], out, live: bool) -> dict:
    steps = [load_step({k: m * mags[k] for k, m in step.items()}) for step in CASES[name]]
    inp = write_inp(inp_path, model_data(mesh, PETG, couplings=["LOAD"], dofs="1,6"), steps)
    conf = domains_conf([Domain("design", True), Domain("fixed_root", False),
                         Domain("load_tip", False), Domain("keep_in_boss", False)], PETG)
    conf += CONF.format(mass_goal=MASS_GOAL, filter_range=FILTER)
    run = run_beso(out / name, inp, conf, live_stl=VIEWER_DIR / "latest.stl" if live else None)
    table = beso_table(run.run_dir / inp.with_suffix(".log").name)
    iters = len(table["i"])
    mass = table["mass"] / table["mass"][0]
    np.savetxt(run.run_dir / "history.csv", np.column_stack([mass, table["LOAD(u_total)"]]),
               delimiter=",", header="mass_ratio,max_tip_u_mm", comments="")
    summary = {
        "case": name, "steps": len(steps), "iterations": iters, "final_mass": round(float(mass[-1]), 4),
        "wall_s": round(run.wall_s, 1), "ccx_s_per_iter": round(sum(run.ccx_s) / iters, 2),
        "beso_s_per_iter": round(run.overhead_s / iters, 2),
    }
    (run.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def report(mesh: FEMesh, mags: dict[str, float], solid_c: dict[str, float], out) -> bool:
    """Cross-evaluate every finished case under each load; write CSV and a render."""
    done = [c for c in CASES if (out / c / "summary.json").exists()]
    rows = []
    for name in done:
        row = json.loads((out / name / "summary.json").read_text())
        vtk = final_vtk(out / name)
        regions = check_regions(vtk)
        row |= {"pads_and_boss_solid": regions["pads_and_boss_solid"],
                "elements_in_hole": regions["elements_in_hole"], "pieces": regions["pieces"]}
        comp = evaluate(mesh, solid_element_ids(mesh, vtk), mags, out / name / "eval.inp")
        row |= {f"C_{k}": round(comp[k], 4) for k in SINGLE}
        row |= {f"C_{k}/solid": round(comp[k] / solid_c[k], 3) for k in SINGLE}
        rows.append(row)

    specialist = {k: next((r[f"C_{k}"] for r in rows if r["case"] == k), None) for k in SINGLE}
    for r in rows:
        if all(specialist.values()):
            ratios = [r[f"C_{k}"] / specialist[k] for k in SINGLE]
            r |= {f"C_{k}/spec": round(x, 3) for k, x in zip(SINGLE, ratios)}
            r["worst/spec"] = round(max(ratios), 3)
        r["ok"] = (abs(r["final_mass"] - MASS_GOAL) < 0.02 and r["pads_and_boss_solid"]
                   and r["elements_in_hole"] == 0 and r["pieces"] == 1)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "stage5_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "loads.json").write_text(json.dumps({"magnitudes": mags, "solid_compliance": solid_c}, indent=1))

    print(f"\nbalanced loads: Fz {mags['fz']} N, Fy {mags['fy']} N, Mx {mags['mx']} N·mm "
          f"(solid compliance {solid_c['fz']:.3f} N·mm each)")
    cols = ["iterations", "final_mass", "pieces", "ok"] + [f"C_{k}/solid" for k in SINGLE]
    cols += [f"C_{k}/spec" for k in SINGLE] + ["worst/spec"]
    print(f"{'case':>8} " + " ".join(f"{c:>11}" for c in cols))
    for r in rows:
        print(f"{r['case']:>8} " + " ".join(f"{str(r.get(c, '')):>11}" for c in cols))

    plotter = pv.Plotter(shape=(len(done), 3), window_size=(1800, 330 * len(done)), off_screen=True)
    for row, name in enumerate(done):
        surface = beso_solid(final_vtk(out / name)).extract_surface(algorithm="dataset_surface")
        for col, view in enumerate(["view_xz", "view_xy", "view_isometric"]):
            plotter.subplot(row, col)
            plotter.add_mesh(surface, color="lightsteelblue")
            plotter.add_text(f"{name} ({view[5:]})", font_size=10)
            getattr(plotter, view)()
            if col < 2:
                plotter.camera.zoom(1.3)
    plotter.screenshot(out / "stage5_designs.png")
    print(f"wrote {out / 'stage5_results.csv'} and {out / 'stage5_designs.png'}")

    ok = all(r["ok"] for r in rows)
    by_case = {r["case"]: r for r in rows}
    if "env" in by_case and all(k in by_case for k in SINGLE):
        best = min(rows, key=lambda r: r["worst/spec"])["case"]
        print(f"lowest worst-case ratio: {best}")
        ok &= best == "env"
    else:
        print("envelope check needs fz, fy, mx and env")
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help=f"default: all of {list(CASES)}")
    parser.add_argument("--size", type=float, default=2.5, help="mesh size in mm (default 2.5)")
    parser.add_argument("--live", action="store_true", help="export each iteration to viewer/latest.stl")
    parser.add_argument("--report-only", action="store_true", help="only re-evaluate finished cases")
    args = parser.parse_args()
    unknown = set(args.cases) - set(CASES)
    if unknown:
        parser.error(f"unknown cases {sorted(unknown)}")
    out = REPO_ROOT / "runs" / ("stage5" if args.size == 2.5 else f"stage5_h{args.size}")
    out.mkdir(parents=True, exist_ok=True)

    mesh = build_mesh(args.size, out)
    mags, solid_c = balance(mesh, out)
    print(f"balanced loads {mags}", flush=True)
    if not args.report_only:
        for name in args.cases or list(CASES):
            print(f"[{name}] steps {CASES[name]}", flush=True)
            print(run_case(name, mesh, out / f"cantilever_{name}.inp", mags, out, args.live), flush=True)
    ok = report(mesh, mags, solid_c, out)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
