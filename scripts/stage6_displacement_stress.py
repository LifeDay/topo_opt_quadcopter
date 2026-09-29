"""Stage 6: applied tip displacement and a von Mises stress limit on the stage 4 cantilever.

PETG allowable: 50 MPa yield / safety factor 2 = 25 MPa, on the design domain and the keep-in
boss. The clamped and loaded pads carry the corner singularities of the boundary conditions,
so their peak is reported, not limited: BESO needs a criterion on every domain once any has
one (elements without one crash `import_FI_int_pt`), so the pads get an allowable of NO_LIMIT.

The displacement is prescribed in z on all LOAD face nodes (ccx 2.23 still ignores *BOUNDARY
on a *DISTRIBUTING reference node). Its size puts the solid block's peak design stress at
SOLID_UTIL × the allowable. The force case (on the tip coupling) is sized the same way at
FORCE_UTIL, so there is room for stress to rise as material is removed.

Cases, all with mass goal 0.3:
  disp_stiff      displacement, stiffness objective (no stress limit)
  disp_fi         displacement, failure_index objective
  force_stiff     force, stiffness objective (the stage 4 design; magnitude does not matter)
  force_fi        force, failure_index objective
  force_stiff_fi  force, stiffness objective with the stress limit (BESO freezes the mass
                  when more elements than at the start have FI >= 1, for either objective)

Each final design is re-solved under both loads (void elements at 1e-6 E, as in BESO): peak
von Mises (max over integration points) in the limited sets (design + boss) and the pads, elements with
FI >= 1, reaction force (displacement) or tip displacement (force). Pass: the usual stage 4
checks, and every stress-limited design has at most one element over the allowable
(BESO's FI_violated_tolerance) in the limited sets under its own load.

    uv run python scripts/stage6_displacement_stress.py [case ...] [--size MM] [--live] [--report-only]
"""

import argparse
import csv
import dataclasses
import json
import sys

import numpy as np

from stage4_single_load import beso_table, check_regions
from stage5_multi_load import build_mesh, final_vtk, solid_element_ids
from topo_opt_quadcopter.beso_runner import REPO_ROOT, Domain, domains_conf, run_beso
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.inp_writer import PETG, model_data, static_step, void_material, write_inp
from topo_opt_quadcopter.mesh import FEMesh
from topo_opt_quadcopter.stl_export import VIEWER_DIR

YIELD = 50.0      # MPa, PETG baseline
SAFETY = 2.0
ALLOWABLE = YIELD / SAFETY
SOLID_UTIL = 0.8  # displacement: solid-block peak design stress / allowable
FORCE_UTIL = 0.25
NO_LIMIT = 1e9  # MPa
LIMITED = ["design", "keep_in_boss"]  # element sets held to ALLOWABLE
FILTER = 6.0
MASS_GOAL = 0.3
LOADS = ["disp", "force"]


@dataclasses.dataclass
class Case:
    load: str
    objective: str       # BESO optimization_base
    stress_limit: bool   # domain_FI on the design domain and boss (NO_LIMIT on the pads)


CASES = {
    "disp_stiff": Case("disp", "stiffness", False),
    "disp_fi": Case("disp", "failure_index", True),
    "force_stiff": Case("force", "stiffness", False),
    "force_fi": Case("force", "failure_index", True),
    "force_stiff_fi": Case("force", "stiffness", True),
}

CONF = """
mass_goal_ratio = {mass_goal}
filter_list = [["simple", {filter_range!r}]]
optimization_base = {objective!r}
FI_violated_tolerance = 1
displacement_graph = [["LOAD", "total"]]
save_iteration_results = 1
save_resulting_format = "vtk"
"""


def load_step(load: str, value: float, print_elsets=()) -> str:
    """-z tip displacement (mm) on the LOAD face nodes, or -z force (N) on the coupling."""
    if load == "disp":
        return static_step(["FIXED"], displacements=[("LOAD", 3, -value)], print_nsets=["LOAD"],
                           print_elsets=print_elsets)
    return static_step(["FIXED"], loads=[("REF_LOAD", 3, -value)], print_nsets=["LOAD"],
                       print_elsets=print_elsets)


def von_mises(s: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, sxz, syz = s.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))


def element_vm(dat, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(element ids, max von Mises over each element's integration points) for set SOLID."""
    rows = dat.get("stresses", "SOLID", step)
    ids, inverse = np.unique(rows[:, 0].astype(int), return_inverse=True)
    vm = np.zeros(len(ids))
    np.maximum.at(vm, inverse, von_mises(rows[:, 2:]))
    return ids, vm


def evaluate(mesh: FEMesh, solid_ids: np.ndarray, mags: dict[str, float], path) -> dict:
    """Re-solve the design (void elsewhere) under each load: stresses, reaction or tip u."""
    all_ids = np.concatenate(list(mesh.elsets.values()))
    void_ids = np.setdiff1d(all_ids, solid_ids)
    elsets = {"solid": np.sort(solid_ids)} | ({"void": void_ids} if len(void_ids) else {})
    model = model_data(dataclasses.replace(mesh, elsets=elsets), PETG, couplings=["LOAD"],
                       elset_materials={"void": void_material(PETG)})
    write_inp(path, model, [load_step(k, mags[k], print_elsets=["solid"]) for k in LOADS])
    dat = run_ccx(path, threads=8).dat
    limited = np.concatenate([mesh.elsets[e] for e in LIMITED])
    res = {}
    for i, k in enumerate(LOADS):
        ids, vm = element_vm(dat, i)
        in_limited = np.isin(ids, limited)
        u = dat.get("displacements", "LOAD", i)
        rf = dat.get("forces", "LOAD", i)
        res[k] = {
            "vm_limited": float(vm[in_limited].max()),
            "vm_pads": float(vm[~in_limited].max()),
            "fi_violated": int((vm[in_limited] >= ALLOWABLE).sum()),
            "reaction_N": float(-rf[:, 3].sum()),
            "tip_u_mm": float(-u[:, 3].mean()),
        }
    return res


def size_loads(mesh: FEMesh, out) -> tuple[dict[str, float], dict]:
    """Displacement and force that give the solid block its target peak design stress."""
    unit = evaluate(mesh, np.concatenate(list(mesh.elsets.values())), {k: 1.0 for k in LOADS},
                    out / "solid_unit.inp")
    mags = {"disp": float(f"{SOLID_UTIL * ALLOWABLE / unit['disp']['vm_limited']:.3g}"),
            "force": float(f"{FORCE_UTIL * ALLOWABLE / unit['force']['vm_limited']:.3g}")}
    solid = evaluate(mesh, np.concatenate(list(mesh.elsets.values())), mags, out / "solid.inp")
    return mags, solid


def run_case(name: str, mesh: FEMesh, mags: dict[str, float], out, live: bool) -> dict:
    case = CASES[name]
    inp = write_inp(out / f"cantilever_{name}.inp", model_data(mesh, PETG, couplings=["LOAD"]),
                    [load_step(case.load, mags[case.load])])
    fi, pad_fi = (ALLOWABLE, NO_LIMIT) if case.stress_limit else (None, None)
    conf = domains_conf([Domain("design", True, fi=fi), Domain("fixed_root", False, fi=pad_fi),
                         Domain("load_tip", False, fi=pad_fi), Domain("keep_in_boss", False, fi=fi)], PETG)
    conf += CONF.format(mass_goal=MASS_GOAL, filter_range=FILTER, objective=case.objective)
    run = run_beso(out / name, inp, conf, live_stl=VIEWER_DIR / "latest.stl" if live else None)
    table = beso_table(run.run_dir / inp.with_suffix(".log").name)
    iters = len(table["i"])
    mass = table["mass"] / table["mass"][0]
    np.savetxt(run.run_dir / "history.csv", np.column_stack([mass, table["LOAD(u_total)"]]),
               delimiter=",", header="mass_ratio,max_tip_u_mm", comments="")
    summary = {
        "case": name, "load": case.load, "objective": case.objective, "stress_limit": case.stress_limit,
        "iterations": iters, "final_mass": round(float(mass[-1]), 4), "wall_s": round(run.wall_s, 1),
        "ccx_s_per_iter": round(sum(run.ccx_s) / iters, 2),
        "beso_s_per_iter": round(run.overhead_s / iters, 2),
    }
    (run.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def report(mesh: FEMesh, mags: dict[str, float], solid: dict, out) -> bool:
    done = [c for c in CASES if (out / c / "summary.json").exists()]
    rows = [{"case": "solid", "load": "", "final_mass": 1.0}
            | {f"{k}_{q}": round(v, 3) for k in LOADS for q, v in solid[k].items()}]
    for name in done:
        row = json.loads((out / name / "summary.json").read_text())
        vtk = final_vtk(out / name)
        regions = check_regions(vtk)
        row |= {"pads_and_boss_solid": regions["pads_and_boss_solid"],
                "elements_in_hole": regions["elements_in_hole"], "pieces": regions["pieces"]}
        ev = evaluate(mesh, solid_element_ids(mesh, vtk), mags, out / name / "eval.inp")
        row |= {f"{k}_{q}": round(v, 3) for k in LOADS for q, v in ev[k].items()}
        own = ev[row["load"]]
        row["own_fi_max"] = round(own["vm_limited"] / ALLOWABLE, 3)
        row["ok"] = (row["final_mass"] <= 1 and (row["final_mass"] - MASS_GOAL) > -0.02
                     and regions["pads_and_boss_solid"] and regions["elements_in_hole"] == 0
                     and regions["pieces"] == 1 and (not row["stress_limit"] or own["fi_violated"] <= 1))
        rows.append(row)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "stage6_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "loads.json").write_text(json.dumps({"allowable_MPa": ALLOWABLE, "magnitudes": mags,
                                                "solid": solid}, indent=1))

    print(f"\nallowable {ALLOWABLE} MPa; tip displacement {mags['disp']} mm, force {mags['force']} N")
    cols = ["final_mass", "iterations", "own_fi_max", "disp_vm_limited", "disp_fi_violated", "disp_reaction_N",
            "disp_vm_pads", "force_vm_limited", "force_fi_violated", "force_tip_u_mm", "force_vm_pads", "ok"]
    print(f"{'case':>15} " + " ".join(f"{c[:16]:>16}" for c in cols))
    for r in rows:
        print(f"{r['case']:>15} " + " ".join(f"{str(r.get(c, '')):>16}" for c in cols))
    print(f"wrote {out / 'stage6_results.csv'}")
    return all(r["ok"] for r in rows[1:]) and len(done) == len(CASES)


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
    out = REPO_ROOT / "runs" / ("stage6" if args.size == 2.5 else f"stage6_h{args.size}")
    out.mkdir(parents=True, exist_ok=True)

    mesh = build_mesh(args.size, out)
    mags, solid = size_loads(mesh, out)
    print(f"loads {mags}; solid block {solid}", flush=True)
    if not args.report_only:
        for name in args.cases or list(CASES):
            print(f"[{name}] {CASES[name]}", flush=True)
            print(run_case(name, mesh, mags, out, args.live), flush=True)
    ok = report(mesh, mags, solid, out)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
