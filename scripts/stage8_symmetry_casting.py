"""Stage 8: half model with symmetry boundary conditions, and BESO's casting filter.

The stage 4 block is cut at y = 0 and only the y ≥ 0 half is meshed. The cut face (SYM) gets:
  symmetric      uy = 0          for loads that mirror onto themselves (Fz)
  antisymmetric  ux = uz = 0     for loads that mirror onto their negative (Fy, Mx)
Each *STATIC step sets its own condition (OP=NEW), so the stage 5 envelope (Fz, Fy, Mx as
separate steps) is exact on the half: the full model's energy density under any of the three
loads is the mirror image of the half's. A single step mixing symmetric and antisymmetric
loads would not be.

Loads on the half are the half face's share of the full coupling's loads. Forces halve. The
full coupling spreads Mx as a traction M (0, -z, y) / J over the whole face (J = polar area
moment about the x axis); over the half face that is a net force (0, -M Sz/J, M Sy/J) plus a
moment M (J_half - ȳ Sy - z̄ Sz) / J about the half face's centroid, where REF_LOAD sits.
Compliances are reported for the full model: full load × tip response on the half.

Cases (h 2.5, filter 6 mm, mass goal 0.3, stage 5 balanced loads):
  sym_fz    Fz only                      vs stage 4 base / stage 5 fz (full model)
  sym_env   Fz, Fy, Mx envelope          vs stage 5 env
  cast_pz   Fz, casting filter (0, 0, +1) then simple
  cast_nz   Fz, casting filter (0, 0, -1) then simple
  cast_pz_fix  as cast_pz, with the casting filter's neighbour bug fixed (beso_fast_filter.py)
  cast_pz_fix_t3  as cast_pz_fix, casting tolerance 3 mm (about one element) instead of 6

The casting filter runs vectorized (beso_fast_filter.py); cast_pz and cast_nz were first run
with BESO's original, kept in <case>_orig/, and the rerun must give the same final design.

Printability: each design (the half ones mirrored) is sampled on a 1 mm grid over the whole
block; for a build direction, a solid voxel is unsupported when the voxel below it and its
four side neighbours below are all empty (45° overhang rule; the first layer is on the
plate). The solid block is the baseline (the roof of the keep-out hole in z). The voxels are
first closed and opened by one voxel (6-neighbour), since the jagged tet surfaces and walls one
element thick otherwise count as overhangs even on designs that are extrusions along the build
direction. The count is a relative measure; stage 9's smoothed surface is the real test.

Pass: half solid-block compliances within 1% of the full model's; sym_fz and sym_env within
10% of the full designs under every load; the usual region checks; printing along +z,
cast_pz has fewer unsupported voxels than cast_nz and at most 20% of sym_fz's (which fixes
the sign: the casting vector is the build direction); the vectorized filter reproduces the
original runs.

    uv run python scripts/stage8_symmetry_casting.py [case ...] [--live] [--report-only]
"""

import argparse
import csv
import dataclasses
import json
import sys

import numpy as np
import pyvista as pv
from build123d import Align, Box, Pos
from scipy import ndimage

from stage4_single_load import BEAM, beso_table, check_regions
from stage5_multi_load import SINGLE, final_vtk, solid_element_ids, tip_compliance
from topo_opt_quadcopter.beso_runner import REPO_ROOT, Domain, domains_conf, run_beso
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.geometry import export_named_step
from topo_opt_quadcopter.inp_writer import PETG, model_data, static_step, void_material, write_inp
from topo_opt_quadcopter.mesh import TET_FACES, BoxSelect, FEMesh, mesh_step
from topo_opt_quadcopter.stl_export import VIEWER_DIR, beso_solid

OUT = REPO_ROOT / "runs" / "stage8"
SIZE, FILTER, MASS_GOAL = 2.5, 6.0, 0.3
MAGS = json.loads((REPO_ROOT / "runs/stage5/loads.json").read_text())["magnitudes"]
FULL = {r["case"]: r for r in csv.DictReader(open(REPO_ROOT / "runs/stage5/stage5_results.csv"))}
SOLID_TOL, DESIGN_TOL, OVERHANG_RATIO = 0.01, 0.10, 0.2
REGIONS = ("design", "fixed_root", "load_tip", "keep_in_boss")


@dataclasses.dataclass
class Case:
    loads: list[str]
    filters: list
    reference: str | None = None  # stage 5 case to compare with
    casting_fix: bool = False


SIMPLE = ["simple", FILTER]
CASES = {
    "sym_fz": Case(["fz"], [SIMPLE], "fz"),
    "sym_env": Case(["fz", "fy", "mx"], [SIMPLE], "env"),
    "cast_pz": Case(["fz"], [["casting", FILTER, (0, 0, 1)], SIMPLE]),
    "cast_nz": Case(["fz"], [["casting", FILTER, (0, 0, -1)], SIMPLE]),
    "cast_pz_fix": Case(["fz"], [["casting", FILTER, (0, 0, 1)], SIMPLE], casting_fix=True),
    "cast_pz_fix_t3": Case(["fz"], [["casting", 3.0, (0, 0, 1)], SIMPLE], casting_fix=True),
}
# Full-model designs from earlier stages, for the printability table.
REFERENCES = {"full_fz": REPO_ROOT / "runs/stage5/fz", "full_env": REPO_ROOT / "runs/stage5/env"}
BUILD_DIRS = {"+z": (2, 1), "-z": (2, -1), "+y": (1, 1), "+x": (0, 1)}

CONF = """
mass_goal_ratio = {mass_goal}
filter_list = {filters!r}
optimization_base = "stiffness"
displacement_graph = [["LOAD", "total"]]
save_iteration_results = 1
save_resulting_format = "vtk"
"""


def build_half_mesh() -> FEMesh:
    cut = Pos(BEAM.length / 2, 0, 0) * Box(2 * BEAM.length, BEAM.width, 2 * BEAM.height,
                                          align=(Align.CENTER, Align.MIN, Align.CENTER))
    solids = {n: s & cut for n, s in BEAM.solids().items()}
    step = export_named_step(solids, OUT / "half.step")
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, BEAM.length)),
           "SYM": (REGIONS, BoxSelect.plane(1, 0.0))}
    return mesh_step(step, bcs, size=SIZE)


def face_moments(mesh: FEMesh, name: str) -> dict[str, float]:
    """Area, first moments Sy, Sz and polar second moment ∫(y² + z²) of a surface about the
    x axis, exact for its (straight-sided) triangles."""
    tri = np.concatenate([mesh._corner_coords([e], TET_FACES[f - 1]) for e, f in mesh.surfaces[name]])
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    y, z = tri[:, :, 1], tri[:, :, 2]
    second = lambda c: area / 12 * ((c ** 2).sum(axis=1) + c.sum(axis=1) ** 2)
    return {"A": area.sum(), "Sy": (area * y.mean(axis=1)).sum(), "Sz": (area * z.mean(axis=1)).sum(),
            "J": (second(y) + second(z)).sum()}


def half_step(mesh: FEMesh, load: str, value: float, print_load: bool = False) -> str:
    """One step: the half face's share of a full-model load, with its symmetry condition."""
    if load == "fz":
        bc, loads = [("SYM", 2, 0.0)], [("REF_LOAD", 3, value / 2)]
    elif load == "fy":
        bc, loads = [("SYM", 1, 0.0), ("SYM", 3, 0.0)], [("REF_LOAD", 2, value / 2)]
    else:
        m = face_moments(mesh, "LOAD")
        j_full = 2 * m["J"]
        yc, zc = m["Sy"] / m["A"], m["Sz"] / m["A"]
        bc = [("SYM", 1, 0.0), ("SYM", 3, 0.0)]
        loads = [("REF_LOAD", 2, -value * m["Sz"] / j_full), ("REF_LOAD", 3, value * m["Sy"] / j_full),
                 ("REF_LOAD", 4, value * (m["J"] - yc * m["Sy"] - zc * m["Sz"]) / j_full)]
    return static_step(["FIXED"], loads=loads, displacements=bc, print_nsets=["LOAD"] if print_load else [])


def evaluate(mesh: FEMesh, solid_ids: np.ndarray, path) -> dict[str, float]:
    """Full-model compliance under each stage 5 load alone, elements not in solid_ids void."""
    all_ids = np.concatenate(list(mesh.elsets.values()))
    void_ids = np.setdiff1d(all_ids, solid_ids)
    elsets = {"solid": np.sort(solid_ids)} | ({"void": void_ids} if len(void_ids) else {})
    model = model_data(dataclasses.replace(mesh, elsets=elsets), PETG, couplings=["LOAD"], dofs="1,6",
                       elset_materials={"void": void_material(PETG)})
    write_inp(path, model, [half_step(mesh, k, MAGS[k], print_load=True) for k in SINGLE])
    dat = run_ccx(path, threads=8).dat
    return {k: tip_compliance(dat, i, mesh, k, MAGS[k]) for i, k in enumerate(SINGLE)}


def run_case(name: str, mesh: FEMesh, live: bool) -> dict:
    case = CASES[name]
    model = model_data(mesh, PETG, couplings=["LOAD"], dofs="1,6")
    inp = write_inp(OUT / f"half_{name}.inp", model, [half_step(mesh, k, MAGS[k]) for k in case.loads])
    conf = domains_conf([Domain("design", True)] + [Domain(r, False) for r in REGIONS[1:]], PETG)
    conf += CONF.format(mass_goal=MASS_GOAL, filters=case.filters)
    run = run_beso(OUT / name, inp, conf, live_stl=VIEWER_DIR / "latest.stl" if live else None,
                   casting_fix=case.casting_fix)
    table = beso_table(run.run_dir / inp.with_suffix(".log").name)
    iters = len(table["i"])
    mass = table["mass"] / table["mass"][0]
    np.savetxt(run.run_dir / "history.csv", np.column_stack([mass, table["LOAD(u_total)"]]),
               delimiter=",", header="mass_ratio,max_tip_u_mm", comments="")
    summary = {"case": name, "steps": len(case.loads), "iterations": iters,
               "final_mass": round(float(mass[-1]), 4), "wall_s": round(run.wall_s, 1),
               "ccx_s_per_iter": round(sum(run.ccx_s) / iters, 2),
               "beso_s_per_iter": round(run.overhead_s / iters, 2)}
    (run.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def voxels(solid: pv.UnstructuredGrid, mirror: bool) -> np.ndarray:
    """Solid 1 mm voxels over the whole block, indexed [x, y, z]. mirror: the design is the
    y ≥ 0 half; sample it at |y|."""
    b = BEAM
    axes = [np.arange(0.5, b.length, 1.0), np.arange(-b.width / 2 + 0.5, b.width / 2, 1.0),
            np.arange(-b.height / 2 + 0.5, b.height / 2, 1.0)]
    p = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    q = p.reshape(-1, 3).copy()
    if mirror:
        q[:, 1] = np.abs(q[:, 1])
    return (solid.linear_copy().find_containing_cell(q) >= 0).reshape(p.shape[:3])


def clean(vox: np.ndarray) -> np.ndarray:
    """Close then open by one voxel (6-neighbour); edges padded so the outer faces stay."""
    p = np.pad(vox, 2, mode="edge")
    p = ndimage.binary_opening(ndimage.binary_closing(p))
    return p[2:-2, 2:-2, 2:-2]


def unsupported(vox: np.ndarray, axis: int, sign: int) -> int:
    """Solid voxels (above the first layer) with no solid voxel below them or below-and-beside
    them in the two other directions (45° rule), building along sign × axis."""
    v = np.moveaxis(vox, axis, -1)
    if sign < 0:
        v = v[..., ::-1]
    lower = v[..., :-1]
    support = lower.copy()
    for ax in (0, 1):
        for shift in (1, -1):
            rolled = np.roll(lower, shift, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = 0 if shift == 1 else -1
            rolled[tuple(edge)] = False
            support |= rolled
    return int((v[..., 1:] & ~support).sum())


def mirror_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


def report(mesh: FEMesh) -> bool:
    ok = True
    solid_c = evaluate(mesh, np.concatenate(list(mesh.elsets.values())), OUT / "solid_eval.inp")
    full_solid = json.loads((REPO_ROOT / "runs/stage5/loads.json").read_text())["solid_compliance"]
    print("solid block, full-model compliance from the half vs the full model:")
    for k in SINGLE:
        err = solid_c[k] / full_solid[k] - 1
        ok &= abs(err) <= SOLID_TOL
        print(f"  {k}: {solid_c[k]:.4f} vs {full_solid[k]:.4f} ({err:+.2%})")

    xs, zs = np.arange(0.5, BEAM.length, 1.0), np.arange(-BEAM.height / 2 + 0.5, BEAM.height / 2, 1.0)
    r_hole = np.hypot(*np.meshgrid(xs - BEAM.length / 2, zs, indexing="ij"))
    block_vox = np.broadcast_to((r_hole >= BEAM.hole_diameter / 2)[:, None, :],
                                (len(xs), int(BEAM.width), len(zs))).copy()
    baseline = {d: unsupported(clean(block_vox), *a) for d, a in BUILD_DIRS.items()}

    rows, grids = [], {}
    for name in [c for c in CASES if (OUT / c / "summary.json").exists()] + list(REFERENCES):
        if name in CASES:
            row = json.loads((OUT / name / "summary.json").read_text())
            vtk = final_vtk(OUT / name)
            regions = check_regions(vtk)
            comp = evaluate(mesh, solid_element_ids(mesh, vtk), OUT / name / "eval.inp")
            row |= {"pads_and_boss_solid": regions["pads_and_boss_solid"],
                    "elements_in_hole": regions["elements_in_hole"], "pieces": regions["pieces"]}
            row |= {f"C_{k}": round(comp[k], 4) for k in SINGLE}
            row |= {f"C_{k}/solid": round(comp[k] / full_solid[k], 3) for k in SINGLE}
            ref = CASES[name].reference
            if ref:
                ratios = {k: comp[k] / float(FULL[ref][f"C_{k}"]) for k in SINGLE}
                row |= {f"C_{k}/full_{ref}": round(x, 3) for k, x in ratios.items()}
                # an envelope must match under every load; a single-load design under its own
                keys = SINGLE if len(CASES[name].loads) > 1 else CASES[name].loads
                row["matches_full"] = all(abs(ratios[k] - 1) <= DESIGN_TOL for k in keys)
            row["ok"] = (abs(row["final_mass"] - MASS_GOAL) < 0.02 and regions["pads_and_boss_solid"]
                         and regions["elements_in_hole"] == 0 and regions["pieces"] == 1
                         and row.get("matches_full", True))
            ok &= row["ok"]
            solid, mirror = regions["solid"], True
        else:
            row = {"case": name} | {f"C_{k}/solid": round(float(FULL[name[5:]][f"C_{k}/solid"]), 3) for k in SINGLE}
            solid, mirror = beso_solid(final_vtk(REFERENCES[name])), False
        vox = clean(voxels(solid, mirror))
        grids[name] = (solid, mirror)
        row["solid_voxels"] = int(vox.sum())
        row["mirror_iou"] = round(mirror_iou(vox, vox[:, ::-1, :]), 3)
        row |= {f"unsupported_{d}": unsupported(vox, *a) - baseline[d] for d, a in BUILD_DIRS.items()}
        row["_vox"] = vox
        rows.append(row)

    by = {r["case"]: r for r in rows}
    for half, full in (("sym_fz", "full_fz"), ("sym_env", "full_env")):
        if half in by:
            by[half]["iou_vs_full"] = round(mirror_iou(by[half]["_vox"], by[full]["_vox"]), 3)
    if all(c in by for c in ("cast_pz", "cast_nz", "sym_fz")):
        u = {c: by[c]["unsupported_+z"] for c in ("cast_pz", "cast_nz", "sym_fz")}
        cast_ok = u["cast_pz"] < u["cast_nz"] and u["cast_pz"] <= OVERHANG_RATIO * u["sym_fz"]
        print(f"casting, unsupported voxels printing along +z: {u} ({'ok' if cast_ok else 'FAIL'})")
        ok &= cast_ok
    else:
        print("casting check needs cast_pz, cast_nz and sym_fz")
        ok = False
    for name in ("cast_pz", "cast_nz"):
        orig = OUT / f"{name}_orig"
        if name in by and orig.exists():
            a, b = (solid_element_ids(mesh, final_vtk(d)) for d in (OUT / name, orig))
            differ = len(np.setxor1d(a, b))
            print(f"{name}: {differ} elements differ from the run with BESO's original casting filter")
            by[name]["differs_from_orig"] = differ
            ok &= differ == 0

    fields = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with open(OUT / "stage8_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"unsupported voxels of the solid block (subtracted below): {baseline}")
    cols = ["iterations", "final_mass", "ccx_s_per_iter", "beso_s_per_iter", "ok"]
    cols += [f"C_{k}/solid" for k in SINGLE] + ["iou_vs_full", "mirror_iou"]
    cols += [f"unsupported_{d}" for d in BUILD_DIRS]
    print(f"{'case':>9} " + " ".join(f"{c[:13]:>13}" for c in cols))
    for r in rows:
        print(f"{r['case']:>9} " + " ".join(f"{str(r.get(c, '')):>13}" for c in cols))
    for r in rows:
        extra = {k: v for k, v in r.items() if "/full_" in k or k == "matches_full"}
        if extra:
            print(f"  {r['case']}: {extra}")

    plotter = pv.Plotter(shape=(len(rows), 3), window_size=(1800, 330 * len(rows)), off_screen=True)
    for i, r in enumerate(rows):
        solid, mirror = grids[r["case"]]
        surface = solid.extract_surface(algorithm="dataset_surface")
        if mirror:
            surface = surface.merge(surface.reflect((0, 1, 0), point=(0, 0, 0)))
        for col, view in enumerate(["view_xz", "view_xy", "view_isometric"]):
            plotter.subplot(i, col)
            plotter.add_mesh(surface, color="lightsteelblue")
            plotter.add_text(f"{r['case']} ({view[5:]})", font_size=10)
            getattr(plotter, view)()
            if col < 2:
                plotter.camera.zoom(1.3)
    plotter.screenshot(OUT / "stage8_designs.png")
    print(f"wrote {OUT / 'stage8_results.csv'} and {OUT / 'stage8_designs.png'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help=f"default: all of {list(CASES)}")
    parser.add_argument("--live", action="store_true", help="export each iteration to viewer/latest.stl")
    parser.add_argument("--report-only", action="store_true", help="only re-evaluate finished cases")
    args = parser.parse_args()
    unknown = set(args.cases) - set(CASES)
    if unknown:
        parser.error(f"unknown cases {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)

    mesh = build_half_mesh()
    print(f"half mesh: {len(mesh.node_ids)} nodes, {len(mesh.elsets['design'])} design elements, "
          f"SYM {len(mesh.nsets['SYM'])} nodes", flush=True)
    if not args.report_only:
        for name in args.cases or list(CASES):
            print(f"[{name}] {CASES[name]}", flush=True)
            print(run_case(name, mesh, args.live), flush=True)
    ok = report(mesh)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
