"""Stage 7: modal and stress check on every saved BESO iteration, to choose the final one.

Baseline quad: 3" props, 3 blades, 3500 KV motors on 4S. Excitation band: the rotor
frequency (1P) from idle to full throttle, and the blade pass (3P) over the same range:
  1P  IDLE_RPM/60 = 67 Hz  to  KV × 4.2 V × 4 / 60 = 980 Hz  (no-load RPM at full charge,
                                                              an upper bound)
  3P  200 Hz to 2940 Hz
so no mode may lie in 67–2940 Hz. Idle is Betaflight dynamic idle for a 3" (about 4000 rpm).

The motor + prop (MOTOR_MASS) sits on the load pad, spread over the LOAD face nodes by their
share of its area (ccx 2.23 ignores a *MASS on the *DISTRIBUTING reference node). check_tip_mass
compares this with the exact Euler–Bernoulli tip-mass frequency on the stage 2 beam.

Each iteration's design is solved with the void elements removed, not kept at 1e-6 E: void
elements have the same E/ρ as solid, so large void regions could add spurious modes, and pieces
not connected to the clamped pad would have zero-frequency modes. Elements not joined to the
clamped piece through shared faces are dropped (their mass is reported): BESO's hard 0/1
switching leaves slivers hanging by an edge or a node, which hinge and gave 0 Hz modes. The final iteration is also solved with void kept, and without the
motor, for comparison.

Sources (existing BESO runs at h 2.5 mm, every iteration saved) and their own loads for the
von Mises check against the stage 6 allowable (25 MPa) on the design domain and boss:
  env          stage 5 envelope: Fz, Fy, Mx as three steps (small loads; stress is low)
  force_stiff  stage 6: 658 N tip force (same design as the stage 4 base)
  disp_stiff   stage 6: 1.9 mm tip displacement

Choice: the lightest iteration with no computed mode in the band and FI ≤ 1. Pass: tip-mass
check within 2%, every iteration evaluated.

    uv run python scripts/stage7_modal.py [source ...] [--every N] [--report-only]
"""

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from stage5_multi_load import UNIT, build_mesh, solid_element_ids
from stage6_displacement_stress import ALLOWABLE, von_mises
from topo_opt_quadcopter.beso_runner import REPO_ROOT
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.geometry import Cantilever, export_named_step
from topo_opt_quadcopter.inp_writer import PETG, frequency_step, model_data, static_step, void_material, write_inp
from topo_opt_quadcopter.mesh import TET_FACES, BoxSelect, FEMesh, mesh_step

OUT = REPO_ROOT / "runs" / "stage7"
KV, CELLS, V_CELL, IDLE_RPM, BLADES = 3500, 4, 4.2, 4000, 3
RPM_MAX = KV * CELLS * V_CELL
BAND_1P = (IDLE_RPM / 60, RPM_MAX / 60)
BAND_3P = (BLADES * BAND_1P[0], BLADES * BAND_1P[1])
BAND = (BAND_1P[0], BAND_3P[1])
MOTOR_MASS = 10.5e-6  # t: 9 g motor + 1.5 g prop
N_MODES = 6
LIMITED = ["design", "keep_in_boss"]
TIP_MASS_TOL = 0.02


def _stage5_steps():
    mags = json.loads((REPO_ROOT / "runs/stage5/loads.json").read_text())["magnitudes"]
    return [static_step(["FIXED"], loads=[("REF_LOAD", UNIT[k][0], mags[k])], print_elsets=LIMITED)
            for k in ("fz", "fy", "mx")]


def _stage6_steps(load: str):
    mags = json.loads((REPO_ROOT / "runs/stage6/loads.json").read_text())["magnitudes"]
    if load == "disp":
        return [static_step(["FIXED"], displacements=[("LOAD", 3, -mags["disp"])], print_elsets=LIMITED)]
    return [static_step(["FIXED"], loads=[("REF_LOAD", 3, -mags["force"])], print_elsets=LIMITED)]


SOURCES = {
    "env": ("stage5/env", _stage5_steps),
    "force_stiff": ("stage6/force_stiff", lambda: _stage6_steps("force")),
    "disp_stiff": ("stage6/disp_stiff", lambda: _stage6_steps("disp")),
}


def tip_mass_frequency(E, I, rho, A, L, tip_mass) -> float:
    """First bending frequency of a clamped Euler–Bernoulli beam with a point mass at the tip:
    1 + cos β cosh β + μ β (cos β sinh β − sin β cosh β) = 0, μ = tip mass / beam mass."""
    mu = tip_mass / (rho * A * L)
    beta = brentq(lambda b: 1 + math.cos(b) * math.cosh(b)
                  + mu * b * (math.cos(b) * math.sinh(b) - math.sin(b) * math.cosh(b)), 0.5, 2.0)
    return beta ** 2 / (2 * math.pi) * math.sqrt(E * I / (rho * A * L ** 4))


def check_tip_mass() -> dict:
    """The stage 2 beam (200×10×20 mm, h 2.5) with MOTOR_MASS on the tip face, vs the exact value."""
    beam, out = Cantilever(), OUT / "tip_mass_check"
    out.mkdir(parents=True, exist_ok=True)
    step = export_named_step(beam.solids(), out / "beam.step")
    mesh = mesh_step(step, {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
                            "LOAD": ("load_tip", BoxSelect.plane(0, beam.length))}, size=2.5)
    model = model_data(mesh, PETG, couplings=["LOAD"], point_masses={"LOAD": MOTOR_MASS})
    f = run_ccx(write_inp(out / "beam_tip_mass.inp", model, [frequency_step(["FIXED"], 4)]), threads=8).dat.frequencies_hz
    res = {}
    for axis, name in (("y", "y_bending"), ("z", "z_bending")):
        exact = tip_mass_frequency(PETG.E, beam.second_moment(name), PETG.density, beam.area, beam.length, MOTOR_MASS)
        fe = min(f, key=lambda x: abs(x - exact))
        res[axis] = {"exact_hz": round(exact, 2), "ccx_hz": round(fe, 2), "error": round(fe / exact - 1, 4)}
    return res


def element_volumes(mesh: FEMesh) -> dict[int, float]:
    return {int(e): float(v) for n, ids in mesh.elsets.items() for e, v in zip(ids, mesh.element_volumes(n))}


def clamped_piece(mesh: FEMesh, solid_ids: np.ndarray) -> np.ndarray:
    """The solid elements connected through shared faces to an element on the FIXED node set.
    Elements joined only by an edge or a node hinge about it (0 Hz modes in some iterations)."""
    conn = np.array([mesh.connectivity[int(e)] for e in solid_ids])
    faces = np.sort(conn[:, np.array(TET_FACES)], axis=2).reshape(-1, 3)
    _, face_id, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    shared = np.flatnonzero(counts[face_id.ravel()] == 2)
    order = shared[np.argsort(face_id.ravel()[shared], kind="stable")]
    pairs = (order // len(TET_FACES)).reshape(-1, 2)
    n_e = len(solid_ids)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_e, n_e))
    _, labels = connected_components(graph, directed=False)
    on_fixed = np.isin(conn, mesh.nsets["FIXED"]).any(axis=1)
    return solid_ids[np.isin(labels, np.unique(labels[on_fixed]))]


def classify_modes(dat, mesh: FEMesh, n: int) -> list[dict]:
    """Per mode, from the LOAD face: the dominant tip motion (x, y, z translation or twist
    about x, the twist scaled by the face's rms radius) and the share of the modal kinetic
    energy in the motor mass (ccx mass-normalizes the modes, so Σ m u² = 1 over the model)."""
    ids, weights = mesh.surface_node_weights("LOAD")
    out = []
    for k in range(n):
        u = dat.get("displacements", "LOAD", k)
        xyz = mesh.coords[np.searchsorted(mesh.node_ids, u[:, 0].astype(int))]
        r = xyz[:, 1:] - xyz[:, 1:].mean(axis=0)
        a = np.r_[-r[:, 1], r[:, 0]]
        theta = float(a @ np.r_[u[:, 2], u[:, 3]] / (a @ a))
        motion = {"x": abs(u[:, 1].mean()), "y": abs(u[:, 2].mean()), "z": abs(u[:, 3].mean()),
                  "twist": abs(theta) * math.sqrt((a @ a) / len(u))}
        uu = (u[:, 1:] ** 2).sum(axis=1)
        w = dict(zip(ids.tolist(), weights))
        tip = MOTOR_MASS * sum(w.get(int(nid), 0.0) * x for nid, x in zip(u[:, 0], uu))
        out.append({"dir": max(motion, key=motion.get), "tip_energy": tip})
    return out


def solve(mesh: FEMesh, solid_ids: np.ndarray, steps: list[str], path, void: bool = False,
          motor: bool = True) -> dict:
    """Modes (and stresses under steps) of the design. void=False: removed elements are
    dropped; True: kept at 1e-6 E and density, as BESO sees them."""
    masses = {"LOAD": MOTOR_MASS} if motor else None
    if void:
        all_ids = np.concatenate(list(mesh.elsets.values()))
        void_ids = np.setdiff1d(all_ids, solid_ids)
        sub = dataclasses.replace(mesh, elsets={"solid": np.sort(solid_ids)} | ({"void": void_ids} if len(void_ids) else {}))
        model = model_data(sub, PETG, couplings=["LOAD"], dofs="1,6", point_masses=masses,
                           elset_materials={"void": void_material(PETG)})
    else:
        sub = mesh.subset(solid_ids)
        model = model_data(sub, PETG, couplings=["LOAD"], dofs="1,6", point_masses=masses)
    run = run_ccx(write_inp(path, model, [frequency_step(["FIXED"], N_MODES, print_nsets=["LOAD"])] + steps),
                  threads=8)
    freqs = run.dat.frequencies_hz[:N_MODES]
    res = {"freqs": freqs, "modes": classify_modes(run.dat, sub, N_MODES), "ccx_s": run.wall_s}
    if steps:
        vm = [von_mises(run.dat.get("stresses", e.upper(), i)[:, 2:]).max()
              for i in range(len(steps)) for e in LIMITED if e in sub.elsets]
        res["vm_max"] = float(max(vm))
    for p in path.parent.glob(f"{path.stem}.*"):
        if p.suffix in (".frd", ".dat", ".sta", ".cvg", ".12d"):
            p.unlink()
    return res


def in_band(freqs) -> bool:
    return any(BAND[0] <= f <= BAND[1] for f in freqs)


def iteration_vtks(run_dir) -> list:
    return sorted(run_dir.glob("file[0-9]*.vtk"), key=lambda p: int(re.sub(r"\D", "", p.stem)))


FIELDS = (["iteration", "design_ratio", "mass_g", "dropped_g"] + [f"f{k + 1}" for k in range(N_MODES)]
          + [f"mode{k + 1}" for k in range(N_MODES)] + ["vm_max", "fi_max", "in_band", "ccx_s"])


def scan(name: str, mesh: FEMesh, vols: dict[int, float], every: int) -> list[dict]:
    """Evaluate every `every`-th iteration (and the last); resumes from <name>/iterations.csv."""
    src, make_steps = SOURCES[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "iterations.csv"
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    done = {int(r["iteration"]) for r in rows}
    vtks = iteration_vtks(REPO_ROOT / "runs" / src)
    picks = sorted(set(range(0, len(vtks), every)) | {len(vtks) - 1})
    design_ratio = np.loadtxt(REPO_ROOT / "runs" / src / "history.csv", delimiter=",", skiprows=1, ndmin=2)[:, 0]
    steps = make_steps()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not rows:
            writer.writeheader()
        for i in picks:
            if i in done:
                continue
            solid = solid_element_ids(mesh, vtks[i])
            kept = clamped_piece(mesh, solid)
            res = solve(mesh, kept, steps, out / "eval.inp")
            mass = sum(vols[int(e)] for e in solid)
            kept_mass = sum(vols[int(e)] for e in kept)
            row = {"iteration": i, "design_ratio": round(float(design_ratio[i]), 4),
                   "mass_g": round(kept_mass * PETG.density * 1e6, 2),
                   "dropped_g": round((mass - kept_mass) * PETG.density * 1e6, 3),
                   **{f"f{k + 1}": round(f, 1) for k, f in enumerate(res["freqs"])},
                   **{f"mode{k + 1}": f"{m['dir']}:{m['tip_energy']:.2f}" for k, m in enumerate(res["modes"])},
                   "vm_max": round(res["vm_max"], 2), "fi_max": round(res["vm_max"] / ALLOWABLE, 3),
                   "in_band": in_band(res["freqs"]), "ccx_s": round(res["ccx_s"], 1)}
            writer.writerow(row)
            fh.flush()
            rows.append({k: str(v) for k, v in row.items()})
            print(f"[{name}] it {i:3d} design {row['design_ratio']:.3f} ({row['mass_g']} g) "
                  f"f {[row[f'f{k + 1}'] for k in range(3)]} FI {row['fi_max']} {row['ccx_s']} s", flush=True)
    return sorted(rows, key=lambda r: int(r["iteration"]))


def compare_final(name: str, mesh: FEMesh) -> dict:
    """The final design: void removed (as in the scan) vs kept at 1e-6, and without the motor."""
    out = OUT / name
    solid = solid_element_ids(mesh, iteration_vtks(REPO_ROOT / "runs" / SOURCES[name][0])[-1])
    kept = clamped_piece(mesh, solid)
    res = {}
    for tag, kw in {"trimmed": {}, "void_kept": {"void": True}, "no_motor": {"motor": False}}.items():
        r = solve(mesh, solid if tag == "void_kept" else kept, [], out / f"final_{tag}.inp", **kw)
        res[tag] = [round(f, 1) for f in r["freqs"]]
        res[f"{tag}_modes"] = [m["dir"] for m in r["modes"]]
    return res


def choose(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r["in_band"] == "False" and float(r["fi_max"]) <= 1]
    return min(ok, key=lambda r: float(r["mass_g"])) if ok else None


def plot(results: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5), squeeze=False)
    for ax, (name, rows) in zip(axes[0], results.items()):
        m = np.array([float(r["design_ratio"]) for r in rows])
        for k in range(3):
            ax.plot(m, [float(r[f"f{k + 1}"]) for r in rows], marker=".", ms=3, lw=1, label=f"f{k + 1}")
        ax.axhspan(*BAND_1P, color="tab:red", alpha=0.12, label="1P (rotor)")
        ax.axhspan(*BAND_3P, color="tab:orange", alpha=0.12, label="3P (blade pass)")
        ax.set(xlabel="design-domain mass ratio (BESO)", ylabel="frequency (Hz)", title=name, yscale="log")
        ax.invert_xaxis()
        fi = ax.twinx()
        fi.plot(m, [float(r["fi_max"]) for r in rows], color="k", ls="--", lw=1, label="FI (von Mises / 25 MPa)")
        fi.set_ylabel("failure index")
        fi.set_ylim(0, max(1.2, max(float(r["fi_max"]) for r in rows) * 1.1))
        lines = ax.get_legend_handles_labels()[0] + fi.get_legend_handles_labels()[0]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "stage7_modal.png", dpi=130)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", help=f"default: all of {list(SOURCES)}")
    parser.add_argument("--every", type=int, default=1, help="evaluate every N-th iteration (default 1)")
    parser.add_argument("--report-only", action="store_true", help="only report iterations already evaluated")
    args = parser.parse_args()
    unknown = set(args.sources) - set(SOURCES)
    if unknown:
        parser.error(f"unknown sources {sorted(unknown)}")
    names = args.sources or list(SOURCES)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"band: 1P {BAND_1P[0]:.0f}–{BAND_1P[1]:.0f} Hz, 3P {BAND_3P[0]:.0f}–{BAND_3P[1]:.0f} Hz "
          f"→ avoid {BAND[0]:.0f}–{BAND[1]:.0f} Hz; motor + prop {MOTOR_MASS * 1e6:g} g", flush=True)
    tip = check_tip_mass()
    print(f"tip-mass check: {tip}", flush=True)
    ok = all(abs(v["error"]) <= TIP_MASS_TOL for v in tip.values())

    mesh = build_mesh(2.5, OUT)
    vols = element_volumes(mesh)
    results, summary = {}, {"band_hz": BAND, "band_1p_hz": BAND_1P, "band_3p_hz": BAND_3P,
                            "motor_mass_g": MOTOR_MASS * 1e6, "tip_mass_check": tip}
    for name in names:
        start = time.perf_counter()
        if args.report_only:
            path = OUT / name / "iterations.csv"
            results[name] = sorted(csv.DictReader(open(path)), key=lambda r: int(r["iteration"])) if path.exists() else []
        else:
            results[name] = scan(name, mesh, vols, args.every)
        n_vtk = len(iteration_vtks(REPO_ROOT / "runs" / SOURCES[name][0]))
        ok &= args.every == 1 and len(results[name]) == n_vtk
        final = compare_final(name, mesh)
        pick = choose(results[name])
        summary[name] = {"iterations": n_vtk, "evaluated": len(results[name]), "final": final,
                         "first": {k: results[name][0][k] for k in ("design_ratio", "mass_g", "f1", "f2", "f3")},
                         "last": {k: results[name][-1][k] for k in ("design_ratio", "mass_g", "f1", "f2", "f3", "fi_max")},
                         "any_outside_band": any(r["in_band"] == "False" for r in results[name]),
                         "choice": pick and {k: pick[k] for k in ("iteration", "design_ratio", "mass_g", "f1", "fi_max")},
                         "scan_s": round(time.perf_counter() - start, 1)}
        print(f"[{name}] {json.dumps(summary[name])}", flush=True)
    (OUT / "stage7_summary.json").write_text(json.dumps(summary, indent=1))
    plot(results)
    print(f"wrote {OUT / 'stage7_summary.json'} and {OUT / 'stage7_modal.png'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
