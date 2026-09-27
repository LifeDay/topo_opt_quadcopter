"""Stage 2: solid cantilever vs beam theory, C3D4 vs C3D10, mesh convergence.

Cases: tip force (deflection), applied tip displacement (root reaction), and modal (the
first bending mode in each direction). The displacement is prescribed on the LOAD face nodes,
not the coupling reference node: ccx 2.21 ignores *BOUNDARY on a *DISTRIBUTING reference node.
A uniform uz on the end face still lets it rotate about y, matching the free-rotation tip. Pass: C3D10 within 5% of Euler–Bernoulli on the
finest mesh. Timoshenko (shear) values are printed for context; at L/h = 10 shear adds
about 1% to the deflection.
"""

import csv
import math
import sys

from topo_opt_quadcopter.beso_runner import REPO_ROOT
from topo_opt_quadcopter.ccx import run_ccx
from topo_opt_quadcopter.geometry import Cantilever, export_named_step
from topo_opt_quadcopter.inp_writer import PETG, frequency_step, model_data, static_step, write_inp
from topo_opt_quadcopter.mesh import BoxSelect, mesh_step

OUT = REPO_ROOT / "runs" / "stage2"
P = -10.0       # N, tip force in z
DELTA = -1.0    # mm, applied tip displacement in z
SIZES = {"C3D10": [10.0, 5.0, 2.5, 1.5], "C3D4": [5.0, 2.5, 1.5, 1.0]}
TOL = 0.05

beam, mat = Cantilever(), PETG
L, A = beam.length, beam.area
I_z, I_y = beam.second_moment("z_bending"), beam.second_moment("y_bending")
G = mat.E / (2 * (1 + mat.nu))
kappa = 10 * (1 + mat.nu) / (12 + 11 * mat.nu)  # rectangular section (Cowper)
beta1 = 1.875104

ref = {
    "tip_uz": P * L**3 / (3 * mat.E * I_z),
    "reaction_z": 3 * mat.E * I_z * abs(DELTA) / L**3,
    "f1_y": beta1**2 / (2 * math.pi) * math.sqrt(mat.E * I_y / (mat.density * A * L**4)),
    "f1_z": beta1**2 / (2 * math.pi) * math.sqrt(mat.E * I_z / (mat.density * A * L**4)),
}
timoshenko_tip_uz = ref["tip_uz"] + P * L / (kappa * G * A)


def run_case(elem_type: str, size: float) -> dict:
    tag = f"{elem_type}_h{size:g}"
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, L))}
    mesh = mesh_step(OUT / "beam.step", bcs, size=size, elem_type=elem_type)
    loaded = model_data(mesh, mat, couplings=["LOAD"])
    force = run_ccx(write_inp(OUT / f"{tag}_force.inp", loaded, [
        static_step(["FIXED"], loads=[("REF_LOAD", 3, P)], print_nsets=["LOAD"])]))
    disp = run_ccx(write_inp(OUT / f"{tag}_disp.inp", loaded, [
        static_step(["FIXED"], displacements=[("LOAD", 3, DELTA)], print_nsets=["LOAD"])]))
    modal = run_ccx(write_inp(OUT / f"{tag}_modal.inp", model_data(mesh, mat, couplings=[]),
                              [frequency_step(["FIXED"], n_modes=6)]))

    f = modal.dat.frequencies_hz
    return {
        "elem": elem_type, "size": size, "nodes": len(mesh.node_ids), "elements": len(mesh.connectivity),
        "tip_uz": force.dat.get("displacements", "LOAD")[:, 3].mean(),
        "disp_tip_uz": disp.dat.get("displacements", "LOAD")[:, 3].mean(),
        "reaction_z": abs(disp.dat.get("total force", "FIXED")[0][2]),
        # First bending mode in each direction: nearest computed frequency to each reference.
        "f1_y": min(f, key=lambda x: abs(x - ref["f1_y"])),
        "f1_z": min(f, key=lambda x: abs(x - ref["f1_z"])),
        "modes": " ".join(f"{x:.1f}" for x in f),
        "static_s": force.wall_s, "modal_s": modal.wall_s,
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_named_step(beam.solids(), OUT / "beam.step")
    print(f"Euler–Bernoulli: tip uz {ref['tip_uz']:.4f} mm (Timoshenko {timoshenko_tip_uz:.4f}), "
          f"reaction {ref['reaction_z']:.4f} N, f1_y {ref['f1_y']:.2f} Hz, f1_z {ref['f1_z']:.2f} Hz\n")
    keys = ["tip_uz", "reaction_z", "f1_y", "f1_z"]
    print(f"{'elem':<6}{'h':>5}{'nodes':>8}" + "".join(f"{k:>12}" for k in keys)
          + f"{'disp uz':>9}{'static s':>9}{'modal s':>9}")
    rows = []
    for elem_type, sizes in SIZES.items():
        for size in sizes:
            r = run_case(elem_type, size)
            rows.append(r)
            errs = "".join(f"{100 * (r[k] / ref[k] - 1):>+11.2f}%" for k in keys)
            print(f"{elem_type:<6}{size:>5g}{r['nodes']:>8}{errs}{r['disp_tip_uz']:>9.4f}"
                  f"{r['static_s']:>9.1f}{r['modal_s']:>9.1f}", flush=True)

    with open(OUT / "stage2_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finest = [r for r in rows if r["elem"] == "C3D10"][-1]
    worst = max(abs(finest[k] / ref[k] - 1) for k in keys)
    ok = worst < TOL and abs(finest["disp_tip_uz"] / DELTA - 1) < 1e-6
    print(f"\nC3D10 finest mesh: worst error {100 * worst:.2f}% → {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
