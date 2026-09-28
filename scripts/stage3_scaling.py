"""Stage 3: CalculiX run time and memory vs mesh size, extrapolated to a BESO run.

Proxy for a quarter-model of the quad frame: a solid 150×40×20 mm block (one 5"-class arm
plus a quarter of the hub), clamped at one end with a tip force at the other. It is meshed
with C3D10 at decreasing sizes. Each size gets a static solve (what BESO runs each
iteration) and a 6-mode frequency solve (the stage 7 check). Static runs are timed with 1
and 16 solver threads up to THREAD_COMPARE_NODES, then with 16 only.

Sizes whose peak memory, extrapolated from the smaller runs, would exceed MEM_LIMIT_MB are
skipped. The solver is whatever find_ccx() picks (TOPO_CCX=ccx for the apt SPOOLES build).
Power-law fits (t = a·nᵇ) extrapolate to finer meshes and a full BESO run.
"""

import csv
import math
import sys

import numpy as np

from topo_opt_quadcopter.beso_runner import REPO_ROOT
from topo_opt_quadcopter.ccx import LOCAL_CCX, find_ccx, run_ccx
from topo_opt_quadcopter.geometry import Cantilever, export_named_step
from topo_opt_quadcopter.inp_writer import PETG, frequency_step, model_data, static_step, write_inp
from topo_opt_quadcopter.mesh import BoxSelect, mesh_step

OUT = REPO_ROOT / "runs" / "stage3"
BLOCK = Cantilever(length=150.0, width=40.0, height=20.0)
SIZES = [6.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.25]
THREADS = 16
THREAD_COMPARE_NODES = 150_000
MEM_LIMIT_MB = 22_000
# BESO run to extrapolate to: iterations × static steps (load cases) per iteration.
ITERATIONS, LOAD_STEPS = 50, 3
TARGET_SIZES = [2.0, 1.5, 1.25, 1.0]


def fit(n, y):
    """Least-squares y = a·nᵇ on log axes → (a, b)."""
    b, log_a = np.polyfit(np.log(n), np.log(y), 1)
    return math.exp(log_a), b


def predict(rows, key, nodes):
    pts = [(r["nodes"], r[key]) for r in rows if r.get(key)]
    if len(pts) < 2:
        return 0.0
    a, b = fit(*zip(*pts[-3:]))  # the largest runs set the trend
    return a * nodes**b


def run_size(size: float, threads: list[int]) -> dict:
    tag = f"h{size:g}"
    bcs = {"FIXED": ("fixed_root", BoxSelect.plane(0, 0.0)),
           "LOAD": ("load_tip", BoxSelect.plane(0, BLOCK.length))}
    mesh = mesh_step(OUT / "block.step", bcs, size=size, elem_type="C3D10")
    row = {"size": size, "nodes": len(mesh.node_ids), "elements": len(mesh.connectivity)}
    static = write_inp(OUT / f"{tag}_static.inp", model_data(mesh, PETG, couplings=["LOAD"]), [
        static_step(["FIXED"], loads=[("REF_LOAD", 3, -10.0)], print_nsets=["LOAD"])])
    for t in threads:
        run = run_ccx(static, threads=t)
        row[f"static_s_t{t}"], row[f"static_mb_t{t}"] = run.wall_s, run.max_rss_mb
        row["tip_uz"] = run.dat.get("displacements", "LOAD")[:, 3].mean()
    modal = run_ccx(write_inp(OUT / f"{tag}_modal.inp", model_data(mesh, PETG, couplings=[]),
                              [frequency_step(["FIXED"], n_modes=6)]), threads=THREADS)
    row["modal_s"], row["modal_mb"] = modal.wall_s, modal.max_rss_mb
    row["f1_hz"] = modal.dat.frequencies_hz[0]
    return row


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ccx = find_ccx()
    solver = "pardiso" if ccx == str(LOCAL_CCX) else "apt_spooles"
    print(f"ccx: {ccx} ({solver})")
    export_named_step(BLOCK.solids(), OUT / "block.step")
    t1, tn = "static_s_t1", f"static_s_t{THREADS}"
    print(f"{'h':>5}{'nodes':>9}{'elems':>9}{'static 1t':>11}{f'static {THREADS}t':>11}"
          f"{'static GB':>10}{'modal s':>9}{'modal GB':>9}{'tip uz':>9}{'f1 Hz':>8}")
    rows = []
    for size in SIZES:
        # Nodes scale as 1/h³; memory from the fit on nodes.
        if rows:
            est_nodes = rows[-1]["nodes"] * (rows[-1]["size"] / size) ** 3
            est_mb = max(predict(rows, f"static_mb_t{THREADS}", est_nodes), predict(rows, "modal_mb", est_nodes))
            if est_mb > MEM_LIMIT_MB:
                print(f"{size:>5g}  skipped: ~{est_nodes:,.0f} nodes, ~{est_mb / 1024:.0f} GB predicted")
                continue
            threads = [1, THREADS] if est_nodes < THREAD_COMPARE_NODES else [THREADS]
        else:
            threads = [1, THREADS]
        r = run_size(size, threads)
        rows.append(r)
        s1 = f"{r[t1]:>11.1f}" if t1 in r else f"{'-':>11}"
        print(f"{size:>5g}{r['nodes']:>9}{r['elements']:>9}{s1}{r[tn]:>11.1f}"
              f"{r[f'static_mb_t{THREADS}'] / 1024:>10.2f}{r['modal_s']:>9.1f}{r['modal_mb'] / 1024:>9.2f}"
              f"{r['tip_uz']:>9.4f}{r['f1_hz']:>8.1f}", flush=True)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"stage3_results_{solver}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Nodes per mm³·h³ from the finest run, to turn a target size into a node count.
    density = rows[-1]["nodes"] * rows[-1]["size"] ** 3
    for key, label in [(tn, "static"), (f"static_mb_t{THREADS}", "static memory"),
                       ("modal_s", "modal"), ("modal_mb", "modal memory")]:
        a, b = fit(*zip(*[(r["nodes"], r[key]) for r in rows[-3:]]))
        print(f"fit {label}: ∝ nodes^{b:.2f}")
    print(f"\nExtrapolated BESO run ({ITERATIONS} iterations × {LOAD_STEPS} static steps, "
          f"{THREADS} threads, solver time only):")
    for h in TARGET_SIZES:
        n = density / h**3
        step_s = predict(rows, tn, n)
        print(f"  h {h:>4g} mm: ~{n:>9,.0f} nodes, {step_s:>6.0f} s/step, "
              f"{predict(rows, f'static_mb_t{THREADS}', n) / 1024:>5.1f} GB, "
              f"BESO ≈ {ITERATIONS * LOAD_STEPS * step_s / 3600:>5.1f} h, "
              f"modal {predict(rows, 'modal_s', n):>6.0f} s / {predict(rows, 'modal_mb', n) / 1024:.1f} GB")
    sys.exit(0)
