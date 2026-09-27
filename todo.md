# Topology optimization for a 3D-printed quadcopter — TODO

Goal: validate a scripted topology-optimization toolchain on a trivial cantilever beam
before optimizing real quadcopter parts. Load cases to support: applied force,
applied (non-zero) displacement, and modal.

## Decision: BESO + CalculiX (not dl4to4ocp)

Why dl4to4ocp was rejected (from reading its source, and that of dl4to underneath it):
- Static linear elasticity only. No modal analysis; displacements can only be fixed at
  zero (`pde.py:750`, `b[Ω_dirichlet] = 0`).
- SIMP given a list of problems optimizes each separately; no combined load cases.
- The force per voxel is the force vector × how deep the voxel is inside the load
  shape (`voxels.py:262`), so the total load changes with mesh resolution.
- Solves on the CPU with a finite-difference voxel grid; needs Python 3.14.

Other options considered: PyTopo3D (compliance only, one load case), TopOpt_in_PETSc
Python wrapper (scales best, dormant, compliance only), custom SIMP on scikit-fem or
FEniCSx (most capable, most code), and Altair Inspire / Fusion Generative Design
(commercial). A custom SIMP, starting from PyTopo3D, is the fallback if BESO fails
the stop criteria below.

## BESO facts (checked in https://github.com/calculix/beso source)

- Runs headless with only numpy and matplotlib; FreeCAD is optional.
- `beso_main.py:65` reads `beso_conf.py` **from the BESO folder** → a wrapper must write it for each run.
- Objectives (`optimization_base`): `stiffness`, `failure_index`, `buckling`, `heat`.
  **No frequency objective** → modal is a check on the results only.
- Several load cases = several CalculiX steps, combined as an **envelope**: each
  element uses its highest energy density across the steps (`beso_main.py:467`).
  The relative magnitudes of the loads matter.
- Applied non-zero displacement is fine in CalculiX, but the `stiffness` objective makes
  the part stiffer, which raises reaction forces and stresses under that displacement.
  Use `failure_index` (von Mises allowable) for those cases.
- Filters: simple, morphological (open/close), and `casting` (a possible proxy for
  printing without supports). **No symmetry filter** → model half or a quarter of the
  part with symmetry boundary conditions.
- Supported elements include C3D4, C3D10, C3D8, C3D20 and shells. Use **C3D10**; C3D4 is too stiff in bending.
- The mesh is never changed: removed elements stay at density 1e-6, so the output
  is a jagged set of elements that needs smoothing.

## Environment notes

- Linux Mint 22.3 (Ubuntu 24.04 base), 16 cores, 31 GB RAM, GTX 1070 Ti (unused by CalculiX).
- Nothing installed natively yet: no CalculiX, gmsh, build123d or ParaView.
- FreeCAD 1.1.3 is installed as a Flatpak (bundles CalculiX, but sandboxed). A FreeCAD MCP is available.
- Plan: CalculiX from apt (`calculix-ccx`), gmsh from pip, and a `uv` project on Python 3.12.

## Pipeline

```
build123d (named solids: design / keep-in / keep-out / fixed pads / load pads)
  → STEP → gmsh (fragment so the solids share nodes; physical groups; C3D10)
  → .inp writer (element sets, node sets, material, one step per load case)
  → BESO (wrapper writes beso_conf.py) → per-iteration result meshes
  → CalculiX *FREQUENCY on selected iterations
  → surface extraction + Taubin smoothing → union with keep-in parts (manifold3d) → STL
  → re-mesh + CalculiX check of the final shape
```

## Stages (cantilever)

Analytical references: δ = PL³/3EI; reaction under applied tip displacement = 3EIδ/L³;
f₁ = (1.875²/2π)·√(EI/ρAL⁴).

- [x] **0. Toolchain**: apt `calculix-ccx`; `uv` project with build123d, gmsh, meshio,
      pyvista, manifold3d, numpy, matplotlib; BESO as a git submodule.
      Pass: BESO's bundled example runs headless. (`scripts/stage0_beso_example.py`)
  - Done 2026-09-27. ccx 2.21 (apt); build123d 0.13, gmsh 4.15.2, Python 3.12, NumPy 2.
  - `beso_runner.run_beso` copies the BESO sources into `<run_dir>/_beso/`, writes
    `beso_conf.py` there, runs with `MPLBACKEND=Agg`. It patches one NumPy-2
    incompatibility (`np.linalg.linalg.norm`) in the copy; the submodule is untouched.
  - Example 1 (shell plate, 40% mass): about 60 iterations, about 80 s, clean truss.
  - **BESO runs are not deterministic**: identical inputs gave 58, 59 and 63 iterations, and
    one run froze at 79% mass after a one-off spike in the failure index at iteration 7.
    Check the cause (ccx threads vs BESO's `cpu_cores` multiprocessing) in stage 4.
  - The apt `ccx` links **SPOOLES only** (no PARDISO/PaStiX) against the reference `libblas`
    → stage 3 should try OpenBLAS (`update-alternatives`) or a source build.
  - [ ] **Remote progress viewer**: a one-page three.js viewer served from the headless Linux
        box, for watching in-progress geometry from Windows and Android browsers over Tailscale
    - [ ] Serve the output folder with `python3 -m http.server 8000 --directory <out_dir>`
      - Run it as a systemd user service so it survives reboots
      - Optional: use `tailscale serve` to get HTTPS on the tailnet instead of a bare port
    - [ ] Write `viewer.html` using three.js, STLLoader and OrbitControls, with the libraries
          copied into the folder so it doesn't depend on a CDN
      - Every ~5 s, send a HEAD request for `latest.stl` (use `cache: 'no-store'`) and reload
        only when `Last-Modified` changes
      - Replace the mesh without resetting the camera, and show the file's timestamp on screen
      - Check that orbit and pinch-zoom work on Android
- [x] **1. Geometry → mesh → `.inp`**: build123d cantilever with named regions → STEP →
      gmsh fragment → sets. Apply loads through `*COUPLING`/`*DISTRIBUTING` on a pad,
      not a single node. Pass: set counts and volumes match the CAD; solid-beam run completes.
      (`scripts/stage1_cantilever_mesh.py`; modules `geometry`, `mesh`, `inp_writer`, `ccx`)
  - Done 2026-09-27. Solid labels set the role by prefix: `keep_out` > `fixed` > `load` >
    `keep_in` > `design`. Overlaps go to the higher priority; `keep_out` is removed.
    Region volumes match the CAD booleans (exact for boxes; within 0.2% around a curved
    hole, since C3D10 is straight-sided). BC faces are chosen by a box on a region's boundary.
  - **ccx 2.21 does not output results for a `*DISTRIBUTING` reference node** (U and RF print
    as 0), **and ignores `*BOUNDARY` on it**. Read displacements from the face node set;
    prescribe displacements on the face nodes (see stage 2). Forces on it work.
  - [ ] Shared STL export helper for the viewer:
    - [ ] Make exports atomic: write `latest.stl.tmp`, then rename it to `latest.stl`, so the
          viewer never loads a half-written file
    - [ ] Export binary STL rather than ASCII; the files are several times smaller and load
          faster on the phone
  - Viewer done when: `http://<box>:8000/viewer.html` opens on both the PC and the phone, and
    a new export shows up within ~10 s without losing the current view
- [x] **2. Solver check (no optimization)**: tip force, applied tip displacement, and f₁
      vs the analytical values; C3D4 vs C3D10; mesh convergence. Pass: within 5% with C3D10.
      (`scripts/stage2_solver_check.py` → `runs/stage2/stage2_results.csv`)
  - Done 2026-09-27. 200×10×20 mm PETG beam (E 2100, ν 0.38). Error vs Euler–Bernoulli:

    | elem  | h mm | nodes | tip δ  | reaction | f₁ (y) | f₁ (z) | static s |
    |-------|------|-------|--------|----------|--------|--------|----------|
    | C3D10 | 10   | 1075  | −0.11% | +0.11%   | +0.75% | −0.20% | 0.1      |
    | C3D10 | 2.5  | 22506 | +0.01% | −0.01%   | +0.58% | −0.28% | 3.6      |
    | C3D10 | 1.5  | 85010 | +0.07% | −0.06%   | +0.53% | −0.32% | 35       |
    | C3D4  | 2.5  | 3466  | −5.41% | +5.72%   | +11.4% | +2.61% | 0.4      |
    | C3D4  | 1.0  | 34608 | −1.08% | +1.10%   | +2.72% | +0.28% | 7.4      |

    C3D10 is within 1% even with one element through the width; C3D4 is stiff unless fine.
  - Applied displacement: prescribe uz on all `LOAD` face nodes (the face can still rotate
    about y, so it matches 3EIδ/L³). This constrains the face more than a real pad would.
  - Stage 3 lead: user time ≈ wall time, so the apt SPOOLES solve is **single-threaded**
    apart from matrix setup (85k nodes: 35 s static, 55 s modal).
- [ ] **3. Scaling**: CalculiX time and memory vs element count, times ~30–60 BESO iterations.
      Check which linear solvers the CalculiX build has (SPOOLES slow; PARDISO/PaStiX faster).
      Pass: acceptable extrapolated run time for a quarter-model of the quad frame.
- [ ] **4. Single-load BESO**: `stiffness`, `mass_goal_ratio` ≈ 0.3, keep-out holes,
      keep-in bosses, filter radius, mesh sensitivity. Pass: converges; regions respected;
      sensible truss; topology holds when the mesh is refined.
  - [ ] Confirm the run writes intermediate snapshots. If it only exports at the end, add a
        periodic export step, or there's nothing to see mid-run. Note: with
        `save_iteration_results = 1`, BESO writes `.inp`/`.vtk` result meshes each iteration,
        not STL, so something must convert the newest one to `latest.stl` (surface of the
        solid elements, via the atomic export helper).
- [ ] **5. Several loads**: steps for Fy, Fz and torsion, individually and combined; scale one ×5.
      Pass: the combined result handles all loads; sensitivity to load magnitudes understood.
- [ ] **6. Applied displacement + stress**: non-zero `*BOUNDARY` step; `failure_index` with
      the PETG/PLA allowable stress divided by a safety factor. Pass: failure index ≤ 1;
      compare with the stiffness-objective run.
- [ ] **7. Modal check**: `*FREQUENCY` on each saved iteration → plot f₁ vs mass → choose
      the final iteration (avoid motor/prop frequency bands).
- [ ] **8. Symmetry and printability**: half-model with symmetry boundary conditions vs the
      full model; `casting` filter along the print Z axis (test the direction sign).
      Pass: results match; slices without supports.
- [ ] **9. Printable solid + re-check**: solid elements → surface → smoothing → union with
      keep-in parts → watertight STL → re-mesh → CalculiX. Pass: stiffness, failure index
      and f₁ within 10–15% of BESO's last iteration.
- [ ] **10. Physical calibration** (recommended): print the solid beam, hang a known weight,
      measure the deflection → effective E for the printer, material and orientation.

**Stop and reconsider BESO if:** stage 2 misses by more than 10% with C3D10, stage 3 is
many hours per run at the resolution a quad arm needs, or stage 9's smoothing loses
more than about 20% of the stiffness.

## Next step

Stages 0–2 are done (except the remote viewer and the STL export helper). Next: stage 3
(scaling; try OpenBLAS / a multithreaded solver), then stage 4. Check that BESO accepts the
generated decks, including `*COUPLING`, element-face `*SURFACE` and the extra reference node.
