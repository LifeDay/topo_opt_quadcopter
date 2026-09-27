"""Stage 0 pass check: run BESO's bundled example 1 (2D shell plate) headless.

Two variants:
- "fi": the example as shipped, with a 450 MPa von Mises limit. BESO freezes the mass
  once elements violate it, so it stops above mass_goal_ratio.
- "nofi": no stress limit, so the mass should reach mass_goal_ratio.
"""

import re
import sys

from topo_opt_quadcopter.beso_runner import BESO_DIR, REPO_ROOT, run_beso

INP = BESO_DIR / "wiki_files" / "example_1" / "Plane_mesh.inp"
MASS_GOAL = 0.4

CONF = '''
elset_name = "SolidMaterialElementGeometry2D"
domain_optimized[elset_name] = True
domain_density[elset_name] = [1e-6, 1]
domain_thickness[elset_name] = [1.0, 1.0]
domain_offset[elset_name] = 0.0
domain_orientation[elset_name] = []
domain_FI[elset_name] = {fi}
domain_material[elset_name] = ["*ELASTIC \\n210000e-6,  0.3", "*ELASTIC \\n210000,  0.3"]
domain_same_state[elset_name] = False
mass_goal_ratio = {mass_goal}
filter_list = [["simple", "auto"]]
optimization_base = "stiffness"
save_iteration_results = 5
save_resulting_format = "inp vtk"
'''

VARIANTS = {
    "fi": '[[("stress_von_Mises", 450.0e6)], [("stress_von_Mises", 450.0)]]',
    "nofi": "[]",
}


def mass_history(beso_log):
    """(iteration, mass) rows from the table BESO writes to <inp>.log."""
    rows = re.findall(r"^\s*(\d+)\s+([\d.]+)\s", beso_log.read_text(), re.MULTILINE)
    return [(int(i), float(m)) for i, m in rows]


if __name__ == "__main__":
    ok = True
    for name, fi in VARIANTS.items():
        run_dir = REPO_ROOT / "runs" / f"stage0_example1_{name}"
        run_beso(run_dir, INP, CONF.format(fi=fi, mass_goal=MASS_GOAL))
        history = mass_history(run_dir / "Plane_mesh.log")
        last_i, last_mass = history[-1]
        ratio = last_mass / history[0][1]
        results = sorted(run_dir.glob("file*_state1.inp"))
        print(f"{name}: {last_i} iterations, final mass ratio {ratio:.3f}, {len(results)} result meshes")
        ok &= bool(results)
        if name == "nofi" and abs(ratio - MASS_GOAL) > 0.02:
            print(f"  FAIL: expected mass ratio ~{MASS_GOAL}")
            ok = False
    sys.exit(0 if ok else 1)
