"""Stage 4 comparison of finished runs: pairwise IoU of the final designs and a render.

Reads runs/stage4/<case>/ (from stage4_single_load.py) and writes
runs/stage4/stage4_iou.csv and runs/stage4/stage4_designs.png.

    uv run python scripts/stage4_compare.py [case ...]
"""

import csv
import re
import sys
from pathlib import Path

import pyvista as pv

from stage4_single_load import CASES, OUT, design_grid, solid_mask
from topo_opt_quadcopter.stl_export import beso_solid

DEFAULT = ["h3.0_r6", "h2.5_r6", "h2.0_r6", "h2.5_rauto", "h2.5_r10"]


def final_vtk(case: str) -> Path:
    return max((OUT / case).glob("file*.vtk"), key=lambda p: int(re.sub(r"\D", "", p.stem)))


def main() -> None:
    cases = sys.argv[1:] or DEFAULT
    unknown = set(cases) - set(CASES)
    if unknown:
        sys.exit(f"unknown cases {sorted(unknown)}")
    solids = {c: beso_solid(final_vtk(c)) for c in cases}

    grid = design_grid()
    masks = {c: solid_mask(s, grid) for c, s in solids.items()}
    iou = {a: {b: float((masks[a] & masks[b]).sum() / (masks[a] | masks[b]).sum()) for b in cases}
           for a in cases}
    with open(OUT / "stage4_iou.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case"] + cases)
        writer.writerows([a] + [f"{iou[a][b]:.3f}" for b in cases] for a in cases)
    print(" " * 11 + "".join(f"{c:>11}" for c in cases))
    for a in cases:
        print(f"{a:>11}" + "".join(f"{iou[a][b]:>11.2f}" for b in cases))

    plotter = pv.Plotter(shape=(len(cases), 2), window_size=(1400, 350 * len(cases)), off_screen=True)
    for row, c in enumerate(cases):
        surface = solids[c].extract_surface(algorithm="dataset_surface")
        for col in range(2):
            plotter.subplot(row, col)
            plotter.add_mesh(surface, color="lightsteelblue")
            plotter.add_text(f"{c} ({final_vtk(c).stem})", font_size=10)
            if col == 0:
                plotter.view_xz()
                plotter.camera.zoom(1.3)
            else:
                plotter.view_isometric()
    plotter.screenshot(OUT / "stage4_designs.png")
    print(f"wrote {OUT / 'stage4_iou.csv'} and {OUT / 'stage4_designs.png'}")


if __name__ == "__main__":
    main()
