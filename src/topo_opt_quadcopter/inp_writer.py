"""Write CalculiX .inp decks from an FEMesh.

Units: mm, N, MPa, t/mm³ (so frequencies come out in Hz).
Loads go through a reference node tied to a surface with *COUPLING / *DISTRIBUTING, so the
total load does not depend on the mesh.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import FEMesh


@dataclass
class Material:
    name: str
    E: float        # MPa
    nu: float
    density: float  # t/mm³


# Nominal datasheet-style values; stage 10 calibrates E for the actual printer.
PETG = Material("PETG", E=2100.0, nu=0.38, density=1.27e-9)


def void_material(material: Material, ratio: float = 1e-6) -> Material:
    """BESO's removed-element state: ratio × E and density."""
    return Material(f"{material.name}_VOID", material.E * ratio, material.nu, material.density * ratio)


def _rows(values, per_line: int = 16) -> str:
    values = list(values)
    return "\n".join(", ".join(str(v) for v in values[i:i + per_line]) + ","
                     for i in range(0, len(values), per_line))


def ref_node_ids(mesh: FEMesh, couplings: list[str]) -> dict[str, int]:
    """Reference node id for each coupling surface, numbered after the mesh nodes."""
    first = int(mesh.node_ids.max()) + 1
    return {name: first + i for i, name in enumerate(couplings)}


def model_data(mesh: FEMesh, material: Material, couplings: list[str], dofs: str = "1,3",
               elset_materials: dict[str, Material] | None = None) -> str:
    """Everything before the first *STEP. couplings lists surface names that get a reference
    node REF_<name> at the surface's area centroid. dofs: "1,3" for forces only, "1,6" to
    also take moments (*CLOAD on dofs 4-6 of the reference node; works in ccx 2.23).
    elset_materials overrides the material of some element sets.

    ccx does not report results for a *DISTRIBUTING reference node (U prints as zero), so
    read displacements from the surface's node set instead.
    """
    out = ["*HEADING", "topo_opt_quadcopter", "*NODE, NSET=Nall"]
    out += [f"{int(n)}, {x:.9g}, {y:.9g}, {z:.9g}" for n, (x, y, z) in zip(mesh.node_ids, mesh.coords)]

    refs = ref_node_ids(mesh, couplings)
    for name, ref in refs.items():
        x, y, z = mesh.surface_centroid(name)
        out += [f"*NODE, NSET=REF_{name}", f"{ref}, {x:.9g}, {y:.9g}, {z:.9g}"]

    for elset, ids in mesh.elsets.items():
        out.append(f"*ELEMENT, TYPE={mesh.elem_type}, ELSET={elset}")
        out += [f"{e}, " + ", ".join(str(int(n)) for n in mesh.connectivity[e]) for e in ids.tolist()]

    for name, nodes in mesh.nsets.items():
        out += [f"*NSET, NSET={name}", _rows(nodes.tolist())]
    for name, faces in mesh.surfaces.items():
        out.append(f"*SURFACE, NAME=S_{name}, TYPE=ELEMENT")
        out += [f"{e}, S{f}" for e, f in faces]

    by_elset = {elset: (elset_materials or {}).get(elset, material) for elset in mesh.elsets}
    for m in {m.name: m for m in by_elset.values()}.values():
        out += [f"*MATERIAL, NAME={m.name}", "*ELASTIC", f"{m.E:g}, {m.nu:g}", "*DENSITY", f"{m.density:g}"]
    for elset, m in by_elset.items():
        out += [f"*SOLID SECTION, ELSET={elset}, MATERIAL={m.name}"]

    for name, ref in refs.items():
        out += [f"*COUPLING, REF NODE={ref}, SURFACE=S_{name}, CONSTRAINT NAME=C_{name}",
                "*DISTRIBUTING", dofs]
    return "\n".join(out) + "\n"


def static_step(fixed: list[str], loads: list[tuple[str, int, float]] = (),
                displacements: list[tuple[str, int, float]] = (), print_nsets: list[str] = ()) -> str:
    """Linear static step. fixed: node sets clamped in 1-3. loads / displacements:
    (node set, dof, value) as *CLOAD / *BOUNDARY. print_nsets: sets whose U and RF go to .dat.

    OP=NEW drops the previous step's boundaries and loads; ccx carries them over otherwise,
    so each step of a multi-step deck is its own load case.
    """
    out = ["*STEP", "*STATIC", "*BOUNDARY, OP=NEW"]
    out += [f"{n}, 1, 3" for n in fixed]
    out += [f"{n}, {d}, {d}, {v:g}" for n, d, v in displacements]
    out += ["*CLOAD, OP=NEW"] + [f"{n}, {d}, {v:g}" for n, d, v in loads]
    out += ["*NODE FILE", "U", "*EL FILE", "S"]
    for n in print_nsets:
        out += [f"*NODE PRINT, NSET={n}", "U, RF"]
    for n in fixed:
        out += [f"*NODE PRINT, NSET={n}, TOTALS=ONLY", "RF"]
    out.append("*END STEP")
    return "\n".join(out) + "\n"


def frequency_step(fixed: list[str], n_modes: int = 6) -> str:
    out = ["*STEP", "*FREQUENCY", str(n_modes), "*BOUNDARY"]
    out += [f"{n}, 1, 3" for n in fixed]
    out += ["*NODE FILE", "U", "*END STEP"]
    return "\n".join(out) + "\n"


def write_inp(path: Path, model: str, steps: list[str]) -> Path:
    Path(path).write_text(model + "".join(steps))
    return Path(path)
