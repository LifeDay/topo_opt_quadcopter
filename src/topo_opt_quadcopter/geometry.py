"""build123d geometry with named regions, exported to STEP for meshing.

Each solid's label names its region. The label prefix sets its role (see REGION_PRIORITY):
solids may overlap, and where they do the higher-priority role wins when meshing.
Units: mm.
"""

from dataclasses import dataclass
from pathlib import Path

from build123d import Align, Box, Compound, Cylinder, Pos, Rot, Solid, export_step

# Highest priority first. "keep_out" is removed from the mesh; the rest become element sets.
REGION_PRIORITY = ["keep_out", "fixed", "load", "keep_in", "design"]


def region_role(name: str) -> str:
    for role in REGION_PRIORITY:
        if name.startswith(role):
            return role
    raise ValueError(f"solid label {name!r} does not start with one of {REGION_PRIORITY}")


def expected_region_volumes(solids: dict[str, Solid]) -> dict[str, float]:
    """Volume each region should have once overlaps are resolved by priority (CAD booleans)."""
    order = sorted(solids, key=lambda n: REGION_PRIORITY.index(region_role(n)))
    volumes, higher = {}, None
    for name in order:
        shape = solids[name] if higher is None else solids[name] - higher
        if region_role(name) != "keep_out":
            volumes[name] = shape.volume
        higher = solids[name] if higher is None else higher + solids[name]
    return volumes


def export_named_step(solids: dict[str, Solid], path: Path) -> Path:
    for name, solid in solids.items():
        region_role(name)
        solid.label = name
    export_step(Compound(children=list(solids.values())), str(path))
    return Path(path)


@dataclass
class Cantilever:
    """Prismatic cantilever along +x, clamped at x = 0, loaded at x = length.

    Cross-section is width (y) × height (z), centred on the x axis.
    """
    length: float = 200.0
    width: float = 10.0
    height: float = 20.0
    pad_length: float = 5.0
    hole_diameter: float = 0.0  # > 0 adds a keep-out hole through y at mid-span
    boss_diameter: float = 0.0  # > 0 adds a keep-in ring around the hole

    @property
    def area(self) -> float:
        return self.width * self.height

    def second_moment(self, axis: str = "z_bending") -> float:
        """I for deflection in z (about y) or in y (about z)."""
        b, h = self.width, self.height
        return b * h**3 / 12 if axis == "z_bending" else h * b**3 / 12

    def solids(self) -> dict[str, Solid]:
        section = (Align.MIN, Align.CENTER, Align.CENTER)
        w, h = self.width, self.height
        out = {
            "design": Box(self.length, w, h, align=section),
            "fixed_root": Box(self.pad_length, w, h, align=section),
            "load_tip": Pos(self.length - self.pad_length, 0, 0) * Box(self.pad_length, w, h, align=section),
        }
        mid = Pos(self.length / 2, 0, 0) * Rot(90, 0, 0)
        if self.boss_diameter > 0:
            out["keep_in_boss"] = mid * Cylinder(self.boss_diameter / 2, w)
        if self.hole_diameter > 0:
            out["keep_out_hole"] = mid * Cylinder(self.hole_diameter / 2, w)
        return out
