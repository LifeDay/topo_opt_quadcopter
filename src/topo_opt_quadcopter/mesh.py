"""Mesh a named-region STEP with gmsh into tetrahedra with CalculiX-ready sets.

The solids are fragmented so that regions share nodes on their interfaces. Where solids
overlap, the region with the highest priority (geometry.REGION_PRIORITY) owns the volume.
Boundary-condition surfaces are chosen as the boundary faces of a named region that lie
inside a box, e.g. the root end face of the fixed pad. A surface can span several regions
(e.g. a symmetry plane cutting all of them).
"""

from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from .geometry import REGION_PRIORITY, region_role

# gmsh node order → CalculiX (Abaqus) node order. For tet10, gmsh's last two mid-edge
# nodes (edges 3-4 and 2-4) are swapped relative to CalculiX.
GMSH_TYPES = {"C3D4": (4, [0, 1, 2, 3]), "C3D10": (11, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8])}
# CalculiX faces of a tet, as 0-based corner indices: S1 = 1-2-3, S2 = 1-4-2, S3 = 2-4-3, S4 = 3-4-1.
TET_FACES = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
# Mid-edge nodes (0-based, CalculiX C3D10 order) of each face, in the same order.
TET10_FACE_MIDS = [(4, 5, 6), (7, 8, 4), (8, 9, 5), (9, 7, 6)]


@dataclass
class BoxSelect:
    """Axis-aligned box; a surface is selected when its bounding box lies inside."""
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @staticmethod
    def plane(axis: int, value: float, tol: float = 1e-6) -> "BoxSelect":
        lo, hi = [-np.inf] * 3, [np.inf] * 3
        lo[axis], hi[axis] = value - tol, value + tol
        return BoxSelect(tuple(lo), tuple(hi))

    def contains(self, bbox) -> bool:
        return all(self.lo[i] <= bbox[i] and bbox[i + 3] <= self.hi[i] for i in range(3))


@dataclass
class FEMesh:
    elem_type: str
    node_ids: np.ndarray                  # (N,)
    coords: np.ndarray                    # (N, 3)
    elsets: dict[str, np.ndarray]         # region name → element ids
    connectivity: dict[int, np.ndarray]   # element id → node ids, CalculiX order
    nsets: dict[str, np.ndarray] = field(default_factory=dict)
    surfaces: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # (elem id, face 1-4)
    cad_volumes: dict[str, float] = field(default_factory=dict)  # per region, after fragment

    def _corner_coords(self, elem_ids, corners=(0, 1, 2, 3)) -> np.ndarray:
        index = {n: i for i, n in enumerate(self.node_ids)}
        return self.coords[[[index[self.connectivity[e][k]] for k in corners] for e in elem_ids]]

    def surface_faces(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(area, centroid) of each element face in a surface, from its corner nodes."""
        tris = np.concatenate([self._corner_coords([e], TET_FACES[f - 1]) for e, f in self.surfaces[name]])
        areas = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
        return areas, tris.mean(axis=1)

    def surface_area(self, name: str) -> float:
        return float(self.surface_faces(name)[0].sum())

    def surface_centroid(self, name: str) -> np.ndarray:
        areas, centroids = self.surface_faces(name)
        return (areas[:, None] * centroids).sum(axis=0) / areas.sum()

    def surface_node_weights(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(node ids, weights summing to 1): each node's share of a uniform load on the
        surface. A 6-node triangle puts it all on the mid-edge nodes (A/3 each); a 3-node
        triangle on the corners."""
        areas, _ = self.surface_faces(name)
        nodes, shares = [], []
        for (e, f), area in zip(self.surfaces[name], areas):
            conn = self.connectivity[e]
            local = TET10_FACE_MIDS[f - 1] if len(conn) == 10 else TET_FACES[f - 1]
            nodes += [conn[k] for k in local]
            shares += [area / 3] * 3
        ids, inverse = np.unique(nodes, return_inverse=True)
        weights = np.bincount(inverse, weights=shares)
        return ids, weights / weights.sum()

    def subset(self, elem_ids) -> "FEMesh":
        """The mesh with only elem_ids: element sets, node sets and surfaces are trimmed to
        them, and nodes no kept element uses are dropped."""
        keep = np.asarray(elem_ids)
        elsets = {n: ids[np.isin(ids, keep)] for n, ids in self.elsets.items()}
        elsets = {n: ids for n, ids in elsets.items() if len(ids)}
        connectivity = {int(e): self.connectivity[int(e)] for ids in elsets.values() for e in ids}
        used = np.unique(np.concatenate(list(connectivity.values())))
        in_used = np.isin(self.node_ids, used)
        return FEMesh(self.elem_type, self.node_ids[in_used], self.coords[in_used], elsets, connectivity,
                      {n: ids[np.isin(ids, used)] for n, ids in self.nsets.items()},
                      {n: [(e, f) for e, f in faces if e in connectivity] for n, faces in self.surfaces.items()},
                      self.cad_volumes)

    def element_volumes(self, elset: str) -> np.ndarray:
        """Volumes from the corner nodes (exact for straight-sided tets)."""
        p = self._corner_coords(self.elsets[elset])
        return np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                                np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6


def mesh_step(step_path: Path, bc_surfaces: dict[str, tuple[str | tuple[str, ...], BoxSelect]], size: float,
              elem_type: str = "C3D10", size_min: float | None = None,
              save_msh: Path | None = None) -> FEMesh:
    """Mesh step_path. bc_surfaces maps a set name to (region name or names, box selecting
    their boundary faces)."""
    gmsh_type, order = GMSH_TYPES[elem_type]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        imported = occ.importShapes(str(step_path))
        occ.synchronize()  # entity names from STEP labels are only visible after this
        names = [gmsh.model.getEntityName(*dt).split("/")[-1] for dt in imported]
        _, fragment_map = occ.fragment(imported, [])
        occ.synchronize()

        # Each fragment volume belongs to the highest-priority input solid that contains it.
        owner: dict[int, str] = {}
        rank = lambda n: REGION_PRIORITY.index(region_role(n))
        for name, pieces in zip(names, fragment_map):
            for dim, tag in pieces:
                if dim == 3 and (tag not in owner or rank(name) < rank(owner[tag])):
                    owner[tag] = name
        removed = [(3, t) for t, n in owner.items() if region_role(n) == "keep_out"]
        if removed:
            occ.remove(removed, recursive=True)
            occ.synchronize()
        regions: dict[str, list[int]] = {}
        for tag, name in owner.items():
            if region_role(name) != "keep_out":
                regions.setdefault(name, []).append(tag)
        cad_volumes = {n: sum(occ.getMass(3, t) for t in tags) for n, tags in regions.items()}
        for name, tags in regions.items():
            gmsh.model.addPhysicalGroup(3, tags, name=name)

        surface_tags = {}
        for set_name, (region, box) in bc_surfaces.items():
            names = (region,) if isinstance(region, str) else region
            boundary = gmsh.model.getBoundary([(3, t) for n in names for t in regions[n]], combined=False,
                                              oriented=False)
            tags = sorted({t for _, t in boundary if box.contains(gmsh.model.getBoundingBox(2, t))})
            if not tags:
                raise ValueError(f"no faces of {region!r} inside the box for {set_name!r}")
            surface_tags[set_name] = tags

        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_min if size_min is not None else size / 4)
        gmsh.option.setNumber("Mesh.ElementOrder", 2 if elem_type == "C3D10" else 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT: parallel Delaunay
        gmsh.model.mesh.generate(3)
        if save_msh:
            gmsh.write(str(save_msh))

        node_ids, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        n_nodes = len(order)
        elsets, connectivity = {}, {}
        for name, tags in regions.items():
            ids_all = []
            for tag in tags:
                types, elem_ids, elem_nodes = gmsh.model.mesh.getElements(3, tag)
                (i,) = [k for k, t in enumerate(types) if t == gmsh_type]
                conn = elem_nodes[i].reshape(-1, n_nodes)[:, order]
                connectivity.update(zip(elem_ids[i].tolist(), conn))
                ids_all.append(elem_ids[i])
            elsets[name] = np.concatenate(ids_all)

        # Map sorted corner-node triples to (element, face) to find element faces on a surface.
        face_of = {}
        for eid, conn in connectivity.items():
            for k, face in enumerate(TET_FACES):
                face_of[tuple(sorted(conn[list(face)]))] = (eid, k + 1)
        nsets, surfaces = {}, {}
        for set_name, tags in surface_tags.items():
            nodes, faces = set(), []
            for tag in tags:
                nodes.update(gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)[0].tolist())
                types, _, tri_nodes = gmsh.model.mesh.getElements(2, tag)
                per = gmsh.model.mesh.getElementProperties(types[0])[3]
                for tri in tri_nodes[0].reshape(-1, per)[:, :3]:
                    faces.append(face_of[tuple(sorted(tri))])
            nsets[set_name] = np.array(sorted(nodes))
            surfaces[set_name] = faces
    finally:
        gmsh.finalize()

    return FEMesh(elem_type, node_ids, coords, elsets, connectivity, nsets, surfaces, cad_volumes)
