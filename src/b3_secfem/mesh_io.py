"""One mesh reader and writer for both backends.

Reads a meshio single-grid XDMF, a dolfinx two-grid XDMF (mesh plus cell
tags), or a gxbeam VTU. Cells come back counter-clockwise, in file order.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from ._deprecation import warn_deprecated

CellType = Literal["triangle", "quad"]
TAG_KEY: Final = "cell_tags"
LEGACY_TAG_KEYS: Final = ("mesh_tags", "Cell tags", "tags", "region", "mat")

_QUAD = frozenset({"quad", "quadrilateral"})
_TRI = frozenset({"triangle", "tri"})
_2D = _QUAD | _TRI


@dataclass(frozen=True)
class SectionMesh:
    """Section mesh in file order, with counter-clockwise cells."""

    points: NDArray[np.float64]
    cells: NDArray[np.int64]
    cell_type: CellType
    cell_tags: NDArray[np.int32] | None
    cell_data: Mapping[str, NDArray[np.generic]]
    source: Path | None

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write a single-grid XDMF. Cell tags use the key ``cell_tags``."""
        return write_section_mesh(path, self)


def orient_ccw(
    points: NDArray[np.float64], cells: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Return cells wound counter-clockwise, and which input cells were flipped.

    This is the only shoelace in the package.
    """
    pts = np.asarray(points, dtype=np.float64)
    conn = np.array(cells, dtype=np.int64, copy=True)
    if conn.ndim != 2 or conn.shape[1] not in (3, 4):
        msg = f"cells must be (n, 3) or (n, 4), got {conn.shape}"
        raise ValueError(msg)
    p = pts[conn]
    if conn.shape[1] == 4:
        shoelace = (
            p[:, 0, 0] * p[:, 1, 1]
            - p[:, 1, 0] * p[:, 0, 1]
            + p[:, 1, 0] * p[:, 2, 1]
            - p[:, 2, 0] * p[:, 1, 1]
            + p[:, 2, 0] * p[:, 3, 1]
            - p[:, 3, 0] * p[:, 2, 1]
            + p[:, 3, 0] * p[:, 0, 1]
            - p[:, 0, 0] * p[:, 3, 1]
        )
        order = [0, 3, 2, 1]
    else:
        shoelace = (
            p[:, 0, 0] * p[:, 1, 1]
            - p[:, 1, 0] * p[:, 0, 1]
            + p[:, 1, 0] * p[:, 2, 1]
            - p[:, 2, 0] * p[:, 1, 1]
            + p[:, 2, 0] * p[:, 0, 1]
            - p[:, 0, 0] * p[:, 2, 1]
        )
        order = [0, 2, 1]
    flip = shoelace < 0
    if flip.any():
        conn[flip] = conn[flip][:, order]
    return conn, flip


def read_section_mesh(path: str | os.PathLike[str]) -> SectionMesh:
    """Read a section mesh. Relative paths are resolved from the process cwd."""
    src = Path(path)
    if not src.is_file():
        msg = f"mesh file does not exist: {src}"
        raise FileNotFoundError(msg)
    if src.suffix.lower() == ".xdmf" and _grid_count(src) > 1:
        mesh = _read_dolfinx_xdmf(src)
    else:
        mesh = _read_meshio(src)
    cells, _flip = orient_ccw(mesh.points, mesh.cells)
    return SectionMesh(
        points=mesh.points,
        cells=cells,
        cell_type=mesh.cell_type,
        cell_tags=mesh.cell_tags,
        cell_data=mesh.cell_data,
        source=src,
    )


def write_section_mesh(path: str | os.PathLike[str], mesh: SectionMesh) -> Path:
    """Write a single-grid meshio XDMF. Cell tags use the key ``cell_tags``."""
    import meshio

    dest = Path(path)
    cells, _flip = orient_ccw(mesh.points, mesh.cells)
    n = cells.shape[0]
    tags = None if mesh.cell_tags is None else _as_tags(mesh.cell_tags, n, "cell_tags")
    pts = np.zeros((mesh.points.shape[0], 3), dtype=np.float64)
    pts[:, :2] = np.asarray(mesh.points, dtype=np.float64)[:, :2]
    kind = "quad" if mesh.cell_type == "quad" else "triangle"
    cell_data = None
    if tags is not None:
        cell_data = {TAG_KEY: [tags]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        str(dest),
        meshio.Mesh(points=pts, cells=[(kind, cells)], cell_data=cell_data),
        file_format="xdmf",
    )
    return dest


def _grid_count(path: Path) -> int:
    root = ET.parse(path).getroot()
    domain = next((el for el in root.iter() if _local(el.tag) == "Domain"), None)
    if domain is None:
        return 0
    return sum(1 for el in list(domain) if _local(el.tag) == "Grid")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_meshio(path: Path) -> SectionMesh:
    import meshio

    loaded = meshio.read(str(path))
    blocks = [(i, block) for i, block in enumerate(loaded.cells) if block.type in _2D]
    if len(blocks) != 1:
        msg = (
            f"{path} has {len(blocks)} 2-D cell blocks; mixed meshes are not supported"
        )
        raise ValueError(msg)
    index, block = blocks[0]
    cell_type: CellType = "quad" if block.type in _QUAD else "triangle"
    cells = np.asarray(block.data, dtype=np.int64)
    points = np.asarray(loaded.points[:, :2], dtype=np.float64)
    cell_data: dict[str, NDArray[np.generic]] = {}
    for key, arrays in (loaded.cell_data or {}).items():
        cell_data[key] = np.asarray(arrays[index])
    tags = _tags_from_mapping(cell_data, cells.shape[0])
    return SectionMesh(points, cells, cell_type, tags, cell_data, path)


def _read_dolfinx_xdmf(path: Path) -> SectionMesh:
    root = ET.parse(path).getroot()
    domain = next(el for el in root.iter() if _local(el.tag) == "Domain")
    grids = [el for el in list(domain) if _local(el.tag) == "Grid"]
    mesh_grid = next(
        (g for g in grids if g.get("Name") in ("mesh", "Grid")),
        grids[0],
    )
    topo = next(el for el in mesh_grid if _local(el.tag) == "Topology")
    geom = next(el for el in mesh_grid if _local(el.tag) == "Geometry")
    cells = np.asarray(_hdf_array(path, _data_item(topo)), dtype=np.int64)
    points = np.asarray(_hdf_array(path, _data_item(geom)), dtype=np.float64)[:, :2]
    kind = (topo.get("TopologyType") or "").lower()
    if "quad" in kind:
        cell_type: CellType = "quad"
    elif "tri" in kind:
        cell_type = "triangle"
    else:
        msg = f"unsupported topology {topo.get('TopologyType')!r} in {path}"
        raise ValueError(msg)
    cell_data: dict[str, NDArray[np.generic]] = {}
    for grid in grids:
        if grid is mesh_grid:
            continue
        for attr in grid:
            if _local(attr.tag) != "Attribute":
                continue
            if attr.get("Center") not in (None, "Cell"):
                continue
            name = attr.get("Name") or TAG_KEY
            cell_data[name] = np.asarray(_hdf_array(path, _data_item(attr)))
    tags = _tags_from_mapping(cell_data, cells.shape[0])
    return SectionMesh(points, cells, cell_type, tags, cell_data, path)


def _data_item(parent: ET.Element) -> ET.Element:
    for el in parent.iter():
        if _local(el.tag) == "DataItem" and (el.text or "").strip():
            return el
    msg = "XDMF element has no DataItem"
    raise ValueError(msg)


def _hdf_array(xdmf: Path, item: ET.Element) -> NDArray[np.generic]:
    import h5py

    raw = (item.text or "").strip()
    if ":" not in raw:
        msg = f"unsupported DataItem {raw!r}"
        raise ValueError(msg)
    fname, internal = raw.split(":", 1)
    key = internal if internal.startswith("/") else f"/{internal}"
    h5_path = (xdmf.parent / fname).resolve()
    with h5py.File(h5_path, "r") as handle:
        return np.asarray(handle[key])


def _tags_from_mapping(
    cell_data: Mapping[str, NDArray[np.generic]], n_cells: int
) -> NDArray[np.int32] | None:
    if TAG_KEY in cell_data:
        return _as_tags(cell_data[TAG_KEY], n_cells, TAG_KEY)
    for key in LEGACY_TAG_KEYS:
        if key in cell_data:
            warn_deprecated(f"mesh cell-data key {key!r}", f"the key {TAG_KEY!r}")
            return _as_tags(cell_data[key], n_cells, key)
    return None


def _as_tags(values: NDArray[np.generic], n_cells: int, key: str) -> NDArray[np.int32]:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape != (n_cells,):
        msg = f"cell data {key!r} has length {arr.shape[0]} but the mesh has {n_cells} cells"
        raise ValueError(msg)
    return arr.astype(np.int32, copy=False)
