"""2D-section visualisation: mesh + region tags + centres + principal axes.

Renders a single PNG per section showing:

  - The quad mesh (grey edges, lightly filled).
  - Region tags as a discrete colour map (when more than one region is
    present in the SectionInput).
  - The tension, elastic, and shear centres as labelled markers.
  - Bending principal axes drawn from the elastic centre, scaled by the
    relative magnitude of the two principal bending stiffnesses.

The plot uses matplotlib only (no PyVista); fast to render and sufficient
for the 2D output the package targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import SectionInput
from .result import SectionResult


def plot_section(
    inp: SectionInput,
    res: SectionResult,
    out_path: str | Path,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 140,
) -> Path:
    """Render the section + overlay to ``out_path``. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.gridspec import GridSpec

    mesh, region_tags = _mesh_and_tags(res, inp)
    nodes, quads = _extract_geometry(mesh)

    bbox = _bbox(nodes)
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]
    aspect = width / height if height > 0 else 1.0
    if figsize is None:
        # Pick a figsize whose section panel preserves true aspect, with a
        # constant 4-inch-wide legend column on the right.
        sec_h = 4.0
        sec_w = max(2.5, min(8.0, sec_h * aspect))
        figsize = (sec_w + 4.0, sec_h + 0.6)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    sec_w, fig_h = figsize
    gs = GridSpec(
        nrows=1, ncols=2,
        width_ratios=[sec_w - 4.0, 4.0],
        wspace=0.25, figure=fig,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    poly_xy = nodes[quads]
    has_regions = region_tags is not None and len(np.unique(region_tags)) > 1
    if has_regions:
        coll = PolyCollection(
            poly_xy,
            array=region_tags.astype(float),
            cmap="viridis",
            edgecolor="0.4",
            linewidth=0.3,
        )
        cbar = fig.colorbar(coll, ax=ax, fraction=0.04, pad=0.02, label="region tag")
        cbar.set_ticks(np.unique(region_tags).astype(float))
    else:
        coll = PolyCollection(
            poly_xy, facecolor="0.85", edgecolor="0.4", linewidth=0.3,
        )
    ax.add_collection(coll)

    handles, labels = _draw_overlay(ax, res, nodes, bbox)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        fig.suptitle(title, fontsize=10)
    ax.autoscale_view()

    ax_legend.legend(
        handles, labels,
        loc="upper left", fontsize=8, framealpha=0.95,
        bbox_to_anchor=(0.0, 1.0),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_overlay(ax, res: SectionResult, nodes, bbox):
    """Draw centres + principal axes; return matplotlib (handles, labels)."""
    handles = []
    labels = []

    cx, cy = res.tension_center
    sx, sy = res.shear_center
    ex, ey = res.elastic_center

    if np.isfinite(cx) and np.isfinite(cy):
        h, = ax.plot([cx], [cy], "o", color="#cc0000", markersize=8,
                     markeredgecolor="white")
        handles.append(h)
        labels.append(f"tension ({cx:+.2e}, {cy:+.2e})")
    if np.isfinite(ex) and np.isfinite(ey):
        h, = ax.plot([ex], [ey], "s", color="#0044aa", markersize=8,
                     markeredgecolor="white")
        handles.append(h)
        labels.append(f"elastic ({ex:+.2e}, {ey:+.2e})")
    if np.isfinite(sx) and np.isfinite(sy):
        h, = ax.plot([sx], [sy], "D", color="#117733", markersize=8,
                     markeredgecolor="white")
        handles.append(h)
        labels.append(f"shear   ({sx:+.2e}, {sy:+.2e})")

    bending_block = res.K[3:5, 3:5]
    if np.all(np.isfinite(bending_block)):
        eigvals, eigvecs = np.linalg.eigh(bending_block)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        L = 0.45 * max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        mag = eigvals / eigvals.max()
        cx_e, cy_e = (ex, ey) if np.isfinite(ex) else (0.0, 0.0)
        for k, (lam, vec) in enumerate(zip(eigvals, eigvecs.T)):
            ex_dir = np.array([vec[1], -vec[0]])
            ex_dir /= np.linalg.norm(ex_dir)
            half = 0.5 * L * mag[k]
            x0, y0 = cx_e - half * ex_dir[0], cy_e - half * ex_dir[1]
            x1, y1 = cx_e + half * ex_dir[0], cy_e + half * ex_dir[1]
            color = "#222222" if k == 0 else "#666666"
            lw = 2.0 if k == 0 else 1.4
            h, = ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw)
            handles.append(h)
            labels.append(f"principal axis {k + 1}\n  EI = {lam:.2e}")
    return handles, labels


def _mesh_and_tags(res: SectionResult, inp: SectionInput):
    if res.mesh is None:
        msg = "SectionResult has no mesh attached"
        raise ValueError(msg)
    mesh = res.mesh
    tags = None
    if inp.region_materials is not None:
        from .mesh import read_xdmf

        try:
            _, mt = read_xdmf(inp.mesh_path)
            if mt is not None:
                n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
                first_tag = next(iter(inp.region_materials))
                arr = np.full(n_cells, first_tag, dtype=np.int32)
                arr[mt.indices] = mt.values
                tags = arr
        except Exception:
            tags = None
    return mesh, tags


def _extract_geometry(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes_2d, quads_4node_CCW) reconstructed from the dolfinx mesh.

    dolfinx 0.10 uses tensor-product quad ordering (BL, TL, BR, TR); we
    permute to CCW (BL, BR, TR, TL) for matplotlib's PolyCollection.
    """
    nodes = np.asarray(mesh.geometry.x[:, :2], dtype=float)
    cell_dim = mesh.topology.dim
    n_cells = mesh.topology.index_map(cell_dim).size_local
    dofmap = mesh.geometry.dofmap[:n_cells]
    if dofmap.shape[1] != 4:
        msg = f"plot_section currently supports quad meshes only; got {dofmap.shape[1]}-node cells"
        raise NotImplementedError(msg)
    return nodes, dofmap[:, [0, 2, 3, 1]]


def plot_warping(
    res: SectionResult,
    mode: int | str,
    out_path: str | Path,
    scale: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: int = 140,
) -> Path:
    """Render the section in its deformed shape under a unit-load mode.

    Parameters
    ----------
    res : SectionResult
        Result of ``solve(inp)``; must carry ``u_solutions`` (and
        ``inplane_shear_warping`` for ``mode="xy"``).
    mode : int or "xy"
        0..5 picks one of the chain modes (Vx, Vy, Fz, Mx, My, Mz). For
        modes 2..5 the displayed deformation is the warping field
        ``d_1`` (the assumed kinematic ``d_0`` is subtracted, so we
        see only the FE-computed correction). For modes 0..1 (shear)
        it is ``d_2``. ``mode="xy"`` shows the in-plane shear warping
        from the 7th cell problem with the assumed (0.5y, 0.5x, 0)
        background ADDED so the section visibly shears.
    scale : float, optional
        Displacement scale factor. Default chosen automatically so the
        peak in-plane displacement is ~10 % of the section bbox diagonal.

    Side-by-side plot of:

      - Original undeformed mesh outline (light grey).
      - Deformed mesh, cells coloured by the out-of-plane warping
        component ``u_z``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if res.mesh is None:
        msg = "result has no mesh attached; cannot plot warping"
        raise ValueError(msg)
    mesh = res.mesh
    nodes, quads = _extract_geometry(mesh)
    bbox = _bbox(nodes)
    diag = np.hypot(bbox[1] - bbox[0], bbox[3] - bbox[2])

    disp_nodes, mode_label, include_assumed = _evaluate_mode_at_nodes(
        res, mode, nodes
    )
    if scale is None:
        peak_inplane = float(np.max(np.linalg.norm(disp_nodes[:, :2], axis=1)))
        if peak_inplane > 1e-30:
            scale = 0.10 * diag / peak_inplane
        else:
            scale = 1.0

    deformed = nodes + scale * disp_nodes[:, :2]
    u_z = disp_nodes[:, 2]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    poly_orig = nodes[quads]
    ax.add_collection(PolyCollection(
        poly_orig, facecolor="none", edgecolor="0.7", linewidth=0.4,
    ))

    poly_def = deformed[quads]
    cell_uz = u_z[quads].mean(axis=1)
    coll = PolyCollection(
        poly_def, array=cell_uz, cmap="coolwarm",
        edgecolor="0.3", linewidth=0.3,
    )
    ax.add_collection(coll)
    cbar = fig.colorbar(coll, ax=ax, fraction=0.04, pad=0.02,
                        label="$u_z$ (out-of-plane warping) [m]")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.autoscale_view()
    sub = f"({mode_label}, in-plane displacement scaled x{scale:.2g})"
    if title:
        ax.set_title(f"{title}  {sub}", fontsize=10)
    else:
        ax.set_title(sub, fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _evaluate_mode_at_nodes(
    res: SectionResult, mode: int | str, nodes: np.ndarray
) -> tuple[np.ndarray, str, bool]:
    """Evaluate the displacement field for ``mode`` at the geometry nodes.

    Returns ``(disp, label, include_assumed)`` where disp is shape
    ``(n_nodes, 3)`` with a visualisation-friendly displacement field:

      * Stage-1 modes (2..5): warping ``d_1`` plus the rigid kinematic
        ``d_0`` (evaluated at z = 1 so the section visibly extends /
        bends / twists, not just warps a microscopic amount).
      * Stage-2 shear modes (0, 1): warping ``d_2`` only — the bending
        kinematic at z = 0 is zero anyway.
      * "xy" in-plane shear: warping ``w_xy`` plus the assumed
        ``(0.5 y, 0.5 x, 0)`` so the section visibly shears.
    """
    from dolfinx.geometry import (
        bb_tree, compute_colliding_cells, compute_collisions_points,
    )

    mesh = res.mesh
    n_nodes = nodes.shape[0]
    points_3d = np.column_stack([nodes, np.zeros(n_nodes)])

    tree = bb_tree(mesh, mesh.topology.dim)
    cand = compute_collisions_points(tree, points_3d)
    coll = compute_colliding_cells(mesh, cand, points_3d)
    cells_per_pt = np.array(
        [coll.links(k)[0] if len(coll.links(k)) else -1 for k in range(n_nodes)],
        dtype=np.int32,
    )

    if mode == "xy":
        if res.inplane_shear_warping is None:
            msg = "result has no inplane_shear_warping field"
            raise ValueError(msg)
        warp = res.inplane_shear_warping.eval(points_3d, cells_per_pt)
        assumed = np.zeros_like(warp)
        assumed[:, 0] = 0.5 * nodes[:, 1]
        assumed[:, 1] = 0.5 * nodes[:, 0]
        return warp + assumed, "in-plane shear $\\gamma_{xy}=1$", True

    if not isinstance(mode, int) or not 0 <= mode <= 5:
        msg = f"mode must be 0..5 or 'xy'; got {mode!r}"
        raise ValueError(msg)
    if res.u_solutions is None or res.u_solutions[mode] is None:
        msg = f"u_solutions[{mode}] missing"
        raise ValueError(msg)

    warp = res.u_solutions[mode].eval(points_3d, cells_per_pt)
    d0 = np.zeros_like(warp)
    if mode == 2:           # axial
        d0[:, 2] = 1.0
    elif mode == 3:         # M_x bending
        d0[:, 2] = -nodes[:, 1]
    elif mode == 4:         # M_y bending
        d0[:, 2] = nodes[:, 0]
    elif mode == 5:         # M_z torsion
        d0[:, 0] = -nodes[:, 1]
        d0[:, 1] = nodes[:, 0]
    # Stage-2 shear modes get no kinematic at z = 0; warp is already d_2.

    labels = [
        "$V_x$ shear (mode 0)",
        "$V_y$ shear (mode 1)",
        "$F_z$ axial (mode 2)",
        "$M_x$ bending (mode 3)",
        "$M_y$ bending (mode 4)",
        "$M_z$ torsion (mode 5)",
    ]
    return warp + d0, labels[mode], False


def _bbox(nodes: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(nodes[:, 0].min()),
        float(nodes[:, 0].max()),
        float(nodes[:, 1].min()),
        float(nodes[:, 1].max()),
    )
