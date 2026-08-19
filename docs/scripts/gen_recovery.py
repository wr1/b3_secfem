"""Unit-load recovery figures for the fumano site.

Run::

    micromamba run -n b3secfem python docs/scripts/gen_recovery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "public" / "figures" / "recovery"
sys.path.insert(0, str(EXAMPLES))

from _anba_runner_host import (  # noqa: E402
    _docker_image_ok,
    _local_anba_ok,
    _quads_to_tris,
    anba_section_from_arrays,
)
from _gx_vtu import write_gx_vtu  # noqa: E402
from validation._common import ASYM, rectangle  # noqa: E402
from validation.afmesh_naca import build_mesh  # noqa: E402
from validation.three_way import ensure_ccw, shear_center_from_K  # noqa: E402

from b3_secfem import (  # noqa: E402
    IsotropicMaterial,
    SectionInput,
    anba_to_secfem_input,
    from_gxbeam_vtu,
    plot_unit_load_fields,
    recover_unit_load_strains,
    solve,
)
from b3_secfem.viz import _extract_geometry  # noqa: E402

INK = "#5c6370"
STEEL = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0, name="iso")
C_TENSION, C_MASS, C_SHEAR = "#cc0000", "#dd6600", "#117733"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "axes.facecolor": "none",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
        }
    )


def _mat_row(mat) -> np.ndarray:
    if isinstance(mat, IsotropicMaterial):
        g = mat.E / (2 * (1 + mat.nu))
        return np.array([mat.E, mat.E, mat.E, g, g, g, mat.nu, mat.nu, mat.nu, mat.rho])
    return np.array(
        [
            mat.E1, mat.E2, mat.E3,
            mat.G12, mat.G13, mat.G23,
            mat.nu12, mat.nu13, mat.nu23, mat.rho,
        ]
    )


def _solve_secfem(coords, quads, mat, beta, alpha, work: Path, tag: str):
    n = quads.shape[0]
    if isinstance(mat, (list, tuple)):
        mats = list(mat)
        beta_arr = np.asarray(beta, float)
        alpha_arr = np.asarray(alpha, float)
        rows = np.stack([_mat_row(m) for m in mats])
    else:
        mats = [mat] * n
        beta_arr = np.full(n, float(beta))
        alpha_arr = np.full(n, float(alpha))
        rows = np.tile(_mat_row(mat), (n, 1))
    vtu = work / f"{tag}.vtu"
    write_gx_vtu(vtu, coords, quads, rows, np.zeros(n))
    info = from_gxbeam_vtu(vtu)
    inp = SectionInput(
        mesh_path=vtu,
        per_cell_material=mats,
        per_cell_beta_deg=beta_arr,
        per_cell_alpha_deg=alpha_arr,
        backend="fenicsx",
        linear_solver="lu",
        degree=2,
    )
    res = solve(inp)
    return res, recover_unit_load_strains(res)


def _mass_center_from_M(M: np.ndarray) -> tuple[float, float]:
    M = np.asarray(M, float)
    m = float(M[2, 2])
    if m <= 0.0 or not np.isfinite(m):
        return float("nan"), float("nan")
    return float(-M[2, 4] / m), float(M[2, 3] / m)


def _centres_secfem(res) -> dict[str, tuple[float, float]]:
    return {
        "tension": tuple(float(x) for x in res.tension_center),
        "mass": tuple(float(x) for x in res.mass_center),
        "shear": tuple(float(x) for x in res.shear_center),
    }


def _centres_anba(K: np.ndarray, M: np.ndarray) -> dict[str, tuple[float, float]]:
    return {
        "mass": _mass_center_from_M(M),
        "shear": shear_center_from_K(K),
    }


def _draw_centres(ax, centres: dict[str, tuple[float, float]], *, hollow: bool = False) -> list:
    handles = []
    spec = (
        ("tension", "o", C_TENSION, 8),
        ("mass", "v", C_MASS, 8),
        ("shear", "D", C_SHEAR, 8),
    )
    for key, mark, color, size in spec:
        if key not in centres:
            continue
        x, y = centres[key]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        kw = dict(marker=mark, markersize=size, linestyle="none", zorder=6)
        if hollow:
            kw.update(markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.6)
        else:
            kw.update(color=color, markeredgecolor="white", markeredgewidth=0.6)
        (h,) = ax.plot([x], [y], **kw)
        handles.append(h)
    return handles


def _moments(field: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(areas, dtype=float)
    tot = float(w.sum())
    mean = (field * w[:, None]).sum(axis=0) / tot
    rms = np.sqrt((field**2 * w[:, None]).sum(axis=0) / tot)
    return mean, rms


def _rel(a: float, b: float, floor: float) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), floor))


def _rel_caption(rel: float) -> str:
    if rel < 1e-10:
        return "machine precision"
    return f"{100 * rel:.2f} %"


def _anba_ok() -> bool:
    return _docker_image_ok("anba4:latest") or _local_anba_ok()


def _panel_cells(ax, verts, values, title: str, *centre_sets) -> None:
    vmax = float(np.percentile(np.abs(values), 98)) or 1.0
    coll = PolyCollection(
        verts,
        array=values,
        cmap="RdBu_r",
        norm=Normalize(-vmax, vmax),
        edgecolors="0.55",
        linewidths=0.15,
    )
    ax.add_collection(coll)
    for i, centres in enumerate(centre_sets):
        if centres:
            _draw_centres(ax, centres, hollow=i > 0)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title, color=INK, fontsize=10)
    ax.tick_params(labelsize=7)
    cb = ax.figure.colorbar(coll, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6, colors=INK)


def _panel_tris(ax, coords, tris, values, title: str, *centre_sets) -> None:
    _panel_cells(ax, coords[tris], values, title, *centre_sets)


def _panel_quads(ax, coords, quads, values, title: str, *centre_sets) -> None:
    _panel_cells(ax, coords[quads], values, title, *centre_sets)


def _centre_legend(fig, y: float = -0.02, *, anba: bool = True) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TENSION,
               markeredgecolor="white", markersize=8, label="tension"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=C_MASS,
               markeredgecolor="white", markersize=8, label="mass"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_SHEAR,
               markeredgecolor="white", markersize=8, label="shear"),
    ]
    if anba:
        handles.append(
            Line2D([0], [0], marker="D", color="none", markerfacecolor="none",
                   markeredgecolor=C_SHEAR, markeredgewidth=1.6, markersize=8,
                   label="ANBA (open)")
        )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=8)


def _naca_mesh() -> dict:
    """Campaign NACA 0018: ply 0° = spanwise → α = fiber = 90, β = tangent."""
    mesh = build_mesh()
    mesh["quads"] = ensure_ccw(mesh["coords"], mesh["quads"])
    ply = np.asarray(mesh["alpha_deg"], float)
    mesh["secfem_alpha"] = 90.0 - ply
    mesh["secfem_beta"] = np.asarray(mesh["beta_deg"], float)
    mesh["anba_fiber"] = 90.0 - ply
    mesh["anba_plane"] = np.asarray(mesh["beta_deg"], float)
    return mesh


def _write_naca(work: Path, manifest: dict) -> None:
    mesh = _naca_mesh()
    coords, quads = mesh["coords"], mesh["quads"]
    n = quads.shape[0]
    res, ul = _solve_secfem(
        coords, quads, mesh["mats"], mesh["secfem_beta"], mesh["secfem_alpha"],
        work, "naca0018",
    )
    c_sec = _centres_secfem(res)
    nodes, sec_quads = _extract_geometry(res.mesh)
    manifest["naca"] = {
        "n_quads": n,
        "secfem_shear": list(c_sec["shear"]),
        "secfem_mass": list(c_sec["mass"]),
        "secfem_tension": list(c_sec["tension"]),
        "K_Fz": float(res.K[2, 2]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.8))
    fig.patch.set_facecolor("none")
    for ax, k, lab in zip(axes, (2, 3, 4), ("Fz", "Mx", "My")):
        _panel_quads(ax, nodes, sec_quads, ul.sigma[k, :, 2], f"{lab}  σ_zz", c_sec)
    fig.suptitle("NACA 0018  ·  secfem Stage-1 σ_zz  ·  centres overlaid", color=INK, fontsize=11)
    _centre_legend(fig, y=-0.06, anba=False)
    fig.tight_layout()
    dest = OUT / "naca_sigma_stage1.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.8))
    fig.patch.set_facecolor("none")
    for ax, k, lab in zip(axes, (2, 3, 4), ("Fz", "Mx", "My")):
        _panel_quads(ax, nodes, sec_quads, ul.epsilon[k, :, 2], rf"{lab}  $\varepsilon_{{zz}}$", c_sec)
    fig.suptitle("NACA 0018  ·  secfem Stage-1 ε_zz  ·  centres overlaid", color=INK, fontsize=11)
    _centre_legend(fig, y=-0.06, anba=False)
    fig.tight_layout()
    dest = OUT / "naca_eps_stage1.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest)

    if not _anba_ok():
        manifest["naca"]["anba"] = {"skipped": True, "reason": "ANBA not available"}
        return

    names = [m.name for m in mesh["mats"]]
    uniq, ids, index = [], [], {}
    for m, name in zip(mesh["mats"], names, strict=True):
        if name not in index:
            index[name] = len(uniq)
            uniq.append(m)
        ids.append(index[name])
    anba = anba_section_from_arrays(
        coords, quads, uniq,
        fiber_orientation_deg=np.asarray(mesh["anba_fiber"], float),
        plane_orientation_deg=np.asarray(mesh["anba_plane"], float),
        material_id=np.asarray(ids, np.int64),
        recover_fields=True,
    )
    c_an = _centres_anba(anba["K"], anba["M"])
    kfz = float(anba["K"][2, 2])
    sm, sr = _moments(ul.sigma[2], ul.cell_areas)
    am, ar = _moments(anba["sigma"][2], anba["tri_area"])
    em, er = _moments(ul.epsilon[2], ul.cell_areas)
    aem, aer = _moments(anba["epsilon"][2], anba["tri_area"])
    _, sr_mx = _moments(ul.sigma[3], ul.cell_areas)
    _, ar_mx = _moments(anba["sigma"][3], anba["tri_area"])
    _, er_mx = _moments(ul.epsilon[3], ul.cell_areas)
    _, aer_mx = _moments(anba["epsilon"][3], anba["tri_area"])
    floor_s = 1e-6 * max(abs(sr[2]), abs(ar[2]), 1.0)
    floor_e = 1e-6 * max(abs(er[2]), abs(aer[2]), 1e-16)
    manifest["naca"]["anba"] = {
        "skipped": False,
        "K_Fz": kfz,
        "K_pd": bool(kfz > 0.0),
        "shear": list(c_an["shear"]),
        "mass": list(c_an["mass"]),
        "Fz_szz_mean_rel": _rel(sm[2], am[2], floor_s),
        "Mx_szz_rms_rel": _rel(sr_mx[2], ar_mx[2], floor_s),
        "Fz_ezz_mean_rel": _rel(em[2], aem[2], floor_e),
        "Mx_ezz_rms_rel": _rel(er_mx[2], aer_mx[2], floor_e),
    }

    tris = _quads_to_tris(quads)
    note = "" if kfz > 0.0 else "  ·  ANBA K[Fz] < 0 (known failed run)"
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.2))
    fig.patch.set_facecolor("none")
    for col, k, lab in zip(range(3), (2, 3, 4), ("Fz", "Mx", "My")):
        _panel_quads(axes[0, col], nodes, sec_quads, ul.sigma[k, :, 2],
                     f"secfem  {lab}  σ_zz", c_sec, c_an)
        _panel_tris(axes[1, col], coords, tris, anba["sigma"][k, :, 2],
                    f"ANBA  {lab}  σ_zz", c_sec, c_an)
    fig.suptitle(
        "NACA 0018  ·  Stage-1 σ_zz  ·  "
        f"Fz mean {_rel_caption(manifest['naca']['anba']['Fz_szz_mean_rel'])}  ·  "
        f"Mx rms {_rel_caption(manifest['naca']['anba']['Mx_szz_rms_rel'])}{note}",
        color=INK, fontsize=11,
    )
    _centre_legend(fig, y=-0.02)
    fig.tight_layout()
    dest = OUT / "naca_sigma_compare.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.2))
    fig.patch.set_facecolor("none")
    for col, k, lab in zip(range(3), (2, 3, 4), ("Fz", "Mx", "My")):
        _panel_quads(axes[0, col], nodes, sec_quads, ul.epsilon[k, :, 2],
                     rf"secfem  {lab}  $\varepsilon_{{zz}}$", c_sec, c_an)
        _panel_tris(axes[1, col], coords, tris, anba["epsilon"][k, :, 2],
                    rf"ANBA  {lab}  $\varepsilon_{{zz}}$", c_sec, c_an)
    fig.suptitle(
        "NACA 0018  ·  Stage-1 ε_zz  ·  "
        f"Fz mean {_rel_caption(manifest['naca']['anba']['Fz_ezz_mean_rel'])}  ·  "
        f"Mx rms {_rel_caption(manifest['naca']['anba']['Mx_ezz_rms_rel'])}{note}",
        color=INK, fontsize=11,
    )
    _centre_legend(fig, y=-0.02)
    fig.tight_layout()
    dest = OUT / "naca_eps_compare.png"
    fig.savefig(dest, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest)


def main() -> int:
    _style()
    import tempfile

    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="gen_recovery_"))
    coords, quads, area = rectangle(0.04, 0.01, 8, 4)
    n_quads = quads.shape[0]
    manifest: dict = {"area": area, "n_quads": n_quads}

    res, ul = _solve_secfem(coords, quads, STEEL, 0.0, 0.0, work, "iso")
    dest = OUT / "iso_unit_sigma.png"
    plot_unit_load_fields(
        res, ul, dest, title="iso rectangle  ·  unit-load σ (dominant component)"
    )
    print(dest)
    dest_e = OUT / "iso_unit_eps.png"
    plot_unit_load_fields(
        res,
        ul,
        dest_e,
        quantity="epsilon",
        title="iso rectangle  ·  unit-load ε (dominant component)",
    )
    print(dest_e)

    place = anba_to_secfem_input(ASYM, 90.0, 0.0)
    res_a, ul_a = _solve_secfem(
        coords, quads, place.material, place.beta_deg, place.alpha_deg, work, "asym_a90"
    )
    dest_a = OUT / "asym_a90_fz_szz.png"
    nodes_a, quads_a = _extract_geometry(res_a.mesh)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))
    fig.patch.set_facecolor("none")
    for ax, k, lab in zip(axes, (2, 3, 4), ("Fz", "Mx", "My")):
        _panel_quads(ax, nodes_a, quads_a, ul_a.sigma[k, :, 2], f"{lab}  σ_zz")
    fig.suptitle(
        "ASYM (β, α) = (0, 90)  ·  fibre on the beam  ·  Stage-1 σ_zz",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(dest_a, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest_a)

    sm_fz, sr_fz = _moments(ul.sigma[2], ul.cell_areas)
    sm_mx, sr_mx = _moments(ul.sigma[3], ul.cell_areas)
    em_fz, er_fz = _moments(ul.epsilon[2], ul.cell_areas)
    em_mx, er_mx = _moments(ul.epsilon[3], ul.cell_areas)
    em_my, er_my = _moments(ul.epsilon[4], ul.cell_areas)
    manifest["secfem_iso"] = {
        "Fz_mean_szz": float(sm_fz[2]),
        "Fz_rms_szz": float(sr_fz[2]),
        "Mx_mean_szz": float(sm_mx[2]),
        "Mx_rms_szz": float(sr_mx[2]),
        "Fz_mean_ezz": float(em_fz[2]),
        "Fz_rms_ezz": float(er_fz[2]),
        "Mx_mean_ezz": float(em_mx[2]),
        "Mx_rms_ezz": float(er_mx[2]),
        "My_rms_ezz": float(er_my[2]),
        "Fz_mean_exx": float(em_fz[0]),
        "expect_1_over_A": 1.0 / area,
    }

    if not _anba_ok():
        manifest["anba"] = {"skipped": True, "reason": "ANBA not available"}
        _write_naca(work, manifest)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("ANBA skipped — wrote secfem figures only")
        return 0

    anba = anba_section_from_arrays(
        coords,
        quads,
        STEEL,
        fiber_orientation_deg=np.zeros(n_quads),
        plane_orientation_deg=np.zeros(n_quads),
        recover_fields=True,
    )
    tris = _quads_to_tris(quads)
    am_fz, ar_fz = _moments(anba["sigma"][2], anba["tri_area"])
    am_mx, ar_mx = _moments(anba["sigma"][3], anba["tri_area"])
    aem_fz, aer_fz = _moments(anba["epsilon"][2], anba["tri_area"])
    aem_mx, aer_mx = _moments(anba["epsilon"][3], anba["tri_area"])
    aem_my, aer_my = _moments(anba["epsilon"][4], anba["tri_area"])
    floor_s = 1e-6 * max(abs(sr_mx[2]), abs(ar_mx[2]), 1.0)
    floor_e = 1e-6 * max(abs(er_mx[2]), abs(aer_mx[2]), 1e-16)
    manifest["anba"] = {
        "skipped": False,
        "Fz_mean_szz": float(am_fz[2]),
        "Fz_rms_szz": float(ar_fz[2]),
        "Mx_mean_szz": float(am_mx[2]),
        "Mx_rms_szz": float(ar_mx[2]),
        "Fz_mean_rel": _rel(sm_fz[2], am_fz[2], floor_s),
        "Mx_rms_rel": _rel(sr_mx[2], ar_mx[2], floor_s),
        "Fz_mean_ezz": float(aem_fz[2]),
        "Fz_rms_ezz": float(aer_fz[2]),
        "Mx_rms_ezz": float(aer_mx[2]),
        "My_rms_ezz": float(aer_my[2]),
        "Fz_mean_exx": float(aem_fz[0]),
        "Fz_ezz_mean_rel": _rel(em_fz[2], aem_fz[2], floor_e),
        "Mx_ezz_rms_rel": _rel(er_mx[2], aer_mx[2], floor_e),
        "My_ezz_rms_rel": _rel(er_my[2], aer_my[2], floor_e),
        "Fz_exx_mean_rel": _rel(em_fz[0], aem_fz[0], floor_e),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))
    fig.patch.set_facecolor("none")
    nodes, sec_quads = _extract_geometry(res.mesh)
    _panel_quads(axes[0], nodes, sec_quads, ul.sigma[3, :, 2], "secfem  σ_zz (Mx=1)")
    _panel_tris(axes[1], coords, tris, anba["sigma"][3, :, 2], "ANBA  σ_zz (Mx=1)")
    fig.suptitle(
        f"iso rectangle  ·  Mx σ_zz rms rel = {_rel_caption(manifest['anba']['Mx_rms_rel'])}",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    dest_c = OUT / "iso_mx_szz_compare.png"
    fig.savefig(dest_c, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest_c)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.4))
    fig.patch.set_facecolor("none")
    cases = (
        (2, "Fz", 2, r"$\varepsilon_{zz}$"),
        (3, "Mx", 2, r"$\varepsilon_{zz}$"),
        (4, "My", 2, r"$\varepsilon_{zz}$"),
    )
    for col, (k, lab, c, clab) in enumerate(cases):
        _panel_quads(
            axes[0, col], nodes, sec_quads, ul.epsilon[k, :, c], f"secfem  {lab}  {clab}"
        )
        _panel_tris(
            axes[1, col], coords, tris, anba["epsilon"][k, :, c], f"ANBA  {lab}  {clab}"
        )
    fig.suptitle(
        "iso rectangle  ·  Stage-1 strain  ·  "
        f"Fz ε_zz mean {_rel_caption(manifest['anba']['Fz_ezz_mean_rel'])}  ·  "
        f"Mx ε_zz rms {_rel_caption(manifest['anba']['Mx_ezz_rms_rel'])}",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    dest_eps = OUT / "iso_strain_compare.png"
    fig.savefig(dest_eps, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest_eps)

    # Poisson contraction under Fz: ε_xx = ε_yy = −ν ε_zz on a free iso section.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))
    fig.patch.set_facecolor("none")
    _panel_quads(axes[0], nodes, sec_quads, ul.epsilon[2, :, 0], r"secfem  Fz  $\varepsilon_{xx}$")
    _panel_tris(axes[1], coords, tris, anba["epsilon"][2, :, 0], r"ANBA  Fz  $\varepsilon_{xx}$")
    fig.suptitle(
        "iso rectangle  ·  Poisson ε_xx under Fz=1  ·  "
        f"mean {_rel_caption(manifest['anba']['Fz_exx_mean_rel'])}  ·  "
        f"ε_xx / ε_zz = {em_fz[0] / em_fz[2]:.2f}  (= −ν)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    dest_p = OUT / "iso_fz_exx_compare.png"
    fig.savefig(dest_p, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest_p)

    _write_naca(work, manifest)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(OUT / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
