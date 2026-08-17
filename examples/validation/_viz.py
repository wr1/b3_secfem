"""Comparison figures — color is *mismatch source*, not solver name.

Reads the JSON the drivers already write. Nothing here changes K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from validation._common import (
    ASYM,
    FORCE_ORDER_SECFEM,
    OUT,
    from_gxbeam_order,
    principal_axes,
    rel_per_dof,
)

LABELS = list(FORCE_ORDER_SECFEM)

# One language for every figure. Color = diagnosed source, never "which bar".
SRC = {
    "agree": "#7f8c8d",
    "axis23": "#c0392b",
    "axis23_z": "#922b21",
    "adapter": "#6c3483",
    "alpha": "#e67e22",
    "gx_shear": "#1a5276",
    "anba_gap": "#0e6655",
    "invalid": "#d5d8dc",
}
SRC_LABEL = {
    "agree": "agree  (<1%)",
    "axis23": "2↔3 assignment  (tangent vs thickness)",
    "axis23_z": "2↔3 on the beam  (E2 vs E3 on z at α=90)",
    "adapter": "2↔3 card bridge  (b3_section → gxbeam)",
    "alpha": "mid-α rotation vs gxbeam  (not 2/3)",
    "gx_shear": "in-plane residual vs gxbeam  (EA matches)",
    "anba_gap": "residual vs ANBA  (not a 2/3 signature)",
    "invalid": "not comparable / not run",
}
C_AX = {1: "#922b21", 2: "#1a5276", 3: "#c0392b"}
AGREE = 0.01


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "legend.framealpha": 0.95,
        }
    )
    return plt


def _save(fig, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    _plt().close(fig)
    return dest


def _source_legend(fig, *, loc="lower center", ncol=3, y=-0.02):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=SRC[k], edgecolor="0.3", label=SRC_LABEL[k]) for k in SRC]
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------------------
# classify every comparison we have
# ---------------------------------------------------------------------------


@dataclass
class Cmp:
    row: str
    rel: np.ndarray
    source: list[str]
    note: str = ""
    group: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _fill(rel: np.ndarray, per_dof: dict[str, str], default: str) -> list[str]:
    out = []
    for i, lab in enumerate(LABELS):
        if rel[i] < AGREE:
            out.append("agree")
        else:
            out.append(per_dof.get(lab, default))
    return out


def _gx_diags(rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "K_diag_secfem" in rec and "K_diag_gx_secfem_order" in rec:
        return (
            np.asarray(rec["K_diag_secfem"], float),
            np.asarray(rec["K_diag_gx_secfem_order"], float),
        )
    us = np.diag(np.asarray(rec["secfem_fenicsx"]["K_secfem_order"], float))
    gx = np.diag(from_gxbeam_order(np.asarray(rec["K_gx"], float)))
    return us, gx


def collect_comparisons(
    two3: dict[str, Any] | None,
    sonata: list[dict[str, Any]] | None,
    gx_raw: dict[str, Any] | None,
    gx_swap: dict[str, Any] | None,
    afmesh: dict[str, Any] | None,
) -> list[Cmp]:
    rows: list[Cmp] = []

    def _vs_anba(label: str, us_diag, an_diag, *, alpha: float = 0.0) -> Cmp:
        rel = rel_per_dof(us_diag, an_diag)
        if float(np.nanmax(np.nan_to_num(rel, nan=0.0))) < AGREE:
            src = ["agree"] * 6
            group = "agree"
            note = "secfem matches ANBA"
        elif abs(alpha - 90.0) < 1 and rel[2] >= AGREE:
            src = _fill(rel, {"Fz": "axis23_z", "Mx": "axis23_z", "My": "axis23_z"}, "axis23")
            group = "axis23_z"
            note = "EA/bending disagree — 2/3 on the beam vs ANBA"
        elif abs(alpha - 45.0) < 1:
            src = _fill(rel, {}, "alpha")
            group = "alpha"
            note = "mid-α residual vs ANBA"
        elif (
            rel[2] < AGREE
            and rel[3] < AGREE
            and rel[4] < AGREE
            and (rel[0] >= AGREE or rel[5] >= AGREE)
        ):
            src = _fill(rel, {}, "axis23")
            group = "axis23"
            note = "fibre terms match; Fx/Mz differ — 2/3 vs ANBA"
        elif rel[2] < AGREE and rel[3] < AGREE and rel[4] < AGREE and rel[1] >= AGREE:
            src = _fill(rel, {}, "anba_gap")
            group = "anba_gap"
            note = "Fy-only residual vs ANBA (E2=E3 cases too — not 2/3)"
        else:
            src = _fill(rel, {}, "anba_gap")
            group = "anba_gap"
            note = "ANBA residual, not the 2/3 signature"
        return Cmp(label, rel, src, note, group)

    # --- ANBA first ---
    if two3:
        fx = {
            r["mapping"]: r
            for r in two3.get("runs", [])
            if r.get("backend") == "fenicsx" and "K_diag" in r
        }
        an = {
            r["mapping"]: r
            for r in two3.get("runs", [])
            if r.get("backend") == "anba" and r.get("K_diag") and not r.get("skipped")
        }
        for name, a in an.items():
            if name not in fx:
                continue
            alpha = 90.0 if "a90" in name else 0.0
            rows.append(
                _vs_anba(
                    f"secfem vs ANBA   {name.replace('secfem_', '')}",
                    fx[name]["K_diag"],
                    a["K_diag"],
                    alpha=alpha,
                )
            )

    if sonata:
        for block in sonata:
            for rec in block.get("cases", []):
                an = rec.get("anba") or {}
                be = rec["backends"].get("fenicsx")
                if be and an and not an.get("skipped") and "K_diag" in an:
                    rows.append(
                        _vs_anba(
                            f"secfem vs ANBA   {rec['name']}",
                            be["1to1"]["K_diag"],
                            an["K_diag"],
                        )
                    )

    if gx_raw:
        for rec in gx_raw.get("cases", []):
            an = rec.get("anba") or {}
            if an.get("skipped") or "K_diag_secfem" not in an:
                continue
            us = rec.get("K_diag_secfem")
            if not us:
                us = np.diag(np.asarray(rec["secfem_fenicsx"]["K_secfem_order"]))
            rows.append(
                _vs_anba(
                    f"secfem vs ANBA   {rec['name']}",
                    us,
                    an["K_diag_secfem"],
                    alpha=float(rec.get("alpha_deg", 0.0)),
                )
            )

    if afmesh and afmesh.get("secfem_1to1") and (afmesh.get("anba") or {}).get("K_diag"):
        if not afmesh["anba"].get("skipped"):
            rows.append(
                _vs_anba(
                    "secfem vs ANBA   NACA 0018",
                    afmesh["secfem_1to1"]["K_diag"],
                    afmesh["anba"]["K_diag"],
                )
            )

    if two3:
        fx = {
            r["mapping"]: r
            for r in two3.get("runs", [])
            if r.get("backend") == "fenicsx" and "K_diag" in r
        }
        mf = {
            r["mapping"]: r
            for r in two3.get("runs", [])
            if r.get("backend") == "mfem" and "K_diag" in r
        }
        rels = []
        for name, a in fx.items():
            if name in mf:
                rels.append(rel_per_dof(a["K_diag"], mf[name]["K_diag"]))
        if rels:
            rel = np.max(np.vstack(rels), axis=0)
            rows.append(
                Cmp(
                    "fenicsx vs mfem   (all 5 mappings)",
                    rel,
                    ["agree"] * 6,
                    "max rel 8e-15 — not a source",
                    "backend",
                )
            )
        if "secfem_1to1_a0" in fx and "secfem_swap23_a0" in fx:
            rel = rel_per_dof(fx["secfem_1to1_a0"]["K_diag"], fx["secfem_swap23_a0"]["K_diag"])
            rows.append(
                Cmp(
                    "1:1 vs swap23   rectangle α=0   ASYM",
                    rel,
                    _fill(rel, {}, "axis23"),
                    "Fz/Mx/My stay (fibre). Fx/Fy/Mz move.",
                    "axis23",
                )
            )
        if "secfem_1to1_a0" in fx and "secfem_1to1_b90_a0" in fx:
            rel = rel_per_dof(fx["secfem_1to1_a0"]["K_diag"], fx["secfem_1to1_b90_a0"]["K_diag"])
            rows.append(
                Cmp(
                    "1:1 vs β=90     rectangle α=0   ASYM",
                    rel,
                    _fill(rel, {}, "axis23"),
                    "identical to the swap23 row (β=90 ≡ 2↔3)",
                    "axis23",
                )
            )
        if "secfem_1to1_a90" in fx and "secfem_swap23_a90" in fx:
            rel = rel_per_dof(fx["secfem_1to1_a90"]["K_diag"], fx["secfem_swap23_a90"]["K_diag"])
            rows.append(
                Cmp(
                    "1:1 vs swap23   rectangle α=90  ASYM",
                    rel,
                    _fill(rel, {"Fz": "axis23_z", "Mx": "axis23_z", "My": "axis23_z"}, "axis23"),
                    "EA/bending = E2 vs E3 on z (75%). Shear still 2↔3.",
                    "axis23_z",
                )
            )

    if sonata:
        cases = []
        for block in sonata:
            cases.extend(block.get("cases", []))
        for rec in cases:
            be = rec["backends"].get("fenicsx") or next(iter(rec["backends"].values()))
            rel = rel_per_dof(be["1to1"]["K_diag"], be["swap23"]["K_diag"])
            e2eq = bool(rec.get("e2_eq_e3"))
            rows.append(
                Cmp(
                    f"1:1 vs swap23   {rec['name']}",
                    rel,
                    ["agree"] * 6 if e2eq else _fill(rel, {}, "axis23"),
                    "E2=E3 → swap is a no-op" if e2eq else "same 2↔3 signature as the rectangle",
                    "axis23",
                )
            )

    if afmesh and "secfem_1to1" in afmesh:
        rel = rel_per_dof(afmesh["secfem_1to1"]["K_diag"], afmesh["secfem_swap23"]["K_diag"])
        rows.append(
            Cmp(
                "1:1 vs swap23   NACA 0018 skin/web",
                rel,
                _fill(rel, {}, "axis23"),
                "caps E2=E3 hold EA/bending; biax skin/web move shear/torsion",
                "axis23",
            )
        )

    def _add_gx(tag: str, rec: dict[str, Any], *, swapped: bool) -> None:
        us, gx = _gx_diags(rec)
        rel = rel_per_dof(us, gx)
        name = rec["name"]
        alpha = rec.get("alpha_deg", 0.0)
        if name.startswith("iso") or (name.startswith("ud_") and abs(alpha) < 1):
            src = _fill(rel, {}, "agree")
            note = "E2=E3 or isotropic — 2/3 invisible"
            default = "agree"
        elif "t45" in name or abs(alpha - 45.0) < 1:
            src = _fill(rel, {}, "alpha")
            note = "identical with or without the bridge — not 2/3"
            default = "alpha"
        elif swapped and abs(alpha - 90.0) < 1:
            src = _fill(rel, {}, "gx_shear")
            note = "bridge puts E3 on z (= secfem); shear residual unchanged"
            default = "adapter"
        elif abs(alpha - 90.0) < 1:
            src = _fill(rel, {"Fz": "axis23_z", "Mx": "axis23_z", "My": "axis23_z"}, "gx_shear")
            note = "raw gxbeam holds E2 on z; secfem/ANBA have E3"
            default = "axis23_z"
        else:
            src = _fill(rel, {}, "gx_shear")
            note = "EA matches. Error follows 2/3 onto Fx vs Fy under swap."
            default = "gx_shear"
        label = f"secfem vs gxbeam   {name}   {'adapter' if swapped else 'raw'}"
        rows.append(Cmp(label, rel, src, note, default))

    if gx_raw:
        for rec in gx_raw.get("cases", []):
            _add_gx(rec["name"], rec, swapped=False)
    if gx_swap:
        # only the discriminating adapter rows — iso/UD-t0/t45 are identical
        for rec in gx_swap.get("cases", []):
            if rec["name"].startswith("asym"):
                _add_gx(rec["name"], rec, swapped=True)

    if afmesh and afmesh.get("gxbeam") and not afmesh["gxbeam"].get("skipped"):
        rows.append(
            Cmp(
                "secfem vs gxbeam   NACA (no β)",
                np.full(6, np.nan),
                ["invalid"] * 6,
                "gxbeam θ=α only; returned K all negative",
                "invalid",
            )
        )
    rows.append(
        Cmp(
            "live SONATA dump",
            np.full(6, np.nan),
            ["invalid"] * 6,
            "no SONATA env; sonata_dump.py exits 2",
            "invalid",
        )
    )
    return rows


# ---------------------------------------------------------------------------
# axes (convention, not a comparison)
# ---------------------------------------------------------------------------


def plot_axes_flange(dest: Path | None = None) -> Path:
    plt = _plt()
    dest = dest or (OUT / "two3_axes.png")
    cases = (
        (0.0, 0.0, "default UD   (β,α)=(0,0)"),
        (90.0, 0.0, "plane roll   (β,α)=(90,0)  ≡ 2↔3"),
        (0.0, 90.0, "fibre in plane   (β,α)=(0,90)  → EA uses E3"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    half = np.array([0.020, 0.005])
    names = {1: "1 fibre", 2: "2 thickness", 3: "3 in-plane"}
    for ax, (beta, alpha, title) in zip(axes, cases, strict=True):
        ax.add_patch(
            plt.Rectangle(
                -half, 2 * half[0], 2 * half[1],
                facecolor="#f4f6f7", edgecolor="0.25", lw=1.0, zorder=0,
            )
        )
        rec = principal_axes(beta, alpha)
        z_notes = []
        for i, key in enumerate(("axis1_fibre", "axis2", "axis3"), start=1):
            v = np.asarray(rec[key], dtype=float)
            xy, nxy = v[:2], float(np.linalg.norm(v[:2]))
            if nxy > 0.15:
                ax.annotate(
                    "",
                    xy=0.014 * xy / nxy,
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=C_AX[i], lw=2.0, mutation_scale=12),
                )
            else:
                ax.plot(0, 0, "o", ms=7, color=C_AX[i], zorder=5)
                z_notes.append((i, "out +z" if v[2] > 0 else "in −z"))
        if z_notes:
            ax.text(
                0.0, -0.012,
                "\n".join(f"{names[i]}  {n}" for i, n in z_notes),
                color="0.15", fontsize=8, ha="center", va="top",
            )
        ax.set_xlim(-0.028, 0.028)
        ax.set_ylim(-0.020, 0.018)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(title, pad=8)
        if ax is axes[0]:
            ax.set_ylabel("y (m)")
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[Line2D([0], [0], color=C_AX[i], lw=2.2, label=names[i]) for i in (1, 2, 3)],
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("What 1 / 2 / 3 mean on a y-thin flange", fontsize=12, y=1.02)
    return _save(fig, dest)


# ---------------------------------------------------------------------------
# mismatch map (hero)
# ---------------------------------------------------------------------------


def plot_mismatch_map(rows: list[Cmp], dest: Path | None = None) -> Path:
    plt = _plt()
    dest = dest or (OUT / "mismatch_map.png")
    n = len(rows)
    fig_h = 0.42 * n + 3.2
    fig, ax = plt.subplots(figsize=(11.4, fig_h))
    for i, cmp_ in enumerate(rows):
        for j in range(6):
            src = cmp_.source[j]
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SRC[src], edgecolor="white", lw=1.2)
            )
            if src == "invalid" or not np.isfinite(cmp_.rel[j]):
                txt = "n/a"
            elif cmp_.rel[j] < AGREE:
                txt = "·"
            else:
                txt = f"{100 * cmp_.rel[j]:.0f}%"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=8,
                color="0.15" if src in {"agree", "invalid"} else "white",
            )
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(6), LABELS)
    ax.set_yticks(range(n), [r.row for r in rows], fontsize=8)
    ax.grid(False)
    ax.set_title("Every comparison, every DOF   ·   colour = source   ·   number = rel %")
    fig.subplots_adjust(left=0.40, right=0.98, top=0.93, bottom=0.20)
    _source_legend(fig, ncol=2, y=0.02)
    return _save(fig, dest)


def plot_source_table(rows: list[Cmp], dest: Path | None = None) -> Path:
    """One-page conclusions: what each source is, what closes it."""
    plt = _plt()
    dest = dest or (OUT / "mismatch_sources.png")
    del rows  # table is the catalog; the map carries the instances
    lines = [
        ("agree", "Control cases",
         "fenicsx = mfem; iso ring; UD α=0; Smith–Chopra (E2=E3);\nfibre terms Fz/Mx/My at α=0.",
         "Nothing to close. These prove the engines match and that\n2/3 is invisible when E2=E3."),
        ("axis23", "2↔3 assignment",
         "1:1 vs swap23 at α=0: Fx/Fy/Mz only. β=90 ≡ swap23.\nSONATA glass_* and NACA skin/web.",
         "secfem keeps the committed convention (α=0: 2=thickness,\n3=tangent) — same as ANBA. A card swap breaks the match."),
        ("axis23_z", "2↔3 on the beam",
         "α=90 1:1 vs swap23: EA/bending = 75% (E2 vs E3 on z).",
         "Contract: α=90 → EA = E3·A, same as ANBA.\nPinned in tests/test_rotation3d.py."),
        ("adapter", "2↔3 card bridge (gxbeam)",
         "gxbeam --gx-swap23 at α=90 brings gxbeam EA onto E3·A.\nShear residual is unchanged.",
         "Bridge belongs on the gxbeam side only;\nsecfem itself needs no swap."),
        ("alpha", "Mid-α vs gxbeam",
         "UD ellipse α=45: 8–38% on every DOF, vs gxbeam and ANBA.\nThe 2↔3 bridge does nothing.",
         "Open. Fibre-tilt / rotation-axis map vs gxbeam.\nNot a 2/3 bug."),
        ("gx_shear", "In-plane residual vs gxbeam",
         "Asym ring/ellipse: EA matches, Fx/Fy/Mz stay at 6–37%.\nAdapter only swaps which of Fx/Fy is worse.",
         "Open. gxbeam in-plane shear when E2≠E3.\nNot affected by any 2↔3 mapping."),
        ("anba_gap", "Residual vs ANBA",
         "Fy-only on thin sections even when E2=E3 (Smith–Chopra 68%,\nUD ellipse 24%). NACA β-map not trusted.",
         "Open. Not 2/3. Thin-section transverse shear vs ANBA;\nfoil needs a checked plane-angle map."),
        ("invalid", "Not comparable",
         "Live SONATA not run.\nNACA gxbeam (no β, K all negative).",
         "Need a SONATA env.\ngxbeam has no β for a wrapped skin."),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Mismatch sources — what they are, what closes them", pad=8, loc="left")
    y = 0.96
    row_h = 0.125
    for key, title, where, close in lines:
        y -= row_h
        ax.add_patch(plt.Rectangle((0.00, y), 0.018, row_h - 0.015, facecolor=SRC[key], edgecolor="none"))
        ax.text(0.035, y + row_h - 0.038, title, fontsize=10, fontweight="bold", va="top")
        ax.text(0.035, y + 0.012, where, fontsize=8, va="bottom", color="0.15")
        ax.text(0.55, y + 0.012, close, fontsize=8, va="bottom", color="0.15")
    ax.text(0.035, 0.015, "Shows up in", fontsize=7.5, color="0.45")
    ax.text(0.55, 0.015, "Close it by", fontsize=7.5, color="0.45")
    return _save(fig, dest)


# ---------------------------------------------------------------------------
# supporting: EA landing (one comparison, one question)
# ---------------------------------------------------------------------------


def plot_ea_landing(two3: dict[str, Any], dest: Path | None = None) -> Path | None:
    plt = _plt()
    dest = dest or (OUT / "two3_ea.png")
    runs = [
        r
        for r in two3.get("runs", [])
        if r.get("backend") == "fenicsx" and "EA_over_A" in r
    ]
    if not runs:
        return None
    labels = []
    vals = []
    cols = []
    for r in runs:
        name = (
            r["mapping"]
            .replace("secfem_", "")
            .replace("1to1_", "1:1  ")
            .replace("swap23_", "swap23  ")
            .replace("b90_a0", "β=90")
        )
        labels.append(name)
        vals.append(r["EA_over_A"] / 1e9)
        closest = r.get("closest_E", {}).get("closest", "")
        cols.append({"E1": SRC["agree"], "E2": SRC["axis23_z"], "E3": SRC["adapter"]}.get(closest, "0.5"))
        # swap23 a90 lands on mapping-E2 = original E3
        if r["mapping"] == "secfem_swap23_a90":
            cols[-1] = SRC["adapter"]
        elif r["mapping"] == "secfem_1to1_a90":
            cols[-1] = SRC["axis23_z"]
        elif closest == "E1":
            cols[-1] = SRC["agree"]
    an_by = {
        r["mapping"]: r["EA_over_A"] / 1e9
        for r in two3.get("runs", [])
        if r.get("backend") == "anba" and "EA_over_A" in r and not r.get("skipped")
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="0.2", lw=0.4, label="secfem")
    xs, ys = [], []
    for i, r in enumerate(runs):
        if r["mapping"] in an_by:
            xs.append(i)
            ys.append(an_by[r["mapping"]])
    if xs:
        ax.plot(xs, ys, "D", color="#0e6655", ms=8, label="ANBA", zorder=5)
    ax.axhline(ASYM.E1 / 1e9, color="0.25", ls="--", lw=0.9)
    ax.axhline(ASYM.E2 / 1e9, color=SRC["axis23_z"], ls="--", lw=0.9)
    ax.axhline(ASYM.E3 / 1e9, color=SRC["adapter"], ls="--", lw=0.9)
    ax.text(len(vals) - 0.45, ASYM.E1 / 1e9 + 3, "E1 = 160", ha="right", fontsize=8)
    ax.text(len(vals) - 0.45, ASYM.E2 / 1e9 + 3, "E2 = 80  ← swap23 card at α=90", ha="right", fontsize=8, color=SRC["axis23_z"])
    ax.text(len(vals) - 0.45, ASYM.E3 / 1e9 + 3, "E3 = 20  ← α=90 contract (= ANBA)", ha="right", fontsize=8, color=SRC["adapter"])
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("EA / A  (GPa)")
    ax.set_ylim(0, 190)
    ax.legend(fontsize=8)
    ax.set_title("Which modulus sits on the beam?   diamonds = ANBA")
    return _save(fig, dest)


def plot_source_23(rows: list[Cmp], dest: Path | None = None) -> Path | None:
    """1:1 vs swap23 at α=0 only — the 2↔3 shear signature, one colour."""
    plt = _plt()
    dest = dest or (OUT / "source_23.png")
    pick = [
        r for r in rows
        if r.group == "axis23" and "β=90" not in r.row and "α=90" not in r.row
    ]
    if not pick:
        return None
    fig, axes = plt.subplots(1, len(pick), figsize=(2.15 * len(pick) + 1.2, 3.6), sharey=True)
    if len(pick) == 1:
        axes = [axes]
    x = np.arange(6)
    for ax, r in zip(axes, pick, strict=True):
        rel = np.nan_to_num(r.rel, nan=0.0) * 100.0
        ax.bar(x, rel, color=[SRC[s] for s in r.source], edgecolor="0.2", lw=0.4)
        ax.set_xticks(x, LABELS, fontsize=7)
        title = r.row.replace("1:1 vs swap23   ", "")
        ax.set_title(title, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("1:1 vs swap23  (%)")
    fig.suptitle(
        "Source: 2↔3 assignment at α=0.  Blank = fibre terms stay.  Red = shear / torsion move.",
        fontsize=11,
        y=1.03,
    )
    return _save(fig, dest)


def plot_source_gx(gx_raw: dict[str, Any], gx_swap: dict[str, Any] | None, dest: Path | None = None) -> Path | None:
    """secfem vs gxbeam, split so each panel asks one question."""
    plt = _plt()
    dest = dest or (OUT / "source_gxbeam.png")
    if not gx_raw:
        return None
    raw = {c["name"]: c for c in gx_raw.get("cases", [])}
    sw = {c["name"]: c for c in (gx_swap or {}).get("cases", [])}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    xx = np.arange(6)
    w = 0.38

    def _rel_of(rec):
        us, gx = _gx_diags(rec)
        return rel_per_dof(us, gx) * 100.0

    # --- leftover catalog (raw card only) ---
    ax = axes[0]
    names = list(raw)
    rels, cols = [], []
    for n in names:
        rels.append(float(np.max(_rel_of(raw[n]))))
        alpha = raw[n].get("alpha_deg", 0.0)
        if n.startswith("iso") or (n.startswith("ud_") and abs(alpha) < 1):
            cols.append(SRC["agree"])
        elif abs(alpha - 45) < 1:
            cols.append(SRC["alpha"])
        else:
            cols.append(SRC["gx_shear"])
    ax.bar(range(len(names)), rels, color=cols, edgecolor="0.2", lw=0.4)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("max rel  (%)")
    ax.set_title("Raw gxbeam card\ncolour = leftover source")

    # --- α=0: adapter only swaps Fx↔Fy ---
    ax = axes[1]
    if "asym_ring_a0" in raw:
        r0 = _rel_of(raw["asym_ring_a0"])
        ax.bar(xx - w / 2, r0, w, color=SRC["gx_shear"], edgecolor="0.2", lw=0.4, label="raw")
    if "asym_ring_a0" in sw:
        r1 = _rel_of(sw["asym_ring_a0"])
        ax.bar(xx + w / 2, r1, w, color=SRC["gx_shear"], edgecolor="0.2", lw=0.4,
               hatch="///", label="adapter")
    ax.set_xticks(xx, LABELS)
    ax.set_title("Asym ring α=0\nadapter moves Fx↔Fy, closes nothing")
    ax.legend(fontsize=7)

    # --- α=90: adapter breaks EA that already matched ---
    ax = axes[2]
    if "asym_ring_a90" in raw:
        r0 = _rel_of(raw["asym_ring_a90"])
        c0 = [SRC["agree"] if v < 100 * AGREE else SRC["gx_shear"] for v in r0]
        ax.bar(xx - w / 2, r0, w, color=c0, edgecolor="0.2", lw=0.4, label="raw")
    if "asym_ring_a90" in sw:
        r1 = _rel_of(sw["asym_ring_a90"])
        c1 = []
        for lab, v in zip(LABELS, r1, strict=True):
            if v < 100 * AGREE:
                c1.append(SRC["agree"])
            elif lab in {"Fz", "Mx", "My"}:
                c1.append(SRC["adapter"])
            else:
                c1.append(SRC["gx_shear"])
        ax.bar(xx + w / 2, r1, w, color=c1, edgecolor="0.2", lw=0.4, hatch="///", label="adapter")
    ax.set_xticks(xx, LABELS)
    ax.set_title("Asym ring α=90\nraw EA matches; adapter breaks it")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, dest)


# ---------------------------------------------------------------------------
# afmesh mesh (geometry + local 2/3)
# ---------------------------------------------------------------------------


def plot_afmesh_mesh(mesh: dict[str, Any], rec: dict[str, Any], dest: Path | None = None) -> Path:
    plt = _plt()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    dest = dest or (OUT / "afmesh_naca.png")
    role_c = {"skin": "#d6eaf8", "spar": "#1c2833", "web": SRC["axis23"]}
    coords = np.asarray(mesh["coords"])
    quads = np.asarray(mesh["quads"])
    roles = mesh["roles"]
    beta = np.asarray(mesh["beta_deg"], float)
    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    ax.add_collection(
        PolyCollection(
            [coords[q] for q in quads],
            facecolors=[role_c[r] for r in roles],
            edgecolors="0.4",
            linewidths=0.18,
        )
    )
    stride = max(1, quads.shape[0] // 28)
    for i in range(0, quads.shape[0], stride):
        c = coords[quads[i]].mean(axis=0)
        b = np.deg2rad(beta[i])
        t = np.array([np.cos(b), np.sin(b)])
        n = np.array([-np.sin(b), np.cos(b)])
        ax.annotate("", xy=c + 0.015 * n, xytext=c,
                    arrowprops=dict(arrowstyle="-|>", color=C_AX[2], lw=1.0, mutation_scale=8))
        ax.annotate("", xy=c - 0.028 * t, xytext=c,
                    arrowprops=dict(arrowstyle="-|>", color=C_AX[3], lw=1.0, mutation_scale=8))
    sc = rec["secfem_1to1"]["shear_center"]
    ax.plot(sc[0], sc[1], "+", color="#148f77", ms=11, mew=1.7, zorder=5)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("NACA 0018   navy=2 thickness   red=3 in-plane   (why skin swap moves shear, not EA)")
    ax.legend(
        handles=[
            Patch(facecolor=role_c["skin"], edgecolor="0.4", label="skin glass biax  E2≠E3"),
            Patch(facecolor=role_c["spar"], edgecolor="0.4", label="spar carbon UD  E2=E3"),
            Patch(facecolor=role_c["web"], edgecolor="0.4", label="web glass biax  E2≠E3"),
        ],
        loc="upper right",
        fontsize=8,
    )
    return _save(fig, dest)


# ---------------------------------------------------------------------------
# backwards-compatible entry points used by the drivers
# ---------------------------------------------------------------------------


def plot_two3(report: dict[str, Any], dest_dir: Path | None = None) -> list[Path]:
    dest_dir = dest_dir or OUT
    written = [plot_axes_flange(dest_dir / "two3_axes.png")]
    p = plot_ea_landing(report, dest_dir / "two3_ea.png")
    if p:
        written.append(p)
    return written


def plot_sonata(reports: list[dict[str, Any]], dest: Path | None = None) -> Path | None:
    rows = collect_comparisons(None, reports, None, None, None)
    return plot_source_23(rows, dest or (OUT / "source_23.png"))


def plot_gxbeam(raw, swapped=None, dest=None):
    return plot_source_gx(raw, swapped, dest or (OUT / "source_gxbeam.png"))


def plot_afmesh_k(rec, dest=None):
    rows = collect_comparisons(None, None, None, None, rec)
    return plot_source_23(rows, dest or (OUT / "source_23.png"))


def plot_source_anba(rows: list[Cmp], dest: Path | None = None) -> Path | None:
    """Just the secfem vs ANBA rows — the first comparison."""
    pick = [r for r in rows if r.row.startswith("secfem vs ANBA")]
    if not pick:
        return None
    return plot_mismatch_map(pick, dest or (OUT / "source_anba.png"))


def plot_overview(two3, sonata, gx_raw, gx_swap, afmesh, dest=None):
    rows = collect_comparisons(two3, sonata, gx_raw, gx_swap, afmesh)
    dest = dest or (OUT / "mismatch_map.png")
    plot_source_table(rows, OUT / "mismatch_sources.png")
    plot_source_23(rows, OUT / "source_23.png")
    plot_source_anba(rows, OUT / "source_anba.png")
    return plot_mismatch_map(rows, dest)
