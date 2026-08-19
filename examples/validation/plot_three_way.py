#!/usr/bin/env python3
"""Figures for the three-way campaign → public/figures/compare/.

    micromamba run -n b3secfem python examples/validation/plot_three_way.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from validation._common import FORCE_ORDER_SECFEM, OUT, REPO  # noqa: E402

FIG = REPO / "public" / "figures" / "compare"
DOF = list(FORCE_ORDER_SECFEM)
INK, MUTED = "#5c6370", "#8a8fa0"
C_SEC, C_AN, C_GX = "#1a5276", "#922b21", "#1b7a4e"
# agree / stage1 / fy-shear / mid-tilt / gx-no-beta
SRC = {
    "agree": "#7f8c8d",
    "stage1": "#1a5276",
    "fy": "#c0392b",
    "tilt": "#e67e22",
    "gxbeta": "#6c3483",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": MUTED,
        }
    )
    return plt


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    dest = FIG / name
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="none")
    _plt().close(fig)
    print(dest)
    return dest


def _load() -> dict:
    p = OUT / "three_way.json"
    if not p.exists():
        raise SystemExit(f"missing {p} — run three_way.py first")
    return json.loads(p.read_text())


def _rel_row(rec: dict, other: str) -> np.ndarray | None:
    key = "rel_vs_anba" if other == "anba" else "rel_vs_gx"
    if key in rec:
        return np.asarray(rec[key], float)
    blk = rec.get(other) or {}
    if blk.get("skipped") or "K_diag" not in blk:
        return None
    a = np.asarray(rec["secfem"]["K_diag"], float)
    b = np.asarray(blk["K_diag"], float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)


def fig_heatmap(report: dict) -> Path:
    plt = _plt()
    cases = report["cases"]
    names = [c["name"] for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, other, title in (
        (axes[0], "anba", "secfem vs ANBA"),
        (axes[1], "gx", "secfem vs gxbeam"),
    ):
        mat = []
        for c in cases:
            r = _rel_row(c, other)
            mat.append(np.full(6, np.nan) if r is None else 100.0 * r)
        Z = np.asarray(mat)
        im = ax.imshow(Z, aspect="auto", cmap="YlOrRd", vmin=0, vmax=30)
        ax.set_xticks(range(6), DOF)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title)
        for i in range(Z.shape[0]):
            for j in range(6):
                v = Z[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=8)
                else:
                    ax.text(
                        j,
                        i,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        color="white" if v > 15 else INK,
                        fontsize=8,
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% rel")
    fig.suptitle("K diagonal  ·  same physical problem  ·  % |a−b|/max(|a|,|b|)")
    fig.tight_layout()
    return _save(fig, "kdiag_heatmap.png")


def fig_bars(report: dict) -> Path:
    plt = _plt()
    want = ["iso_rect", "asym_beam", "asym_inplane", "asym_tilt", "ud_ellipse_beam"]
    cases = [c for c in report["cases"] if c["name"] in want]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    # EA
    ax = axes[0]
    x = np.arange(len(cases))
    w = 0.25
    for k, (key, col, lab) in enumerate(
        (
            ("secfem", C_SEC, "secfem"),
            ("anba", C_AN, "ANBA"),
            ("gxbeam", C_GX, "gxbeam"),
        )
    ):
        vals = []
        for c in cases:
            blk = c.get(key) or {}
            vals.append(np.nan if blk.get("skipped") or "EA" not in blk else blk["EA"])
        ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x, [c["name"] for c in cases], rotation=20, ha="right")
    ax.set_ylabel("K[Fz, Fz]  (N)")
    ax.set_title("axial stiffness")
    ax.legend(frameon=False)
    # Fy
    ax = axes[1]
    for k, (key, col, lab) in enumerate(
        (
            ("secfem", C_SEC, "secfem"),
            ("anba", C_AN, "ANBA"),
            ("gxbeam", C_GX, "gxbeam"),
        )
    ):
        vals = []
        for c in cases:
            blk = c.get(key) or {}
            vals.append(
                np.nan if blk.get("skipped") or "K_diag" not in blk else blk["K_diag"][1]
            )
        ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x, [c["name"] for c in cases], rotation=20, ha="right")
    ax.set_ylabel("K[Fy, Fy]  (N)")
    ax.set_title("thin-direction shear")
    ax.legend(frameon=False)
    fig.suptitle("agreements (EA) vs open residual (Fy)")
    fig.tight_layout()
    return _save(fig, "ea_vs_fy.png")


def _kshow(ax, K, title, vmax=None):
    K = np.asarray(K, float)
    # log-abs for dynamic range; keep sign with a diverging map on signed log
    s = np.sign(K)
    mag = np.log10(np.maximum(np.abs(K), 1.0))
    Z = s * mag
    lim = vmax if vmax is not None else np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(range(6), DOF, fontsize=7)
    ax.set_yticks(range(6), DOF, fontsize=7)
    ax.set_title(title, fontsize=10)
    return im


def fig_airfoil_K(air: dict) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    panels = [
        (axes[0], air.get("secfem"), "secfem"),
        (axes[1], air.get("anba"), "ANBA"),
        (axes[2], air.get("gxbeam"), "gxbeam"),
    ]
    im = None
    for ax, blk, title in panels:
        if not blk or blk.get("skipped") or "K" not in blk:
            ax.set_axis_off()
            ax.set_title(f"{title}  (missing)")
            continue
        im = _kshow(ax, blk["K"], title)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="sign(K) log10|K|")
    fig.suptitle("NACA 0018  ·  6×6 K in [Fx, Fy, Fz, Mx, My, Mz]")
    return _save(fig, "naca_K.png")


def fig_airfoil_centers(air: dict) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    coords = np.asarray(air["coords"], float)
    quads = np.asarray(air["quads"], int)
    roles = air.get("roles") or ["skin"] * len(quads)
    col = {"skin": "#d5d8dc", "spar": "#aed6f1", "web": "#f5cba7"}
    for q, r in zip(quads, roles, strict=True):
        p = coords[q]
        ax.fill(p[:, 0], p[:, 1], facecolor=col.get(r, "#eee"), edgecolor=MUTED, lw=0.3)
    marks = [
        ("secfem", air.get("secfem") or {}, C_SEC, "o"),
        ("ANBA K", air.get("anba") or {}, C_AN, "D"),
        ("gxbeam", air.get("gxbeam") or {}, C_GX, "s"),
    ]
    for lab, blk, c, m in marks:
        if blk.get("skipped"):
            continue
        sc = blk.get("shear_center") or blk.get("shear_center_from_K")
        if sc and np.isfinite(sc[0]):
            ax.plot(sc[0], sc[1], m, color=c, ms=7, label=f"{lab} shear")
        mc = blk.get("mass_center")
        if mc and np.isfinite(mc[0]):
            ax.plot(mc[0], mc[1], m, color=c, ms=5, mfc="white", label=f"{lab} mass")
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title("NACA 0018  ·  shear and mass centres")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    return _save(fig, "naca_centers.png")


def fig_airfoil_diag(air: dict) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(6)
    w = 0.25
    for k, (key, col, lab) in enumerate(
        (("secfem", C_SEC, "secfem"), ("anba", C_AN, "ANBA"), ("gxbeam", C_GX, "gxbeam"))
    ):
        blk = air.get(key) or {}
        if blk.get("skipped") or "K_diag" not in blk:
            continue
        ax.bar(x + (k - 1) * w, np.abs(blk["K_diag"]), w, color=col, label=lab)
    ax.set_xticks(x, DOF)
    ax.set_yscale("log")
    ax.set_ylabel("|K_ii|")
    ax.set_title("NACA 0018  ·  |K_ii|  (ANBA Fz is negative — see text)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "naca_diag.png")


def main() -> int:
    report = _load()
    fig_heatmap(report)
    fig_bars(report)
    air = report.get("airfoil")
    if air:
        fig_airfoil_K(air)
        fig_airfoil_centers(air)
        fig_airfoil_diag(air)
    else:
        print("no airfoil block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
