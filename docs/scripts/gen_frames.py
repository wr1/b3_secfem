"""Convention figures for the fumano site.

Every placed arrow is ``material_axes(β, α)``. Run::

    micromamba run -n b3secfem python docs/scripts/gen_frames.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from b3_secfem.rotation3d import _Ry, _Rz, material_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "figures"
INK, MUTED = "#5c6370", "#8a8fa0"
C1, C2, C3 = "#922b21", "#1a5276", "#c0392b"
SX, SY, SZ = "#1b7a4e", "#6b4c9a", "#0b5f8a"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
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


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    fig.savefig(dest, dpi=160, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    print(dest)
    return dest


def _flange(ax) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (-0.020, -0.005),
            0.040,
            0.010,
            boxstyle="square,pad=0",
            facecolor="#eef1f4",
            edgecolor=MUTED,
            lw=1.3,
        )
    )
    ax.plot(0, 0, "o", color=INK, ms=4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.030, 0.030)
    ax.set_ylim(-0.022, 0.022)
    ax.set_xlabel("sec_1 ≡ x")
    ax.set_ylabel("sec_2 ≡ y")
    ax.axhline(0, color=MUTED, lw=0.35)
    ax.axvline(0, color=MUTED, lw=0.35)


def _arrow(ax, vec, color, label, scale=0.013) -> None:
    v = np.asarray(vec, float)
    xy = v[:2]
    n = np.linalg.norm(xy)
    if n < 1e-12:
        ax.add_patch(Circle((0, 0), 0.0032, fill=False, ec=color, lw=1.7, zorder=6))
        if v[2] > 0:
            ax.plot(0, 0, "o", color=color, ms=3.5, zorder=7)
            extra = " ⊙ +z"
        else:
            ax.plot(0, 0, "x", color=color, ms=6, mew=1.6, zorder=7)
            extra = " ⊗ −z"
        ax.text(
            0.0,
            -0.0175,
            f"{label}{extra}",
            ha="center",
            va="top",
            color=color,
            fontsize=8,
            fontweight="bold",
        )
        return
    d = xy / n * scale
    ax.annotate(
        "",
        xy=d,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=13),
        zorder=6,
    )
    ax.text(
        d[0] * 1.22,
        d[1] * 1.22,
        label,
        color=color,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )


def _panel(ax, beta: float, alpha: float, title: str) -> dict[str, np.ndarray]:
    _flange(ax)
    ax_map = material_axes(beta, alpha)
    _arrow(ax, ax_map["mat_1"], C1, "mat_1")
    _arrow(ax, ax_map["mat_2"], C2, "mat_2")
    _arrow(ax, ax_map["mat_3"], C3, "mat_3")
    ax.set_title(title, pad=6)
    return ax_map


def fig_sec_frame() -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    _flange(ax)
    _arrow(ax, [1, 0, 0], SX, "sec_1", scale=0.016)
    _arrow(ax, [0, 1, 0], SY, "sec_2", scale=0.014)
    _arrow(ax, [0, 0, 1], SZ, "sec_3")
    ax.set_title("1  ·  section frame")
    fig.tight_layout()
    return _save(fig, "sec_frame.png")


def fig_mat_card() -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_xlim(-1.6, 1.8)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        plt.Polygon(
            [[-1.15, -0.35], [1.15, -0.35], [1.15, 0.35], [-1.15, 0.35]],
            closed=True,
            fc="#eef1f4",
            ec=MUTED,
            lw=1.4,
        )
    )
    for y in np.linspace(-0.22, 0.22, 5):
        ax.plot([-1.05, 1.05], [y, y], color=C1, lw=1.0, alpha=0.45)
    ax.annotate(
        "",
        xy=(1.05, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.2, mutation_scale=16),
    )
    ax.annotate(
        "",
        xy=(0, 0.55),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.2, mutation_scale=16),
    )
    ax.add_patch(Circle((0, 0), 0.08, fill=False, ec=C3, lw=2.0))
    ax.plot(0, 0, "o", color=C3, ms=5)
    ax.text(1.12, 0.08, "mat_1  fibre", color=C1, fontsize=11, fontweight="bold")
    ax.text(0.06, 0.62, "mat_2  in-ply transverse", color=C2, fontsize=11, fontweight="bold")
    ax.text(
        0.12,
        -0.18,
        "mat_3  toward viewer\n(ply normal / through-thickness)",
        color=C3,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.78,
        "MATERIAL CARD, before placement.  1–2 = ply plane,  3 = ply normal.",
        ha="center",
        va="top",
        fontsize=10,
        color=INK,
    )
    ax.set_title("2  ·  material card")
    return _save(fig, "mat_card.png")


def fig_element_winding() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    def quad(ax, order, title, labels):
        # Always draw the geometric square. `order` is visit order only —
        # filling in visit order bows the dolfinx perm into two triangles.
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        ax.fill(corners[[0, 1, 2, 3], 0], corners[[0, 1, 2, 3], 1],
                color="#eef1f4", ec=MUTED, lw=1.4)
        seq = corners[list(order)]
        for i, p in enumerate(seq):
            ax.plot(*p, "o", color=INK, ms=6)
            ax.text(p[0] + 0.04, p[1] + 0.04, f"{i}  {labels[i]}", fontsize=9, color=INK)
        # winding along the visit sequence, inset so it stays inside the square
        c = corners.mean(0)
        a = seq[0] * 0.55 + c * 0.45
        b = seq[1] * 0.55 + c * 0.45
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops=dict(arrowstyle="-|>", color=SX, lw=1.6, mutation_scale=12),
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.25, 1.35)
        ax.set_ylim(-0.25, 1.35)
        ax.set_title(title)
        ax.set_xlabel("sec_1 = x")
        ax.set_ylabel("sec_2 = y")
        ax.axhline(0, color=MUTED, lw=0.3)
        ax.axvline(0, color=MUTED, lw=0.3)

    quad(
        axes[0],
        (0, 1, 2, 3),
        "gxbeam / meshio  CCW from +z\nBL → BR → TR → TL",
        ["BL", "BR", "TR", "TL"],
    )
    quad(
        axes[1],
        (0, 3, 1, 2),
        "dolfinx 0.10 tensor-product\nBL → TL → BR → TR   perm [0, 3, 1, 2]",
        ["BL", "TL", "BR", "TR"],
    )
    fig.suptitle("element numbering")
    fig.tight_layout()
    return _save(fig, "element_winding.png")


def fig_place(beta: float, alpha: float, name: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    axm = _panel(ax, beta, alpha, title)
    fig.text(
        0.5,
        0.02,
        "  ".join(f"{k}→{np.round(v, 2)}" for k, v in axm.items()),
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, name)


def fig_beta_row(alpha: float, name: str, subtitle: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    for ax, b in zip(axes, (0.0, 45.0, 90.0), strict=True):
        _panel(ax, b, alpha, f"β = {b:g}°   α = {alpha:g}")
    fig.suptitle(subtitle)
    fig.tight_layout()
    return _save(fig, name)


def fig_beta_alpha45() -> Path:
    fig = plt.figure(figsize=(10.8, 4.6))

    def box(ax, hx=0.020, hy=0.005, hz=0.010):
        faces = [
            [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)],
            [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
            [(-hx, -hy, 0), (hx, -hy, 0), (hx, -hy, hz), (-hx, -hy, hz)],
            [(-hx, hy, 0), (hx, hy, 0), (hx, hy, hz), (-hx, hy, hz)],
            [(-hx, -hy, 0), (-hx, hy, 0), (-hx, hy, hz), (-hx, -hy, hz)],
            [(hx, -hy, 0), (hx, hy, 0), (hx, hy, hz), (hx, -hy, hz)],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors="#dfe3e8", edgecolors=MUTED, alpha=0.45, linewidths=0.5
            )
        )
        ax.set_xlim(-0.026, 0.026)
        ax.set_ylim(-0.026, 0.026)
        ax.set_zlim(0.0, 0.026)
        ax.set_xlabel("x = sec_1", fontsize=8)
        ax.set_ylabel("y = sec_2", fontsize=8)
        ax.set_zlabel("z = sec_3", fontsize=8)
        ax.tick_params(labelsize=6)
        try:
            ax.set_box_aspect((1, 1, 0.55))
        except Exception:
            pass

    for i, b in enumerate((0.0, 45.0, 90.0)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        box(ax)
        axm = material_axes(b, 45.0)
        o = np.array([0.0, 0.0, 0.004])
        scale = 0.015
        for key, c in (("mat_1", C1), ("mat_2", C2), ("mat_3", C3)):
            v = axm[key] * scale
            ax.quiver(*o, *v, color=c, arrow_length_ratio=0.18, linewidth=2.0)
            ax.text(*(o + v * 1.25), key, color=c, fontsize=8, fontweight="bold")
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"β = {b:g}°   α = 45", pad=8)
    fig.suptitle("β at α = 45  (3-D: a 2-D view would collapse mat_1 and mat_2)")
    fig.tight_layout()
    return _save(fig, "place_beta_alpha45.png")


def _swapped_axes(beta: float, alpha: float) -> dict[str, np.ndarray]:
    R = _Ry(alpha) @ _Rz(beta)
    return {
        "mat_1": R @ np.array([1.0, 0.0, 0.0]),
        "mat_2": R @ np.array([0.0, 1.0, 0.0]),
        "mat_3": R @ np.array([0.0, 0.0, 1.0]),
    }


def _panel_from(ax, ax_map: dict[str, np.ndarray], title: str) -> None:
    _flange(ax)
    _arrow(ax, ax_map["mat_1"], C1, "mat_1")
    _arrow(ax, ax_map["mat_2"], C2, "mat_2")
    _arrow(ax, ax_map["mat_3"], C3, "mat_3")
    ax.set_title(title, pad=6)


def fig_angle_order() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.2))
    _panel(axes[0, 0], 45.0, 0.0, "chosen  Rz(β)·Ry(α)\n(β, α) = (45, 0)")
    _panel_from(axes[0, 1], _swapped_axes(45.0, 0.0),
                "swapped  Ry(α)·Rz(β)\n(β, α) = (45, 0)")
    _panel(axes[1, 0], 45.0, 45.0, "chosen  (β, α) = (45, 45)")
    _panel_from(axes[1, 1], _swapped_axes(45.0, 45.0),
                "swapped  (β, α) = (45, 45)")
    fig.suptitle("order of β and α  ·  they do not commute")
    fig.text(
        0.5,
        0.02,
        "Chosen product: α about global y, then β about global z.\n"
        "They agree only when β = 0 or α = 0 (one factor is I).",
        ha="center",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return _save(fig, "place_angle_order.png")


def fig_anba_dropin() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    _panel(axes[0], 0.0, 0.0, "zero angles  (β, α) = (0, 0)\n= ANBA (fiber, plane) = (0, 0)")
    _panel(axes[1], 0.0, 90.0, "fibre on the beam  (β, α) = (0, 90)\n= ANBA (fiber, plane) = (90, 0)")
    fig.suptitle("same zero as ANBA  ·  spanwise UD is 90°")
    fig.text(
        0.5,
        0.02,
        "Drop-in: same card. β = plane, α = fiber. No 2↔3 rewrite.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    return _save(fig, "anba_dropin.png")


def fig_formulas() -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.92, "the rotation", ha="center", va="top", fontsize=14, color=INK)
    ax.text(0.06, 0.78, "secfem  =  ANBA T at the same numbers", fontsize=11, fontweight="bold", color=C2)
    ax.text(
        0.06,
        0.62,
        "R(β, α)  =  Rz(β)  ·  Ry(α)\n"
        "(0, 0)   →  mat_1 = +x,   mat_2 = +y,   mat_3 = +z\n"
        "(0, 90)  →  fibre along −z (beam)",
        fontsize=10,
        family="monospace",
        va="top",
        color=INK,
    )
    ax.text(0.06, 0.38, "SONATA / ANBA drop-in", fontsize=11, fontweight="bold", color=C1)
    ax.text(
        0.06,
        0.26,
        "same card\n"
        "β = plane,   α = fiber\n"
        "anba_to_secfem_input(card, fiber, plane)",
        fontsize=10,
        family="monospace",
        va="top",
        color=INK,
    )
    ax.text(
        0.06,
        0.08,
        "Product order is α about global y, then β about global z.\n"
        "Rz(β) and Ry(α) do not commute except at β = 0 or α = 0.",
        fontsize=10,
        va="bottom",
        color=INK,
    )
    return _save(fig, "formulas.png")


def fig_voigt() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.axis("off")
    cols = ["index", "section", "meaning"]
    rows = [
        ["0", "11 = xx", "σ / ε along sec_1"],
        ["1", "22 = yy", "σ / ε along sec_2"],
        ["2", "33 = zz", "σ / ε along sec_3 (beam)"],
        ["3", "23 = yz", "engineering γ_yz"],
        ["4", "13 = xz", "engineering γ_xz"],
        ["5", "12 = xy", "engineering γ_xy"],
    ]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.55)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d4dc")
        if r == 0:
            cell.set_facecolor("#e8ebf0")
            cell.set_text_props(weight="bold")
    ax.set_title("Voigt order after placement  ·  forces [Fx, Fy, Fz, Mx, My, Mz]")
    return _save(fig, "voigt_force.png")


def main() -> None:
    _style()
    fig_sec_frame()
    fig_mat_card()
    fig_element_winding()
    fig_place(0.0, 0.0, "place_a0.png", "(β, α) = (0, 0)  ·  identity")
    fig_place(0.0, 90.0, "place_a90.png", "(β, α) = (0, 90)  ·  fibre on the beam")
    fig_place(90.0, 0.0, "place_b90_a0.png", "(β, α) = (90, 0)  ·  fibre along +y")
    fig_beta_row(0.0, "place_beta_alpha0.png", "β at α = 0  is the in-plane fibre angle")
    fig_beta_row(90.0, "place_beta_alpha90.png", "β at α = 90  rolls 2/3; fibre stays on −z")
    fig_beta_alpha45()
    fig_angle_order()
    fig_anba_dropin()
    fig_formulas()
    fig_voigt()


if __name__ == "__main__":
    main()
