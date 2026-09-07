"""Strain-recovery analytic checks."""

from __future__ import annotations

import pytest

dolfinx = pytest.importorskip("dolfinx")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from b3_secfem import (
    IsotropicMaterial,
    OrthotropicMaterial,
    RegionMat,
    SectionInput,
    assemble_resultants_from_sigma,
    plot_unit_load_fields,
    recover_strains,
    recover_unit_load_strains,
    solve,
    write_xdmf,
)


def _iso_rectangle(tmp_path, a=0.1, b=0.1, n=12):
    iso = IsotropicMaterial(E=210e9, nu=0.3, rho=7850.0)
    path = tmp_path / "rect.xdmf"
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)
    inp = SectionInput(
        mesh_path=path,
        region_materials={1: RegionMat(material=iso)},
        linear_solver="lu",
    )
    return iso, path, inp, a, b


def test_axial_recovery_isotropic(tmp_path):
    """Under unit axial generalised strain (eps_zz = 1), a homogeneous
    isotropic free section recovers eps_xx = eps_yy = -nu (Poisson
    contraction) and eps_zz = 1, with negligible shear.
    """
    iso, _path, inp, _a, _b = _iso_rectangle(tmp_path, n=12)
    res = solve(inp)
    fields = recover_strains(res)

    eps_zz = fields.epsilon[2, :, 2].mean()
    eps_xx = fields.epsilon[2, :, 0].mean()
    eps_yy = fields.epsilon[2, :, 1].mean()
    np.testing.assert_allclose(eps_zz, 1.0, rtol=1e-3)
    np.testing.assert_allclose(eps_xx, -iso.nu, rtol=1e-2)
    np.testing.assert_allclose(eps_yy, -iso.nu, rtol=1e-2)
    # Shear components negligible.
    for k in (3, 4, 5):
        np.testing.assert_allclose(fields.epsilon[2, :, k].mean(), 0.0, atol=1e-6)


def test_bending_recovery_eps_zz_linear_in_y(tmp_path):
    """Under unit kappa_x, eps_zz on the section is -y (linear distribution)."""
    _iso, _path, inp, _a, _b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    fields = recover_strains(res)

    # eps_zz over the section under kappa_x = 1 is -y. Cell areas times -y
    # cell-centroid should give zero (symmetric section). Variance > 0.
    eps_zz_kappa_x = fields.epsilon[3, :, 2]
    np.testing.assert_allclose(eps_zz_kappa_x.mean(), 0.0, atol=1e-6)
    assert eps_zz_kappa_x.std() > 0.01  # nontrivial spread


def test_unit_load_resultants_are_identity(tmp_path):
    """Integrated resultants of the 6 recovered unit-load σ fields are I.

    Spatial identity is O(h²) because recovered σ is DG0; the algebraic
    identity R @ inv(R) = I is exact.
    """
    _iso, _path, inp, _a, _b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    recovered = np.vstack(
        [assemble_resultants_from_sigma(ul.sigma[k], res.mesh) for k in range(6)]
    )
    np.testing.assert_allclose(recovered, np.eye(6), atol=2e-2)
    Gamma = np.linalg.solve(res.R, np.eye(6))
    np.testing.assert_allclose(res.R @ Gamma, np.eye(6), atol=1e-10)


def _cell_centroids(mesh) -> np.ndarray:
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    dofmap = mesh.geometry.dofmap[:n_cells]
    return np.asarray(mesh.geometry.x[:, :2], dtype=float)[dofmap].mean(axis=1)


def _area_mean(values: np.ndarray, areas: np.ndarray) -> float:
    return float(np.dot(values, areas) / areas.sum())


def _rel_rms(got: np.ndarray, expect: np.ndarray, areas: np.ndarray) -> float:
    err = got - expect
    rms_err = float(np.sqrt(np.dot(err**2, areas) / areas.sum()))
    rms_ref = float(np.sqrt(np.dot(expect**2, areas) / areas.sum()))
    return rms_err / max(rms_ref, 1e-30)


def test_unit_fz_sigma_zz_is_1_over_A(tmp_path):
    iso, _path, inp, a, b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    area = a * b
    mean_zz = _area_mean(ul.sigma[2, :, 2], ul.cell_areas)
    np.testing.assert_allclose(mean_zz, 1.0 / area, rtol=2e-2)
    np.testing.assert_allclose(
        _area_mean(ul.sigma[2, :, 3], ul.cell_areas), 0.0, atol=1e-4
    )
    np.testing.assert_allclose(
        _area_mean(ul.sigma[2, :, 4], ul.cell_areas), 0.0, atol=1e-4
    )


def test_unit_mx_sigma_zz_is_y_over_I(tmp_path):
    _iso, _path, inp, a, b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    xy = _cell_centroids(res.mesh)
    Ixx = a * b**3 / 12.0
    expect = xy[:, 1] / Ixx
    assert _rel_rms(ul.sigma[3, :, 2], expect, ul.cell_areas) < 0.05


def test_unit_my_sigma_zz_is_minus_x_over_I(tmp_path):
    _iso, _path, inp, a, b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    xy = _cell_centroids(res.mesh)
    Iyy = b * a**3 / 12.0
    expect = -xy[:, 0] / Iyy
    assert _rel_rms(ul.sigma[4, :, 2], expect, ul.cell_areas) < 0.05


def test_unit_fx_mean_tau_xz_is_1_over_A(tmp_path):
    _iso, _path, inp, a, b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    mean_xz = _area_mean(ul.sigma[0, :, 4], ul.cell_areas)
    np.testing.assert_allclose(mean_xz, 1.0 / (a * b), rtol=5e-2)


def test_unit_fy_mean_tau_yz_is_1_over_A(tmp_path):
    _iso, _path, inp, a, b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    mean_yz = _area_mean(ul.sigma[1, :, 3], ul.cell_areas)
    np.testing.assert_allclose(mean_yz, 1.0 / (a * b), rtol=5e-2)


def test_unit_mz_mean_sigma_zz_is_zero(tmp_path):
    _iso, _path, inp, _a, _b = _iso_rectangle(tmp_path, n=16)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    np.testing.assert_allclose(
        _area_mean(ul.sigma[5, :, 2], ul.cell_areas), 0.0, atol=1e-4
    )


def test_plot_unit_load_fields_writes_png(tmp_path):
    _iso, _path, inp, _a, _b = _iso_rectangle(tmp_path, n=8)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    dest = tmp_path / "fields.png"
    plot_unit_load_fields(res, ul, dest, title="unit-load σ")
    assert dest.is_file() and dest.stat().st_size > 0


def _rotated_ortho_rectangle(
    tmp_path, beta_deg=20.0, alpha_deg=15.0, a=0.1, b=0.1, n=10
):
    """Homogeneous rectangle of an off-axis (rotated) orthotropic ply.

    A nonzero, non-multiple-of-90 (beta, alpha) is essential here: it is
    exactly the case for which the material frame and the section (global)
    frame disagree on every Voigt component, so pairing the wrong stress/
    strain pair is guaranteed to be visible.
    """
    mat = OrthotropicMaterial(
        E1=140e9,
        E2=10e9,
        E3=10e9,
        G12=5e9,
        G13=5e9,
        G23=3.5e9,
        nu12=0.3,
        nu13=0.3,
        nu23=0.4,
        rho=1600.0,
    )
    path = tmp_path / "rect_ortho.xdmf"
    m = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [(-a / 2, -b / 2), (a / 2, b / 2)],
        [n, n],
        cell_type=dmesh.CellType.quadrilateral,
    )
    write_xdmf(path, m)
    inp = SectionInput(
        mesh_path=path,
        region_materials={
            1: RegionMat(material=mat, beta_deg=beta_deg, alpha_deg=alpha_deg)
        },
        linear_solver="lu",
    )
    return mat, path, inp


def test_material_frame_energy_matches_global_frame(tmp_path):
    """Strain energy density is frame-invariant: 0.5*sigma.eps must be the
    same whether evaluated in the global or the material (ply) frame.

    Regression guard for a bug where ``sigma_mat`` (correctly rotated into
    the material frame) was paired downstream with the *global* ``epsilon``
    instead of a material-frame strain, silently breaking this invariant
    for any off-axis anisotropic ply while leaving K/M/frequencies
    unaffected (the corruption only shows up in recovered stress/strain
    energy, e.g. modal-strain-energy damping estimates).
    """
    _mat, _path, inp = _rotated_ortho_rectangle(tmp_path)
    res = solve(inp)
    ul = recover_unit_load_strains(res)

    for k in range(6):
        energy_global = np.einsum("cv,cv->c", ul.sigma[k], ul.epsilon[k])
        energy_mat = np.einsum("cv,cv->c", ul.sigma_mat[k], ul.epsilon_mat[k])
        np.testing.assert_allclose(
            energy_mat,
            energy_global,
            rtol=1e-6,
            atol=1e-6 * max(np.abs(energy_global).max(), 1e-30),
            err_msg=f"energy not frame-invariant for unit load case {k}",
        )


def test_epsilon_mat_differs_from_global_epsilon_when_rotated(tmp_path):
    """Guard against silently re-aliasing epsilon_mat to the global strain.

    For a genuinely off-axis ply the material-frame and global strains must
    differ meaningfully under torque (Mz, the load case most affected by
    the real bug). A test that never exercises this would pass whether or
    not epsilon_mat was actually computed.
    """
    _mat, _path, inp = _rotated_ortho_rectangle(tmp_path)
    res = solve(inp)
    ul = recover_unit_load_strains(res)

    mz = 5  # unit-load order is [Fx, Fy, Fz, Mx, My, Mz]
    diff = np.abs(ul.epsilon_mat[mz] - ul.epsilon[mz])
    scale = np.abs(ul.epsilon[mz]).max()
    assert diff.max() > 1e-3 * max(scale, 1e-30)


def test_sigma_mat_consistent_with_local_constitutive_law(tmp_path):
    """sigma_mat must equal C_local @ epsilon_mat cell-by-cell: that is the
    definition of "material frame", and is what makes energy frame-
    invariant. Catches a regression to pairing sigma_mat with the global
    strain (which instead satisfies sigma_mat == C_local @ T.T @ epsilon,
    not C_local @ epsilon_mat).
    """
    mat, _path, inp = _rotated_ortho_rectangle(tmp_path, beta_deg=33.0, alpha_deg=-12.0)
    res = solve(inp)
    ul = recover_unit_load_strains(res)
    C_local = mat.C_local()

    for k in range(6):
        predicted = np.einsum("ij,cj->ci", C_local, ul.epsilon_mat[k])
        np.testing.assert_allclose(
            predicted,
            ul.sigma_mat[k],
            rtol=1e-6,
            atol=1e-6 * max(np.abs(ul.sigma_mat[k]).max(), 1e-30),
        )
