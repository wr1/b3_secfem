"""Run ANBA4 inside the anba4 Docker image (old API).

Reads /workdir/anba_spec.json, writes /workdir/anba_out.json.

Spec schema::

    {
      "node_xy":   [[x, y], ...],          # (n_nodes, 2) float
      "tri_conn":  [[i0, i1, i2], ...],    # (n_tri, 3) int, 0-indexed
      "material": {
          "type": "isotropic",   E, nu, rho
          "type": "orthotropic", E1..E3, G12..G23, nu12..nu23, rho
      },
      "fiber_orientation_deg":  per-cell array (degrees),
      "plane_orientation_deg":  per-cell array (degrees),
      "degree": 2
    }

Output schema::

    {
      "K": (6, 6),
      "M": (6, 6),
      "anba_order": ["Fz", "Mz", "Fx", "Fy", "Mx", "My"],
    }

The dockerised ANBA in `anba4:latest` exposes the **old** API:
``anbax(mesh, degree, matLibrary, materials, plane_orientations, fiber_orientations)``
(note plane / fiber order is REVERSED from the example_isotropic.py docstring,
matching the public ``examples/anbax_C_section.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    import dolfin
    import numpy as np
    from anba4 import anbax, material

    spec_path = Path("/workdir/anba_spec.json")
    out_path = Path("/workdir/anba_out.json")

    spec = json.loads(spec_path.read_text())

    node_xy = np.asarray(spec["node_xy"], dtype=np.float64)
    tri_conn = np.asarray(spec["tri_conn"], dtype=np.int64)
    fiber = np.asarray(spec["fiber_orientation_deg"], dtype=np.float64)
    plane = np.asarray(spec["plane_orientation_deg"], dtype=np.float64)
    degree = int(spec.get("degree", 2))

    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(node_xy.shape[0])
    editor.init_cells(tri_conn.shape[0])
    for i, p in enumerate(node_xy):
        editor.add_vertex(int(i), [float(p[0]), float(p[1])])
    for i, c in enumerate(tri_conn):
        editor.add_cell(int(i), [int(c[0]), int(c[1]), int(c[2])])
    editor.close()

    n_cells = mesh.num_cells()
    materials_mf = dolfin.MeshFunction(
        "size_t", mesh, mesh.topology().dim()
    )
    fiber_mf = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    plane_mf = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    for i in range(n_cells):
        fiber_mf[i] = float(fiber[i])
        plane_mf[i] = float(plane[i])

    def _build_mat(mat_spec):
        if mat_spec["type"] == "isotropic":
            return material.IsotropicMaterial(
                [float(mat_spec["E"]), float(mat_spec["nu"])],
                float(mat_spec.get("rho", 0.0)),
            )
        if mat_spec["type"] == "orthotropic":
            # ANBA's OrthotropicMaterial 3x3 layout (per material.cpp):
            #   row 0: (E_xx, E_yy, E_zz)         = (E1, E2, E3)
            #   row 1: (G_yz, G_xz, G_xy)         = (G23, G13, G12)
            #   row 2: (nu_zy, nu_zx, nu_xy)      = (nu23, nu13, nu12)
            prop = np.array(
                [
                    [mat_spec["E1"], mat_spec["E2"], mat_spec["E3"]],
                    [mat_spec["G23"], mat_spec["G13"], mat_spec["G12"]],
                    [mat_spec["nu23"], mat_spec["nu13"], mat_spec["nu12"]],
                ],
                dtype=np.float64,
            )
            return material.OrthotropicMaterial(
                prop, float(mat_spec.get("rho", 0.0))
            )
        msg = f"unknown material type: {mat_spec['type']}"
        raise ValueError(msg)

    if "material" in spec:
        # Single-material shortcut (back-compat).
        mat_library = [_build_mat(spec["material"])]
        materials_mf.set_all(0)
    else:
        mat_library = [_build_mat(m) for m in spec["material_library"]]
        material_id = np.asarray(spec["material_id"], dtype=np.int64)
        if material_id.shape != (n_cells,):
            msg = (
                f"material_id length {material_id.shape} != n_cells {n_cells}"
            )
            raise ValueError(msg)
        for i in range(n_cells):
            materials_mf[i] = int(material_id[i])

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

    # NOTE the order: anbax(..., materials, plane_orientations, fiber_orientations)
    # — plane comes BEFORE fiber here, per the public example.
    anba = anbax(mesh, degree, mat_library, materials_mf, plane_mf, fiber_mf)
    stiff = anba.compute()
    mass = anba.inertia()

    K = np.array(stiff.getValues(range(6), range(6)))
    M = np.array(mass.getValues(range(6), range(6)))

    # Empirically (verified on iso unit square): the dockerised ANBA's
    # K matrix is in [Fx, Fy, Fz, Mx, My, Mz] order — same as b3_secfem.
    # No permutation needed to compare against b3_secfem.K directly.
    result = {
        "anba_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "K": K.tolist(),
        "M": M.tolist(),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
