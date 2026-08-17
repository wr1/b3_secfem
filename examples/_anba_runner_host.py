"""Host-side wrapper around examples/anba_runner.py.

Drives ANBA4 inside the ``anba4:latest`` Docker image. The container runs
as the host user (``--user $(id -u):$(id -g)``) so it can write back to
the bind-mounted directory; FFC / dijitso / dolfin caches are redirected
to that same directory so the read-only conda site-packages stays
untouched.

Snap-installed Docker on this host blocks bind-mounting ``/tmp``; the
wrapper requires a workdir under ``$HOME``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np


_RUNNER = Path(__file__).resolve().parent / "anba_runner.py"

_LOCAL_ANBA_PY = (
    Path(os.environ.get("ANBA_PYTHON", ""))
    if os.environ.get("ANBA_PYTHON")
    else Path.home() / ".local" / "share" / "mamba" / "envs" / "anba4" / "bin" / "python"
)


def _docker_image_ok(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _local_anba_ok() -> bool:
    return _LOCAL_ANBA_PY.is_file()


def _quads_to_tris(quads: np.ndarray) -> np.ndarray:
    """Split each CCW quad ``[n0, n1, n2, n3]`` into two CCW triangles."""
    quads = np.asarray(quads, dtype=np.int64)
    t1 = quads[:, [0, 1, 2]]
    t2 = quads[:, [0, 2, 3]]
    return np.vstack([t1, t2])


def anba_section_from_arrays(
    coords: np.ndarray,
    quads: np.ndarray,
    material: object | list,
    fiber_orientation_deg: np.ndarray,
    plane_orientation_deg: np.ndarray,
    *,
    material_id: np.ndarray | None = None,
    workdir_root: Path | None = None,
    image: str = "anba4:latest",
    degree: int = 2,
    recover_fields: bool = False,
) -> dict:
    """Run ANBA4 in Docker and return its 6×6 K, M as numpy arrays.

    Parameters
    ----------
    coords : (n_nodes, 2) float
        2D node positions.
    quads : (n_cells, 4) int
        Quad connectivity, CCW. Split into two tris each before passing
        to ANBA (which uses dolfin 2019 CG triangles).
    material : IsotropicMaterial or OrthotropicMaterial
        b3_secfem material; serialised to JSON for the runner.
    fiber_orientation_deg : (n_cells,) float
        Per-quad fibre angle (degrees). Each tri inherits the parent
        quad's value.
    plane_orientation_deg : (n_cells,) float
        Per-quad plane orientation (degrees). Same fan-out.
    recover_fields : bool
        If True, also recover the 6 unit-load global Voigt strain/stress
        fields (shape ``(6, n_tri, 6)``) plus ``tri_area`` / ``tri_centroid``.
        Child triangles of parent quad ``i`` are ``i`` and ``i + n_quads``.
    workdir_root : Path, optional
        Parent directory for the bind-mounted scratch dir. Must be under
        ``$HOME`` on snap-installed Docker. Defaults to ``$HOME``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64)
    fiber = np.asarray(fiber_orientation_deg, dtype=np.float64)
    plane = np.asarray(plane_orientation_deg, dtype=np.float64)
    if coords.shape[1] == 3:
        coords = coords[:, :2]

    n_quads = quads.shape[0]
    if fiber.shape != (n_quads,):
        msg = f"fiber_orientation_deg shape {fiber.shape} != ({n_quads},)"
        raise ValueError(msg)
    if plane.shape != (n_quads,):
        msg = f"plane_orientation_deg shape {plane.shape} != ({n_quads},)"
        raise ValueError(msg)

    tris = _quads_to_tris(quads)
    fiber_tri = np.concatenate([fiber, fiber])
    plane_tri = np.concatenate([plane, plane])

    if workdir_root is None:
        workdir_root = Path(os.environ.get("HOME", "/home")) / ".anba_workdirs"
    workdir_root.mkdir(parents=True, exist_ok=True)
    workdir = workdir_root / f"anba_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=False)

    try:
        def _to_dict(m):
            cls = type(m).__name__
            if cls == "IsotropicMaterial":
                return {
                    "type": "isotropic",
                    "E": float(m.E), "nu": float(m.nu), "rho": float(m.rho),
                }
            if cls == "OrthotropicMaterial":
                return {
                    "type": "orthotropic",
                    "E1": float(m.E1), "E2": float(m.E2), "E3": float(m.E3),
                    "G12": float(m.G12), "G13": float(m.G13), "G23": float(m.G23),
                    "nu12": float(m.nu12), "nu13": float(m.nu13), "nu23": float(m.nu23),
                    "rho": float(m.rho),
                }
            msg = f"unsupported material class for ANBA: {cls}"
            raise TypeError(msg)

        spec: dict = {
            "node_xy": coords.tolist(),
            "tri_conn": tris.tolist(),
            "fiber_orientation_deg": fiber_tri.tolist(),
            "plane_orientation_deg": plane_tri.tolist(),
            "degree": degree,
            "recover_fields": bool(recover_fields),
        }
        if isinstance(material, (list, tuple)):
            if material_id is None:
                msg = "must supply material_id when material is a list"
                raise ValueError(msg)
            mid_quad = np.asarray(material_id, dtype=np.int64)
            if mid_quad.shape != (n_quads,):
                msg = f"material_id shape {mid_quad.shape} != ({n_quads},)"
                raise ValueError(msg)
            mid_tri = np.concatenate([mid_quad, mid_quad])
            spec["material_library"] = [_to_dict(m) for m in material]
            spec["material_id"] = mid_tri.tolist()
        else:
            spec["material"] = _to_dict(material)
        (workdir / "anba_spec.json").write_text(json.dumps(spec))
        shutil.copy(_RUNNER, workdir / "anba_runner.py")

        if _docker_image_ok(image):
            cmd = [
                "docker", "run", "--rm",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "DIJITSO_CACHE_DIR=/workdir/.dijitso",
                "-e", "XDG_CACHE_HOME=/workdir/.cache",
                "-e", "INSTANT_CACHE_DIR=/workdir/.instant",
                "-e", "DOLFIN_CACHE_DIR=/workdir/.dolfin",
                "-e", "HOME=/workdir",
                "-v", f"{workdir}:/workdir",
                "--entrypoint", "/usr/local/bin/_entrypoint.sh",
                image,
                "python", "/workdir/anba_runner.py",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elif _local_anba_ok():
            env = os.environ.copy()
            env["ANBA_WORKDIR"] = str(workdir)
            env["HOME"] = str(workdir)
            env["DIJITSO_CACHE_DIR"] = str(workdir / ".dijitso")
            env["XDG_CACHE_HOME"] = str(workdir / ".cache")
            env["INSTANT_CACHE_DIR"] = str(workdir / ".instant")
            env["DOLFIN_CACHE_DIR"] = str(workdir / ".dolfin")
            proc = subprocess.run(
                [str(_LOCAL_ANBA_PY), str(workdir / "anba_runner.py")],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=str(workdir),
            )
        else:
            raise RuntimeError(
                f"no ANBA: docker image {image!r} missing and "
                f"local {_LOCAL_ANBA_PY} not found"
            )
        out_path = workdir / "anba_out.json"
        if not out_path.exists():
            msg = (
                f"ANBA runner did not produce output. stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
            raise RuntimeError(msg)

        data = json.loads(out_path.read_text())
        out = {
            "K": np.asarray(data["K"]),
            "M": np.asarray(data["M"]),
            "anba_order": data["anba_order"],
        }
        if recover_fields:
            if "sigma" not in data:
                msg = "ANBA runner did not return recovered fields"
                raise RuntimeError(msg)
            out["epsilon"] = np.asarray(data["epsilon"], dtype=np.float64)
            out["sigma"] = np.asarray(data["sigma"], dtype=np.float64)
            out["tri_area"] = np.asarray(data["tri_area"], dtype=np.float64)
            out["tri_centroid"] = np.asarray(data["tri_centroid"], dtype=np.float64)
            out["load_order"] = data.get("load_order", data["anba_order"])
            out["voigt"] = data.get("voigt")
            out["reference"] = data.get("reference")
        return out
    finally:
        # Tear down the scratch dir; dolfin caches inside it can be
        # several MB.
        shutil.rmtree(workdir, ignore_errors=True)
