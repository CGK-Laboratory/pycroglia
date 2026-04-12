"""Export cell masks and skeleton branch surfaces to mesh and VTK volume formats.

Skeleton branch arrays from analysis are XY-cropped; they are embedded into each
cell's full (Z, Y, X) mask shape before marching cubes so outputs share one volume frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation
from skimage import measure

from pycroglia.core.branch_analysis import (
    KEY_BOUNDING_BOTTOM,
    KEY_BOUNDING_LEFT,
    KEY_ENDPOINTS,
    KEY_FULLMASKS,
)
from pycroglia.core.io.output import CellAnalysis

BRANCH_LEVEL_RGB = {
    4: (1.0, 0.0, 0.0),
    3: (1.0, 1.0, 0.0),
    2: (0.0, 1.0, 0.0),
    1: (0.0, 0.0, 1.0),
}
ENDPOINT_RGB = (1.0, 0.0, 1.0)

_SURFACE_FORMATS = frozenset({"obj", "ply", "vtp", "vtk"})


@dataclass
class GeometryExportSelection:
    # Skeleton surface mesh formats
    skeleton_obj: bool = False
    skeleton_ply: bool = False
    skeleton_vtp: bool = False
    skeleton_vtk: bool = False

    # Cell mask surface mesh formats (marching-cubes PolyData)
    mask_obj: bool = False
    mask_ply: bool = False
    mask_vtp: bool = False
    mask_vtk: bool = False

    # Cell boolean mask volume formats (ImageData)
    mask_volume_vtk: bool = False  # saves as .vtk ImageData
    mask_volume_vti: bool = False  # saves as .vti ImageData

    def any_geometry_selected(self) -> bool:
        return any(
            (
                self.skeleton_obj,
                self.skeleton_ply,
                self.skeleton_vtp,
                self.skeleton_vtk,
                self.mask_obj,
                self.mask_ply,
                self.mask_vtp,
                self.mask_vtk,
                self.mask_volume_vtk,
                self.mask_volume_vti,
            )
        )

    def skeleton_formats(self) -> set[str]:
        s: set[str] = set()
        if self.skeleton_obj:
            s.add("obj")
        if self.skeleton_ply:
            s.add("ply")
        if self.skeleton_vtk:
            s.add("vtk")
        if self.skeleton_vtp:
            s.add("vtp")
        return s

    def mask_surface_formats(self) -> set[str]:
        """Cell mask surface mesh (marching-cubes PolyData): obj/ply/vtp/vtk."""
        s: set[str] = set()
        if self.mask_obj:
            s.add("obj")
        if self.mask_ply:
            s.add("ply")
        if self.mask_vtp:
            s.add("vtp")
        if self.mask_vtk:
            s.add("vtk")
        return s

    def mask_volume_formats(self) -> set[str]:
        """Boolean mask ImageData volumes: vtk and/or vti."""
        s: set[str] = set()
        if self.mask_volume_vtk:
            s.add("vtk")
        if self.mask_volume_vti:
            s.add("vti")
        return s


def _triangulate_faces(faces: NDArray) -> NDArray:
    return np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)


def _embed_bounded_in_full_volume(
    bounded: NDArray,
    full_shape: tuple[int, int, int],
    left: int,
    bottom: int,
) -> NDArray:
    """Place XY-cropped ``bounded`` (Z, Yb, Xb) into a full (Z, Y, X) array."""
    b = np.asarray(bounded)
    nz, ny, nx = full_shape
    bz, by, bx = b.shape
    if bz != nz:
        raise ValueError(
            f"Depth mismatch: bounded Z={bz} vs full Z={nz} (branch data must span full Z)."
        )
    out = np.zeros(full_shape, dtype=b.dtype)
    y_end = min(bottom + by, ny)
    x_end = min(left + bx, nx)
    by_use = y_end - bottom
    bx_use = x_end - left
    if by_use > 0 and bx_use > 0:
        out[:, bottom:y_end, left:x_end] = b[:, :by_use, :bx_use]
    return out


def _mask_to_polydata(mask: NDArray, scale: float, zscale: float) -> pv.PolyData | None:
    m = np.asarray(mask) > 0
    if not np.any(m):
        return None
    verts, faces, _, _ = measure.marching_cubes(m.astype(np.float64), level=0.5)
    verts = verts[:, [2, 1, 0]].astype(np.float64)
    verts[:, 0] *= scale
    verts[:, 1] *= scale
    verts[:, 2] *= zscale
    return pv.PolyData(verts, _triangulate_faces(faces))


def _skeleton_polydata(
    fullmask: NDArray,
    endpoints: NDArray | None,
    scale: float,
    zscale: float,
) -> pv.PolyData | None:
    fm = np.asarray(fullmask)
    if fm.size == 0 or not np.any(fm > 0):
        return None

    meshes: list[pv.PolyData] = []
    for level, rgb in BRANCH_LEVEL_RGB.items():
        sub = fm == level
        if not np.any(sub):
            continue
        part = _mask_to_polydata(sub, scale, zscale)
        if part is None:
            continue
        part["branch_order"] = np.full(part.n_points, level, dtype=np.int32)
        rgb_arr = np.array(rgb, dtype=np.float64)
        colors = np.tile(rgb_arr, (part.n_points, 1))
        part["RGB"] = (colors * 255.0).astype(np.uint8)
        meshes.append(part)

    if not meshes:
        return None

    combined = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]

    if (
        endpoints is not None
        and isinstance(endpoints, np.ndarray)
        and endpoints.ndim == 3
        and endpoints.shape == fm.shape
        and np.any(endpoints)
    ):
        ep_hit = binary_dilation(endpoints, structure=np.ones((3, 3, 3), dtype=bool))
        pts = combined.points
        iz = np.clip(np.floor(pts[:, 2] / zscale + 0.5).astype(int), 0, fm.shape[0] - 1)
        iy = np.clip(np.floor(pts[:, 1] / scale + 0.5).astype(int), 0, fm.shape[1] - 1)
        ix = np.clip(np.floor(pts[:, 0] / scale + 0.5).astype(int), 0, fm.shape[2] - 1)
        is_ep = ep_hit[iz, iy, ix].astype(np.uint8)
        combined["is_endpoint"] = is_ep
        rgb_pts = combined["RGB"].copy().astype(np.float64) / 255.0
        ep_color = np.array(ENDPOINT_RGB, dtype=np.float64)
        rgb_pts[is_ep.astype(bool)] = ep_color
        combined["RGB"] = (rgb_pts * 255.0).astype(np.uint8)

    return combined


def _binary_mask_to_image_data(
    mask: NDArray,
    scale: float,
    zscale: float,
) -> pv.ImageData:
    """VTK ImageData: X,Y,Z axes with spacing (scale, scale, zscale); values 0/1."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    nz, ny, nx = m.shape
    # VTK point dimensions along x, y, z
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(float(scale), float(scale), float(zscale)),
        origin=(0.0, 0.0, 0.0),
    )
    # x varies fastest in VTK layout
    grid.point_data["mask"] = m.transpose(2, 1, 0).ravel(order="F")
    return grid


def _write_surface(mesh: pv.PolyData, out_root: str, fmt: str, relative_stem: str) -> None:
    if fmt not in _SURFACE_FORMATS:
        raise ValueError(f"Unsupported surface format: {fmt}")
    folder = os.path.join(out_root, fmt)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{relative_stem}.{fmt}")
    mesh.save(path)


def _write_image_data(
    grid: pv.ImageData,
    out_root: str,
    fmt: str,
    relative_stem: str,
) -> None:
    """Write *grid* as ImageData into ``out_root/{fmt}/{relative_stem}.{fmt}``."""
    if fmt not in {"vtk", "vti"}:
        raise ValueError(f"Unsupported ImageData format: {fmt}")
    folder = os.path.join(out_root, fmt)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{relative_stem}.{fmt}")
    grid.save(path)


def _branch_dict_for_cell(cells: list[CellAnalysis], index: int) -> dict:
    if not cells or index >= len(cells):
        return {}
    ba = cells[index].branch_analysis
    if not isinstance(ba, dict):
        return {}
    br = ba.get(index)
    return br if isinstance(br, dict) else {}


def _ndarray_to_image_data(
    arr: NDArray,
    scale: float,
    zscale: float,
) -> pv.ImageData:
    """Wrap an arbitrary (Z, Y, X) array as a pyvista ImageData volume.

    The data is stored in the ``point_data['values']`` field with the spacing
    derived from *scale* (XY) and *zscale* (Z).
    """
    a = np.asarray(arr)
    nz, ny, nx = a.shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(float(scale), float(scale), float(zscale)),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["values"] = a.transpose(2, 1, 0).ravel(order="F")
    return grid


def export_geometry(
    out_root: str,
    cells_masks: list[NDArray],
    cells: list[CellAnalysis],
    scale: float,
    zscale: float,
    selection: GeometryExportSelection,
) -> list[str]:
    """Write selected geometry under ``out_root/{fmt}/``.

    Directory layout
    ----------------
    out_root/
      obj/   cell_000_skeleton.obj,  cell_000_surface.obj,  …
      ply/   cell_000_skeleton.ply,  cell_000_surface.ply,  …
      vtp/   cell_000_skeleton.vtp,  cell_000_surface.vtp,  …
      vtk/   cell_000_skeleton.vtk,  cell_000_surface.vtk,  …
             cell_000_boolean_mask.vtk
      vti/   cell_000_boolean_mask.vti

    Returns:
        List of human-readable messages for skipped cells or failures (non-fatal).
    """
    messages: list[str] = []
    if not selection.any_geometry_selected():
        return messages

    sk_fmt = selection.skeleton_formats() & _SURFACE_FORMATS
    mk_fmt = selection.mask_surface_formats() & _SURFACE_FORMATS
    vol_fmt = selection.mask_volume_formats()  # "vtk" and/or "vti"

    # ------------------------------------------------------------------
    # Per-cell loop: skeleton, mask surface, mask volume
    # ------------------------------------------------------------------
    for i, mask in enumerate(cells_masks):
        stem_skel = f"cell_{i:03d}_skeleton"
        stem_mask = f"cell_{i:03d}_surface"
        stem_vol = f"cell_{i:03d}_boolean_mask"

        branch = _branch_dict_for_cell(cells, i)
        left = int(branch.get(KEY_BOUNDING_LEFT, 0))
        bottom = int(branch.get(KEY_BOUNDING_BOTTOM, 0))
        vol_shape = tuple(int(x) for x in np.asarray(cells_masks[i]).shape)

        # ---- skeleton surface ----
        if sk_fmt:
            fullmasks = branch.get(KEY_FULLMASKS)
            if not fullmasks or not isinstance(fullmasks, (list, tuple)):
                messages.append(f"Cell {i}: skip skeleton (no fullmasks).")
            else:
                fm0 = np.asarray(fullmasks[0])
                ep = branch.get(KEY_ENDPOINTS)
                if not isinstance(ep, np.ndarray):
                    ep = None
                try:
                    fm_full = _embed_bounded_in_full_volume(fm0, vol_shape, left, bottom)
                    ep_full = None
                    if ep is not None and ep.shape == fm0.shape:
                        ep_full = _embed_bounded_in_full_volume(
                            ep.astype(bool, copy=False), vol_shape, left, bottom
                        )
                    sk_mesh = _skeleton_polydata(fm_full, ep_full, scale, zscale)
                except Exception as exc:  # noqa: BLE001
                    messages.append(f"Cell {i}: skeleton mesh failed: {exc}")
                    sk_mesh = None
                if sk_mesh is not None:
                    for fmt in sk_fmt:
                        try:
                            _write_surface(sk_mesh, out_root, fmt, stem_skel)
                        except Exception as exc:  # noqa: BLE001
                            messages.append(
                                f"Cell {i}: write skeleton {fmt} failed: {exc}"
                            )

        # ---- mask surface (marching cubes) ----
        if mk_fmt:
            try:
                surf = _mask_to_polydata(mask, scale, zscale)
            except Exception as exc:  # noqa: BLE001
                messages.append(f"Cell {i}: mask surface failed: {exc}")
                surf = None
            if surf is None:
                messages.append(f"Cell {i}: skip mask surface (empty or marching cubes).")
            else:
                for fmt in mk_fmt:
                    try:
                        _write_surface(surf, out_root, fmt, stem_mask)
                    except Exception as exc:  # noqa: BLE001
                        messages.append(f"Cell {i}: write mask surface {fmt} failed: {exc}")

        # ---- per-cell boolean mask volume (ImageData) ----
        if vol_fmt:
            try:
                cell_vol = _binary_mask_to_image_data(mask, scale, zscale)
            except Exception as exc:  # noqa: BLE001
                messages.append(f"Cell {i}: binary mask ImageData failed: {exc}")
                cell_vol = None
            if cell_vol is not None:
                for fmt in vol_fmt:
                    try:
                        _write_image_data(cell_vol, out_root, fmt, stem_vol)
                    except Exception as exc:  # noqa: BLE001
                        messages.append(f"Cell {i}: write mask volume {fmt} failed: {exc}")

    return messages
