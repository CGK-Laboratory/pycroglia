import numpy as np
import pyvista as pv

from pycroglia.core.io.geometry_export import (
    GeometryExportSelection,
    export_geometry,
)
from pycroglia.core.io.output import CellAnalysis


def _sample_cell_with_branch(branch: dict) -> CellAnalysis:
    return CellAnalysis(
        cell_territory_volume=0.0,
        cell_volume=0.0,
        ramification_index=0.0,
        number_of_endpoints=1,
        number_of_branches=0,
        avg_branch_length=0.0,
        max_branch_length=0.0,
        min_branch_length=0.0,
        branch_analysis=branch,
        full_cell_analysis={},
    )


def test_export_mask_surface_and_volume(tmp_path):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[3:7, 3:7, 3:7] = 1
    cells = [
        CellAnalysis(
            0.0,
            0.0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            {},
            {},
        )
    ]
    sel = GeometryExportSelection(
        mask_vtk=True,
        mask_volume_vtk=True,
    )
    msgs = export_geometry(
        str(tmp_path), [mask], cells, scale=0.5, zscale=1.0, selection=sel
    )
    assert (tmp_path / "vtk" / "cell_000_surface.vtk").is_file()
    vol_path = tmp_path / "vtk" / "cell_000_boolean_mask.vtk"
    assert vol_path.is_file()
    grid = pv.read(vol_path)
    assert grid.n_points == 10 * 10 * 10
    assert "mask" in grid.point_data
    assert grid.point_data["mask"].max() >= 1
    assert msgs == []


def test_export_skeleton_vtp_endpoint_metadata(tmp_path):
    fm = np.zeros((8, 8, 8), dtype=np.int32)
    fm[2:6, 2:6, 2:6] = 4
    ep = np.zeros_like(fm, dtype=bool)
    ep[3, 3, 3] = True
    branch = {
        "fullmasks": [fm],
        "endpoints": ep,
        "bounding_left": 0,
        "bounding_bottom": 0,
    }
    cells = [_sample_cell_with_branch(branch)]
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1

    sel = GeometryExportSelection(skeleton_vtp=True)
    msgs = export_geometry(
        str(tmp_path), [mask], cells, scale=1.0, zscale=1.0, selection=sel
    )
    path = tmp_path / "vtp" / "cell_000_skeleton.vtp"
    assert path.is_file()
    mesh = pv.read(path)
    assert "is_endpoint" in mesh.point_data
    assert int(mesh.point_data["is_endpoint"].max()) == 1
    assert msgs == []


def test_skip_skeleton_without_fullmasks(tmp_path):
    cells = [
        CellAnalysis(
            0.0,
            0.0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            {0: {"endpoints": []}},
            {},
        )
    ]
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    sel = GeometryExportSelection(skeleton_ply=True)
    msgs = export_geometry(str(tmp_path), [mask], cells, 1.0, 1.0, sel)
    assert not (tmp_path / "ply").exists()
    assert any("skip skeleton" in m.lower() for m in msgs)
