import numpy as np
import pyvista as pv
from skimage import measure
from numpy.typing import NDArray


class SkeletonCellPlot:
    """Render 3-D isometric branch-order surfaces from one or more skeleton masks.

    Each `fullmask` volume encodes branch order numerically:
        4 → Primary branches (red)
        3 → Secondary branches (yellow)
        2 → Tertiary branches (green)
        1 → Quaternary branches (blue)

    All non-empty branch levels are displayed together with realistic lighting,
    shading, and spatial scaling. Multiple masks are rendered in separate
    PyVista windows.

    Args:
        fullmask (list[NDArray]):
            List of 3-D integer arrays, each encoding branch order per voxel.
        scale (float):
            Physical voxel size along XY axes (µm/pixel).
        zscale (float):
            Scaling factor for the Z-axis (µm/pixel).
        opacity (float, optional):
            Transparency for each surface. Defaults to 0.5.
    """

    def __init__(
        self,
        cell_idx: int,
        fullmask: list[NDArray],
        scale: float,
        zscale: float,
        opacity: float = 0.5,
    ) -> None:
        self.scale = scale
        self.zscale = zscale
        self.opacity = opacity
        self.fullmask_list = fullmask
        self.plotters: list[pv.Plotter] = []

        # Define branch color mapping
        self.branch_colors = {
            4: (1.0, 0.0, 0.0),  # Primary → Red
            3: (1.0, 1.0, 0.0),  # Secondary → Yellow
            2: (0.0, 1.0, 0.0),  # Tertiary → Green
            1: (0.0, 0.0, 1.0),  # Quaternary → Blue
        }

        # Create one plotter per skeleton volume
        for mask_index, fullmask_volume in enumerate(self.fullmask_list, start=1):
            plotter = pv.Plotter(off_screen=False)

            for level, color in self.branch_colors.items():
                mask = fullmask_volume == level
                if not np.any(mask):
                    continue

                # Compute isosurface for this branch level
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(float), level=0.5
                )
                verts = verts[:, [2, 1, 0]]  # (z, y, x) → (x, y, z)
                verts[:, 0] *= self.scale
                verts[:, 1] *= self.scale
                verts[:, 2] *= self.zscale
                faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(
                    np.int32
                )

                mesh = pv.PolyData(verts, faces)
                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=self.opacity,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=10,
                    show_edges=False,
                )

            # Configure visualization
            plotter.camera_position = "xy"  # Equivalent to MATLAB view(0,270)
            plotter.add_axes()
            plotter.show_grid(
                xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)", color="gray"
            )
            plotter.set_background("white")
            plotter.enable_eye_dome_lighting()
            plotter.camera.zoom(1.2)
            plotter.add_text(f"Skeleton Cell {cell_idx}", font_size=10)

            self.plotters.append(plotter)

    def show_all(self) -> None:
        """Display all generated 3-D skeleton visualizations sequentially.

        Each plot is opened in an interactive PyVista window.
        This call blocks execution until all windows are closed.

        Example:
            ```python
            plotter = SkeletonCellPlot(fullmask_list, scale=0.1, zscale=0.1)
            plotter.show()
            ```
        """
        for p in self.plotters:
            p.show()
