import pyvista as pv
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt


class OriginalCellPlot:
    """Render 3-D isometric cell surfaces using PyVista.

    This class visualizes segmented cell volumes as smooth, shaded
    3-D surfaces generated with the marching-cubes algorithm.
    Each cell volume is rendered with physically scaled axes,
    labeled grid lines, and an isometric camera view.

    It supports both desktop (native PyVista windows) and notebook
    rendering (via the Trame backend when available).

    Attributes:
        plots (list[pv.Plotter]): List of PyVista plotters, one per cell.
        scale (float): XY voxel size in physical units (e.g., µm/pixel).
        zscale (float): Z-axis voxel size in physical units.
        opacity (float): Transparency of the cell surfaces in [0, 1].
        cmap (matplotlib.colors.Colormap): Colormap used for per-cell coloring.
    """

    def __init__(
        self,
        masks: list[np.ndarray],
        scale: float,
        zscale: float,
        cmap: str = "tab20",
        opacity: float = 1.0,
    ) -> None:
        """Initialize the 3-D cell surface visualizer.

        Args:
            masks (list[np.ndarray]):
                List of 3-D binary arrays, one per segmented cell.
                Non-zero voxels represent the cell structure.
            scale (float):
                Physical scaling factor for X and Y dimensions (µm/pixel).
            zscale (float):
                Physical scaling factor for Z dimension (µm/pixel).
            cmap (str, optional):
                Matplotlib colormap name used for coloring each cell.
                Defaults to ``"tab20"``.
            opacity (float, optional):
                Surface transparency value in [0, 1]. Defaults to 1.0.
        """
        self.scale, self.zscale = scale, zscale
        self.opacity = opacity
        self.plots: list[pv.Plotter] = []
        self.cmap = plt.get_cmap(cmap, len(masks))

        for i, ex in enumerate(masks, start=1):
            # Extract isosurface with marching cubes
            verts, faces, _, _ = measure.marching_cubes(ex.astype(float), level=0.5)

            # Reorder (z, y, x) → (x, y, z) and apply physical scaling
            verts = verts[:, [2, 1, 0]]
            verts[:, 0] *= scale
            verts[:, 1] *= scale
            verts[:, 2] *= zscale

            # Convert to VTK face format
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
            mesh = pv.PolyData(verts, faces)

            # Configure the PyVista plotter
            plotter = pv.Plotter(off_screen=False)
            color = self.cmap((i - 1) / len(masks))[:3]  # RGB

            plotter.add_mesh(
                mesh,
                color=color,
                opacity=self.opacity,
                smooth_shading=True,
                specular=0.3,
                specular_power=10,
            )
            plotter.show_grid(
                xtitle="X (µm)",
                ytitle="Y (µm)",
                ztitle="Z (µm)",
                color="gray",
            )
            plotter.add_axes()
            plotter.camera_position = "iso"
            plotter.add_text(f"Original cell {i}", font_size=10)
            self.plots.append(plotter)

    def show_all(self, use_trame: bool = False) -> None:
        """Display each rendered cell surface interactively.

        Args:
            use_trame (bool, optional):
                If ``True``, use the Trame backend for inline rendering
                inside Jupyter notebooks (requires ``trame`` packages).
                If ``False``, open native PyVista windows. Defaults to ``False``.

        Example:
            ```python
            plotter = OriginalCellPlot([mask1, mask2], scale=0.1, zscale=0.2)
            plotter.show(use_trame=True)  # for Jupyter notebooks
            ```
        """
        backend = "trame" if use_trame else None
        for p in self.plots:
            if backend:
                p.show(jupyter_backend=backend)
            else:
                p.show()
