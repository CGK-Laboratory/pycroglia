from typing import Any
import numpy as np
import pyvista as pv
from numpy.typing import NDArray


class FullCellAnalysisPlot:
    """3D visualization and export utility for full-cell convex hull analysis results (PyVista version).

    This class creates interactive 3D visualizations of convex hulls for a collection
    of cell masks using PyVista. Each cell’s convex volume and morphological
    complexity are shown with translucent colored meshes.

    Attributes:
        plotters (list[pv.Plotter]): List of PyVista plotter objects for each cell.
    """

    def __init__(
        self,
        fca: dict[str, Any],
        masks: list[NDArray],
        color: str = "cyan",
        opacity: float = 0.3,
        edge_color: str = "black",
        line_width: float = 1.0,
        lighting: str = "gouraud",
    ) -> None:
        """Initialize PyVista-based 3D convex hull visualization.

        Args:
            fca (dict[str, Any]): Analysis results containing convex hull simplices,
                volumes, and complexities for each cell.
            masks (list[NDArray]): List of 3D binary arrays representing each cell.
            color (str, optional): Surface color for convex hull meshes.
            opacity (float, optional): Surface transparency.
            edge_color (str, optional): Color of mesh edges.
            line_width (float, optional): Edge line width.
            lighting (str, optional): Lighting style ('flat', 'gouraud', etc.).
        """
        self.plotters: list[pv.Plotter] = []

        for i, mask in enumerate(masks):
            coords = np.argwhere(mask)  # (z, y, x)
            points = coords[:, [2, 1, 0]].astype(float)  # reorder to (x, y, z)
            simplices = np.asarray(fca["convex_simplices"][i], dtype=np.int32)

            # Build a triangular mesh for PyVista
            faces = np.hstack(
                [np.full((simplices.shape[0], 1), 3), simplices]
            ).flatten()
            mesh = pv.PolyData(points, faces)

            plotter = pv.Plotter()
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                edge_color=edge_color,
                line_width=line_width,
                lighting=lighting,
                smooth_shading=True,
                show_edges=True,
            )

            plotter.add_text(
                f"Cell {i + 1}\nVolume: {fca['convex_volumes'][i]:.3f}\n"
                f"Complexity: {fca['cell_complexities'][i]:.3f}",
                position="upper_left",
                font_size=10,
            )
            plotter.show_grid(
                xtitle="X (µm)",
                ytitle="Y (µm)",
                ztitle="Z (µm)",
                color="gray",
            )
            plotter.show_axes()
            plotter.camera_position = "iso"
            self.plotters.append(plotter)

    def show_all(self) -> None:
        """Show all PyVista windows sequentially (non-blocking)."""
        for plotter in self.plotters:
            plotter.show()
