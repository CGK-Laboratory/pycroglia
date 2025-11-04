from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray


class FullCellAnalysisPlot:
    """3D visualization and export utility for full-cell convex hull analysis results.

    This class creates 3D convex hull visualizations for a collection of cell masks
    based on a computed :class:`AnalysisResult`. Each mask corresponds to a cell,
    and its convex hull is rendered using `matplotlib`'s 3D plotting interface.

    The plots display each cell’s convex volume and morphological complexity,
    providing a geometric summary of cell morphology. Each figure can be individually
    saved to disk in multiple formats (e.g., PNG, PDF, SVG).

    Attributes:
        figs (list[Figure]): List of Matplotlib figures generated for each cell.
        axes (list[Axes3D]): List of 3D axes corresponding to each figure.
    """

    def __init__(
        self,
        fca: dict[str, Any],
        masks: list[NDArray],
        figsize: tuple[int, int] = (5, 5),
        color: str = "cyan",
        alpha: float = 0.3,
        edgecolor: str = "black",
        linewidths: float = 0.8,
    ) -> None:
        """Initialize the 3D plotter for full-cell convex hull visualizations.

        Args:
            fca (AnalysisResult):
                Object containing convex hull simplices, volumes, and complexity
                metrics for each analyzed cell.
            masks (list[NDArray]):
                List of 3D binary arrays, where each element represents a segmented
                cell volume. Nonzero voxels correspond to cell structures.
            figsize (tuple[int, int], optional):
                Size of each generated figure in inches. Defaults to (5, 5).
            color (str, optional):
                Fill color for convex hull surfaces. Defaults to "cyan".
            alpha (float, optional):
                Transparency level for convex hull surfaces, between 0 and 1.
                Defaults to 0.3.
            edgecolor (str, optional):
                Color of the hull edges. Defaults to "black".
            linewidths (float, optional):
                Width of the convex hull edge lines. Defaults to 0.8.
        """
        self.figs = []
        self.axes = []
        plt.ioff()
        
        for i, mask in enumerate(masks):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            # Original voxel coordinates
            coords = np.argwhere(mask)  # (z, y, x)
            # Reorder for plotting: (z, y, x) -> (x, y, z)
            plot_coords = coords[:, [2, 1, 0]]

            simplices = fca["convex_simplices"][i]
            for simplex in simplices:
                tri = plot_coords[simplex]  # reorder axes for each simplex
                ax.add_collection3d(
                    Poly3DCollection(
                        [tri],
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                        linewidths=linewidths,
                    )
                )
            x, y, z = mask.shape[2], mask.shape[1], mask.shape[0]

            # Slightly inset limits to ensure axes are visible
            ax.set_xlim(-0.02 * x, 1.02 * x)
            ax.set_ylim(-0.02 * y, 1.02 * y)
            ax.set_zlim(-0.02 * z, 1.02 * z)
            ax.view_init(elev=30, azim=-60)  # Top-down / orthogonal
            ax.set_proj_type("ortho")  # Orthographic projection
            ax.set_box_aspect([1, 1, 1.5])
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.90)
            ax.set_xlabel("Z (µm)", labelpad=10, fontsize=10)
            ax.set_ylabel("Y (µm)", labelpad=10, fontsize=10)
            ax.set_zlabel("X (µm)", labelpad=10, fontsize=10)

            # Tick formatting: even spacing across dimensions

            ax.set_title(
                f"Cell {i + 1} - Volume: {fca['convex_volumes'][i]:.3f}, "
                f"Complexity: {fca['cell_complexities'][i]:.3f}"
            )

            self.figs.append(fig)
            self.axes.append(ax)

    def show_all(self, block=True) -> None:
        """Display all generated figures together.

        This method re-enables interactive mode and displays all
        figures that were previously created.

        Example:
            ```python
            plotter = FullCellAnalysisPlot(results, masks)
            plotter.show_all()
            ```
        """
        plt.ion()
        plt.show(block=block)
