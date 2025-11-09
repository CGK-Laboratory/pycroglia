import numpy as np
import pyvista as pv
from skimage import measure
from numpy.typing import NDArray


class BranchpointsCellPlot:
    """Render 3D visualization of skeleton cells with global branchpoints overlay.

    This class replicates the MATLAB visualization where each full skeleton
    is displayed in blue (translucent) and its branchpoints are shown in red (solid).
    It supports multiple skeletons while using a single branchpoint volume shared
    across them.

    Args:
        fullmask (list[NDArray]): List of 3D binary arrays representing skeleton masks.
        branchpoints (NDArray): Single 3D binary array representing all branchpoints.
        scale (float): Physical voxel size along XY axes (µm/pixel).
        zscale (float): Scaling factor for the Z-axis (µm/pixel).
        opacity (float, optional): Transparency of the skeleton surfaces. Defaults to 0.1.
    """

    def __init__(
        self,
        cell_idx: int,
        fullmask: list[NDArray],
        allbranch: NDArray,
        scale: float,
        zscale: float,
        opacity: float = 0.1,
    ) -> None:
        self.scale = scale
        self.zscale = zscale
        self.opacity = opacity
        self.fullmask_list = fullmask
        self.branchpoints = allbranch
        self.plotters: list[pv.Plotter] = []

        for i, mask in enumerate(self.fullmask_list, start=1):
            plotter = pv.Plotter(off_screen=False)

            # --- Full skeleton (blue translucent surface) ---
            if np.any(mask):
                verts, faces, _, _ = measure.marching_cubes(
                    mask.astype(float), level=0.5
                )
                verts = verts[:, [2, 1, 0]]
                verts[:, 0] *= scale
                verts[:, 1] *= scale
                verts[:, 2] *= zscale
                faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(
                    np.int32
                )
                mesh = pv.PolyData(verts, faces)
                plotter.add_mesh(
                    mesh,
                    color="#0000FF",
                    opacity=self.opacity,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=10,
                )

            # --- Branchpoints (red solid surface) ---
            if np.any(self.branchpoints):
                verts2, faces2, _, _ = measure.marching_cubes(
                    self.branchpoints.astype(float), level=0.5
                )
                verts2 = verts2[:, [2, 1, 0]]
                verts2[:, 0] *= scale
                verts2[:, 1] *= scale
                verts2[:, 2] *= zscale
                faces2 = np.hstack([np.full((faces2.shape[0], 1), 3), faces2]).astype(
                    np.int32
                )
                mesh2 = pv.PolyData(verts2, faces2)
                plotter.add_mesh(
                    mesh2,
                    color="#FF0000",
                    opacity=1.0,
                    smooth_shading=True,
                    specular=0.4,
                    specular_power=12,
                )

            # --- Visualization setup ---
            plotter.camera_position = "xy"  # equivalent to MATLAB view(0,270)
            plotter.add_axes()
            plotter.show_grid(
                xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)", color="gray"
            )
            plotter.set_background("white")
            plotter.enable_eye_dome_lighting()
            plotter.camera.zoom(1.2)
            plotter.add_text(f"Branchpoints Cell {cell_idx}", font_size=10)

            self.plotters.append(plotter)

    def show_all(self) -> None:
        """Display all generated 3D skeleton + branchpoints plots sequentially."""
        for p in self.plotters:
            p.show()
