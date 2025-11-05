import numpy as np
import pyvista as pv
from skimage import measure
from numpy.typing import NDArray


class EndpointsCellPlot:
    """Render 3D visualization of skeleton cells with endpoints overlaid.

    This class reproduces the MATLAB visualization of full skeleton structure
    combined with endpoint markers. The full skeleton is shown as a translucent
    blue isosurface, and endpoints are displayed as solid red surfaces.

    Args:
        fullmask (list[NDArray]): List of 3D binary arrays representing the full skeleton masks.
        endpoints (list[NDArray]): List of 3D binary arrays representing endpoint voxels.
        scale (float): Physical voxel size along XY axes (µm/pixel).
        zscale (float): Scaling factor for the Z-axis (µm/pixel).
        opacity (float, optional): Transparency of the skeleton surface. Defaults to 0.1.
    """

    def __init__(
        self,
        cell_idx: int,
        fullmask: list[NDArray],
        endpoints: NDArray,
        scale: float,
        zscale: float,
        opacity: float = 0.1,
    ) -> None:
        self.scale = scale
        self.zscale = zscale
        self.opacity = opacity
        self.fullmask_list = fullmask
        self.plotters: list[pv.Plotter] = []

        for i, mask in enumerate(self.fullmask_list, start=1):
            plotter = pv.Plotter(off_screen=False)

            # --- Full skeleton (blue, translucent) ---
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
                    color="#0000FF",  # Blue
                    opacity=self.opacity,
                    smooth_shading=True,
                    specular=0.3,
                    specular_power=10,
                    show_edges=False,
                )

            # --- Endpoints (red, solid) ---
            if np.any(endpoints):
                verts2, faces2, _, _ = measure.marching_cubes(
                    endpoints.astype(float), level=0.5
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
                    color="#FF0000",  # Red
                    opacity=1.0,
                    smooth_shading=True,
                    specular=0.4,
                    specular_power=12,
                    show_edges=False,
                )

            # --- Visualization setup ---
            plotter.camera_position = "xy"  # Top-down (view(0,270))
            plotter.add_axes()
            plotter.show_grid(
                xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)", color="gray"
            )
            plotter.set_background("white")
            plotter.enable_eye_dome_lighting()
            plotter.camera.zoom(1.2)
            plotter.add_text(f"Endpoints Cell {cell_idx}", font_size=10)

            self.plotters.append(plotter)

    def show_all(self) -> None:
        """Display all generated 3D skeleton + endpoints plots sequentially."""
        for p in self.plotters:
            p.show()
