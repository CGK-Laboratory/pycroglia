import numpy as np

import pycroglia.core.centroid as centroids
from pycroglia.ui.controllers.dashboard_state import MetricsDAG


def test_per_cell_analysis_exports_centroids_in_xyz_order():
    """Centroids produced as (z, y, x) are exposed as (x, y, z)."""
    masks = [np.ones((2, 2, 2), dtype=bool), np.ones((2, 2, 2), dtype=bool)]
    dag = MetricsDAG(masks)
    dag._centroids = {
        centroids.KEY_CENTROIDS: np.array(
            [
                [3.0, 5.0, 7.0],
            ]
        )
    }

    per_cell = dag.get_per_cell_analysis()

    assert (per_cell[0].centroid_x, per_cell[0].centroid_y, per_cell[0].centroid_z) == (
        7.0,
        5.0,
        3.0,
    )
    assert (per_cell[1].centroid_x, per_cell[1].centroid_y, per_cell[1].centroid_z) == (
        0.0,
        0.0,
        0.0,
    )
