from enum import Enum


class SkimageCellConnectivity(Enum):
    """Defines connectivity options for labeling connected cell components in 3D images.

    Attributes:
        FACES (int): 6-connectivity (voxels connected by faces).
        EDGES (int): 18-connectivity (voxels connected by faces and edges).
        CORNERS (int): 26-connectivity (voxels connected by faces, edges, and corners).
    """

    FACES = 1
    EDGES = 2
    CORNERS = 3
