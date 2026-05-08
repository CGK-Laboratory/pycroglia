from numpy.typing import NDArray
from typing import List, Optional

from dataclasses import dataclass, field


@dataclass
class ImgWithPathResults:
    """Container for an image together with its source file path.

    This simple data container is used to pass image data and the originating
    file path between UI components.

    Attributes:
        file_path (str): Path to the image file.
        img (NDArray): Image array (numpy) associated with the file.
        cells_masks (List[NDArray]): List of individual cell masks.
        segmented_cell_indices (Optional[List[int]]): All cell indices from segmentation.
        final_cell_indices (Optional[List[int]]): Cell indices not rejected after selection.
        gray_filter_value (Optional[float]): Gray filter threshold used.
        min_size (Optional[int]): Minimum size for small object removal.
        erosion_radius (Optional[int]): Erosion footprint radius.
    """

    file_path: str
    img: NDArray
    cells_masks: List[NDArray]
    segmented_cell_indices: Optional[List[int]] = field(default=None)
    selected_cell_indices: Optional[List[int]] = field(default=None)
    rejected_cell_indices: Optional[List[int]] = field(default=None)
    gray_filter_value: Optional[float] = field(default=None)
    min_size: Optional[int] = field(default=None)
    erosion_radius: Optional[int] = field(default=None)
