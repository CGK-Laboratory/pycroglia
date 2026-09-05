from dataclasses import dataclass
from typing import List

from pycroglia.core.labeled_cells import LabeledCells


@dataclass
class SegmentationResults:
    """Data class containing the results of cell segmentation for a single image.

    Stores the file path and the labeled cells resulting from the segmentation
    process. Used to pass segmentation results between workflow steps.

    Attributes:
        file_path (str): Path to the original image file that was segmented.
        img (LabeledCells): The labeled cells object containing segmentation results.
        cell_indices (Optional[List[int]]): List of all cell indices from segmentation.
        gray_filter_value (Optional[float]): Gray filter threshold used.
        min_size (Optional[int]): Minimum size for small object removal.
        erosion_radius (Optional[int]): Erosion footprint radius.
    """

    file_path: str
    img: LabeledCells
    segmented_cell_indices: List[int]
    gray_filter_value: float
    min_size: int
    erosion_radius: int

    def as_dict(self) -> dict:
        """Convert the segmentation results to a dictionary representation.

        Returns:
            dict: Dictionary containing file_path, img, and metadata as key-value pairs.
        """
        return {
            "file_path": self.file_path,
            "img": self.img,
            "segmented_cell_indices": self.segmented_cell_indices,
            "gray_filter_value": self.gray_filter_value,
            "min_size": self.min_size,
            "erosion_radius": self.erosion_radius,
        }
