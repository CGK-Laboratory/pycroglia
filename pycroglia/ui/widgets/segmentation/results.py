from dataclasses import dataclass

from pycroglia.core.labeled_cells import LabeledCells


@dataclass
class SegmentationResults:
    """Data class containing the results of cell segmentation for a single image.

    Stores the file path and the labeled cells resulting from the segmentation
    process. Used to pass segmentation results between workflow steps.

    Attributes:
        file_path (str): Path to the original image file that was segmented.
        img (LabeledCells): The labeled cells object containing segmentation results.
    """

    file_path: str
    img: LabeledCells

    def as_dict(self) -> dict:
        """Convert the segmentation results to a dictionary representation.

        Returns:
            dict: Dictionary containing file_path and img as key-value pairs.
        """
        return {"file_path": self.file_path, "img": self.img}
