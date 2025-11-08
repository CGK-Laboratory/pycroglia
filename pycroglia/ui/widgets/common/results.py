from numpy.typing import NDArray
from typing import List

from dataclasses import dataclass


@dataclass
class ImgWithPathResults:
    """Container for an image together with its source file path.

    This simple data container is used to pass image data and the originating
    file path between UI components.

    Attributes:
        file_path (str): Path to the image file.
        img (NDArray): Image array (numpy) associated with the file.
    """

    file_path: str
    img: NDArray
    cells_masks: List[NDArray]
