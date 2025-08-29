from numpy.typing import NDArray


class FilterResults:
    """Encapsulates the results and parameters of the filter pipeline."""

    def __init__(
        self,
        file_path: str,
        gray_filter_value: float,
        min_size: int,
        small_object_filtered_img: NDArray,
    ):
        """
        Args:
            file_path (str): Path to the image file.
            gray_filter_value (float): Value used for the gray filter.
            min_size (int): Minimum size for small object removal.
            small_object_filtered_img (np.ndarray): Image after small object removal.
        """
        self.file_path = file_path
        self.gray_filter_value = gray_filter_value
        self.min_size = min_size
        self.small_object_filtered_img = small_object_filtered_img

    def as_dict(self) -> dict:
        """Returns the filter results as a dictionary.

        Returns:
            dict: Dictionary with filter values and resulting images.
        """
        return {
            "file_path": self.file_path,
            "gray_filter_value": self.gray_filter_value,
            "min_size": self.min_size,
            "small_object_filtered_img": self.small_object_filtered_img,
        }
