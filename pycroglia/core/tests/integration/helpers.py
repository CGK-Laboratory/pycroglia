import h5py
from numpy.typing import NDArray

DEFAULT_DATASET_NAME = "imageData"


class HDF5TestCase:
    """Container for HDF5 test case data and metadata.

    Attributes:
        data (NDArray): Image data array.
        x (int): Size in X dimension (width).
        y (int): Size in Y dimension (height).
        z (int): Size in Z dimension (depth).
    """

    def __init__(self, data: NDArray, x: int, y: int, z: int) -> None:
        """Initializes HDF5 test case container.

        Args:
            data: 3D image data array.
            x: Size in X dimension (width).
            y: Size in Y dimension (height).
            z: Size in Z dimension (depth).
        """
        self.data = data
        self.x = x
        self.y = y
        self.z = z


def read_hdf5_results(file: str, dataset: str = DEFAULT_DATASET_NAME) -> HDF5TestCase:
    """Reads HDF5 test results file with dimensional metadata.

    Reads an HDF5 file containing image data and dimensional attributes,
    and returns it in a standardized format for test validation.

    Args:
        file: Path to HDF5 file.
        dataset: Name of dataset containing image data (default: "imageData").

    Returns:
        HDF5TestCase: Container with data and dimensional metadata
    """
    with h5py.File(file, "r") as h5_file:
        img_data = h5_file[dataset]

        y = img_data.attrs["Y"][0]
        x = img_data.attrs["X"][0]
        z = img_data.attrs["Z"][0]

        return HDF5TestCase(img_data[:].transpose(2, 1, 0), x, y, z)
