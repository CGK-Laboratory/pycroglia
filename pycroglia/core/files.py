from abc import ABC, abstractmethod
from numpy import ndarray, dstack
from tifffile import TiffFile
from pathlib import Path

from pycroglia.core.errors.errors import PycrogliaException


class MultiChReader(ABC):
    """Abstract base class for multi-channel image readers."""

    @abstractmethod
    def read(self, ch: int, ch_interest: int) -> ndarray:
        """Reads the specified channels from the image file.

        Args:
            ch (int): The number of channels in the image.
            ch_interest (int): The channel of interest to extract (1-based index).

        Returns:
            numpy.ndarray: The extracted channel data.
        """
        pass


class TiffReader(MultiChReader):
    """Reader for TIFF image files.

    Attributes:
        VALID_EXTENSIONS (list): Valid file extensions ['.tif', '.tiff'].
        EXTENSION_ERROR_CODE (int): Error code for invalid extensions (1001).
        path (str): Path of the file.
    """

    VALID_EXTENSIONS = [".tif", ".tiff"]
    EXTENSION_ERROR_CODE = 1001

    def __init__(self, path: str) -> None:
        """Initializes TiffReader with file path.

        Args:
            path: Path to TIFF file

        Raises:
            PycrogliaException(1000): If the path doesn't exist.
            PycrogliaException(1001): If the file doesn't have the expected extension.
        """
        self.path = Path(path)
        self.validate_path()

    def validate_path(self) -> None:
        """Validates the path of the file.

        Raises:
            PycrogliaException(1000): If the path doesn't exist.
            PycrogliaException(1001): If the file doesn't have the expected extension.
        """
        if not self.path.exists():
            raise PycrogliaException(1000)

        if self.path.suffix not in self.VALID_EXTENSIONS:
            raise PycrogliaException(self.EXTENSION_ERROR_CODE)

    def validate_channels(self, ch: int, ch_interest: int) -> None:
        """Validates the channel and channel of interest values.

        Args:
            ch (int): The number of channels.
            ch_interest (int): The channel extracted from the file.

        Raises:
            PycrogliaException(1003): If the channel value is invalid.
            PycrogliaException(1004): If the channel of interest is out of range.
        """
        if ch < 0:
            raise PycrogliaException(1003)

        if ch_interest - 1 >= ch:
            raise PycrogliaException(1004)

    def read(self, ch: int, ch_interest: int) -> ndarray:
        """Reads the contents of the file and returns the specified channels.

        Args:
            ch (int): The number of channels.
            ch_interest (int): The channel extracted from the file.

        Returns:
            numpy.ndarray: The file data.

        Raises:
            PycrogliaException(1003): If the channel value is invalid.
            PycrogliaException(1004): If the channel of interest is out of range.
        """
        self.validate_channels(ch, ch_interest)
        data = []

        with TiffFile(str(self.path)) as lsm_file:
            number_of_images = len(lsm_file.pages)

            for i in range(ch_interest - 1, number_of_images, ch):
                data.append(lsm_file.pages[i].asarray())

        return dstack(data)


class LsmReader(MultiChReader):
    """Reader for LSM image files.

    Attributes:
        VALID_EXTENSIONS (list): Valid file extensions [".lsm"].
        EXTENSION_ERROR_CODE (int): Error code for invalid extensions (1002).
        path (str): Path of the file.

    Raises:
        PycrogliaException(1000): If the path doesn't exist.
        PycrogliaException(1002): If the file doesn't have the expected extension.
    """

    VALID_EXTENSIONS = [".lsm"]
    EXTENSION_ERROR_CODE = 1002

    def __init__(self, path: str) -> None:
        """Initializes LsmReader with file path.

        Args:
            path: Path to TIFF file
        """
        self.path = Path(path)
        self.validate_path()

    def validate_path(self) -> None:
        """Validates the path of the file.

        Raises:
            PycrogliaException(1000): If the path doesn't exist.
            PycrogliaException(1002): If the file doesn't have the expected extension.
        """
        if not self.path.exists():
            raise PycrogliaException(1000)

        if self.path.suffix not in self.VALID_EXTENSIONS:
            raise PycrogliaException(self.EXTENSION_ERROR_CODE)

    def validate_channels(self, ch: int, ch_interest: int) -> None:
        """Validates the channel and channel of interest values.

        Args:
            ch (int): The number of channels.
            ch_interest (int): The channel extracted from the file.

        Raises:
            PycrogliaException(1003): If the channel value is invalid.
            PycrogliaException(1004): If the channel of interest is out of range.
        """
        if ch < 0:
            raise PycrogliaException(1003)

        if ch_interest - 1 >= ch:
            raise PycrogliaException(1004)

    def read(self, ch: int, ch_interest: int) -> ndarray:
        """Reads the contents of the file and returns the specified channels.

        Args:
            ch (int): The number of channels.
            ch_interest (int): The channel extracted from the file.

        Returns:
            numpy.ndarray: The file data.

        Raises:
            PycrogliaException(1003): If the channel value is invalid.
            PycrogliaException(1004): If the channel of interest is out of range.
        """
        self.validate_channels(ch, ch_interest)
        data = []

        with TiffFile(str(self.path)) as tiff_file:
            number_of_images = len(tiff_file.pages)

            for i in range(ch_interest - 1, number_of_images, ch):
                data.append(tiff_file.pages[i].asarray())

        return dstack(data)


def create_channeled_reader(path: str) -> MultiChReader:
    """Creates a MultiChReader instance based on the file extension.

    Args:
        path (str): Path to the image file.

    Returns:
        MultiChReader: An instance of TiffReader or LsmReader.

    Raises:
        PycrogliaException(1005): If the file extension is not supported.
    """
    suffix = Path(path).suffix

    if suffix in TiffReader.VALID_EXTENSIONS:
        return TiffReader(path)

    if suffix in LsmReader.VALID_EXTENSIONS:
        return LsmReader(path)

    raise PycrogliaException(1005)
