from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets

from pycroglia.core.files import create_channeled_reader, MultiChReader
from pycroglia.core.filters import calculate_otsu_threshold, remove_small_objects


class MultiChImgEditorState(QtCore.QObject):
    """State and logic for multi-channel image editing.

    Attributes:
        reader (MultiChReader): Reader for multi-channel images.
        _img (Optional[np.array]): Loaded image data.
        _gray_filtered_img (Optional[np.array]): Image after gray filter.
        _small_objects_img (Optional[np.array]): Image after small objects filter.
        imageChanged (QtCore.pyqtSignal): Signal emitted when the image changes.
        grayImageChanged (QtCore.pyqtSignal): Signal emitted when the gray-filtered image changes.
    """

    imageChanged = QtCore.pyqtSignal()
    grayImageChanged = QtCore.pyqtSignal()

    def __init__(self, file_path: str, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the multi-channel image editor state.

        Args:
            file_path (str): Path to the image file.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent=parent)

        self.reader: MultiChReader = create_channeled_reader(file_path)

        self._mutex = QtCore.QMutex()
        self._img: Optional[np.array] = None
        self._gray_filtered_img: Optional[np.array] = None
        self._small_objects_img: Optional[np.array] = None

    def get_img(self) -> Optional[np.array]:
        """Return the original loaded image.

        Returns:
            Optional[np.array]: Original image or None if not loaded.
        """

        self._mutex.lock()
        try:
            return self._img
        finally:
            self._mutex.unlock()

    def get_gray_filtered_img(self) -> Optional[np.array]:
        """Return the gray-filtered image.

        Returns:
            Optional[np.array]: Filtered image or None if not filtered.
        """
        self._mutex.lock()
        try:
            return self._gray_filtered_img
        finally:
            self._mutex.unlock()

    def get_small_objects_img(self) -> Optional[np.array]:
        """Return the image after removing small objects.

        Returns:
            Optional[np.array]: Image with small objects removed or None if not filtered.
        """
        self._mutex.lock()
        try:
            return self._small_objects_img
        finally:
            self._mutex.unlock()

    def read_img(self, ch: int, chi: int):
        """Read the image using the specified channels.

        Args:
            ch (int): Number of channels.
            chi (int): Channel of interest.
        """
        self._mutex.lock()
        try:
            self._img = self.reader.read(ch=ch, ch_interest=chi)
            self._gray_filtered_img = None
            self._small_objects_img = None

            self.imageChanged.emit()
        finally:
            self._mutex.unlock()

    def get_midslice(self, arr: np.ndarray):
        """Get the middle slice of a 3D array.

        Args:
            arr (np.ndarray): 3D image array.

        Returns:
            np.ndarray: Middle slice of the image.
        """
        return arr[:, :, arr.shape[2] // 2]

    def apply_otsu_gray_filter(self, adjust_value: float) -> Optional[np.array]:
        """Apply the Otsu threshold filter with adjustment.

        Args:
            adjust_value (float): Adjustment value for the threshold.

        Returns:
            Optional[np.array]: Filtered image or None if no image loaded.
        """
        self._mutex.lock()
        try:
            if self._img is None:
                return None

            masked_image = calculate_otsu_threshold(self._img, adjust_value)
            self._gray_filtered_img = np.where(masked_image, self._img, 0)

            self.grayImageChanged.emit()
            return self._gray_filtered_img
        finally:
            self._mutex.unlock()

    def apply_small_object_filter(self, threshold: int) -> Optional[np.array]:
        """Remove small objects from the filtered image.

        Args:
            threshold (int): Minimum object size threshold.

        Returns:
            Optional[np.array]: Filtered image or None if not filtered.
        """
        self._mutex.lock()
        try:
            if self._gray_filtered_img is None:
                return None

            masked_img = remove_small_objects(self._gray_filtered_img, threshold)
            self._small_objects_img = np.where(masked_img, self._gray_filtered_img, 0)

            return self._small_objects_img
        finally:
            self._mutex.unlock()
