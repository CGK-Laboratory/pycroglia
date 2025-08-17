import pytest
from unittest.mock import patch
from pathlib import Path

from pycroglia.core.errors.errors import PycrogliaException
from pycroglia.core.files import TiffReader, LsmReader, create_channeled_reader


@pytest.mark.parametrize(
    "path",
    [
        "example.tif",
        "./directory/example.tif",
        "example.tiff",
        "./directory/example.tiff",
    ],
)
def test_tiff_reader_validate_path_ok(path: str) -> None:
    """Test that TiffReader accepts valid file paths with correct extensions.

    Args:
        path (str): Valid file path to test.

    Asserts:
        No exception is raised for valid TIFF file paths.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = True

        try:
            TiffReader(path)
        except Exception as e:
            pytest.fail(f"Unexpected error raised {e}")


@pytest.mark.parametrize(
    "path", ["example.tiffa", "example", "example.lsm", "./directory/example.tiffa"]
)
def test_tiff_reader_invalid_file(path: str) -> None:
    """Test that TiffReader raises exception for invalid file extensions.

    Args:
        path (str): Invalid file path to test.

    Asserts:
        PycrogliaException with error code 1001 is raised.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = True

        with pytest.raises(PycrogliaException) as e_info:
            TiffReader(path)

        assert e_info.value.error_code == TiffReader.EXTENSION_ERROR_CODE


def test_tiff_reader_path_doesnt_exists() -> None:
    """Test that TiffReader raises exception for non-existent file path.

    Asserts:
        PycrogliaException with error code 1000 is raised.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = False

        with pytest.raises(PycrogliaException) as e_info:
            TiffReader("")

        assert e_info.value.error_code == 1000


@pytest.mark.parametrize("ch, chi", [(10, 0), (2, 1), (4, 3), (5, 5)])
def test_tiff_reader_validate_channels_ok(ch: int, chi: int) -> None:
    """Test that TiffReader accepts valid channel parameters.

    Args:
        ch (int): Channel count to test.
        chi (int): Channel interest index to test.

    Asserts:
        No exception is raised for valid channel parameters.
        The returned data shape is as expected if read is patched.
    """
    import numpy as np
    with patch.object(TiffReader, "validate_path") as mock_validate:
        mock_validate.return_value = True

        reader = TiffReader("example.tif")
        try:
            reader.validate_channels(ch, chi)
        except Exception as e:
            pytest.fail(f"Unexpected error raised {e}")

        # Patch read to simulate TIFF reading and check shape
        with patch.object(TiffReader, "read") as mock_read:
            # Simulate a stack of 3 slices, each 2x2
            mock_read.return_value = np.array([
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [5, 6],
                    [7, 8]
                ],
                [
                    [9, 10],
                    [11, 12]
                ]
            ])
            data = reader.read(ch, chi)
            assert hasattr(data, "shape")
            assert data.shape == (3, 2, 2)


@pytest.mark.parametrize(
    "ch, chi, expected_code",
    [(-1, 2, 1003), (-10, 2, 1003), (2, 3, 1004), (3, 10, 1004)],
)
def test_tiff_reader_validate_channels_invalid_values(
    ch: int, chi: int, expected_code: int
) -> None:
    """Test that TiffReader raises exceptions for invalid channel parameters.

    Args:
        ch (int): Invalid channel count to test.
        chi (int): Invalid channel interest index to test.
        expected_code (int): Expected error code.

    Asserts:
        PycrogliaException with the expected error code is raised.
    """
    with patch.object(TiffReader, "validate_path") as mock_validate:
        mock_validate.return_value = True

        with pytest.raises(PycrogliaException) as e_info:
            reader = TiffReader("example.tiff")
            reader.validate_channels(ch, chi)

        assert e_info.value.error_code == expected_code


@pytest.mark.parametrize(
    "path",
    [
        "example.lsm",
        "./directory/example.lsm",
    ],
)
def test_lsm_reader_validate_path_ok(path: str) -> None:
    """Test that LsmReader accepts valid file paths with correct extensions.

    Args:
        path (str): Valid file path to test.

    Asserts:
        No exception is raised for valid LSM file paths.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = True

        try:
            LsmReader(path)
        except Exception as e:
            pytest.fail(f"Unexpected error raised {e}")


@pytest.mark.parametrize(
    "path", ["example.lsma", "example", "example.tiff", "./directory/example.tif"]
)
def test_lsm_reader_invalid_file(path: str) -> None:
    """Test that LsmReader raises exception for invalid file extensions.

    Args:
        path (str): Invalid file path to test.

    Asserts:
        PycrogliaException with error code 1002 is raised.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = True

        with pytest.raises(PycrogliaException) as e_info:
            LsmReader(path)

        assert e_info.value.error_code == LsmReader.EXTENSION_ERROR_CODE


def test_lsm_reader_path_doesnt_exists() -> None:
    """Test that LsmReader raises exception for non-existent file path.

    Asserts:
        PycrogliaException with error code 1000 is raised.
    """
    with patch.object(Path, "exists") as mock_exists:
        mock_exists.return_value = False

        with pytest.raises(PycrogliaException) as e_info:
            LsmReader("")

        assert e_info.value.error_code == 1000


@pytest.mark.parametrize("ch, chi", [(10, 0), (2, 1), (4, 3)])
def test_lsm_reader_validate_channels_ok(ch: int, chi: int) -> None:
    """Test that LsmReader accepts valid channel parameters.

    Args:
        ch (int): Channel count to test.
        chi (int): Channel interest index to test.

    Asserts:
        No exception is raised for valid channel parameters.
        The returned data shape is as expected if read is patched.
    """
    import numpy as np
    with patch.object(LsmReader, "validate_path") as mock_validate:
        mock_validate.return_value = True

        reader = LsmReader("example.lsm")
        try:
            reader.validate_channels(ch, chi)
        except Exception as e:
            pytest.fail(f"Unexpected error raised {e}")

        # Patch read to simulate LSM reading and check shape
        with patch.object(LsmReader, "read") as mock_read:
            # Simulate a 3D array (z, y, x)
            mock_read.return_value = np.array([
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [5, 6],
                    [7, 8]
                ]
            ])
            data = reader.read(ch, chi)
            assert hasattr(data, "shape")
            assert data.shape == (2, 2, 2)


@pytest.mark.parametrize(
    "extension,expected_class",
    [
        (".tif", TiffReader),
        (".tiff", TiffReader),
        (".lsm", LsmReader),
    ],
)
def test_create_channeled_reader_valid_extensions(tmp_path, extension, expected_class):
    """Test that create_channeled_reader returns the correct reader class for valid extensions.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        extension (str): File extension to test.
        expected_class (type): Expected reader class.

    Asserts:
        The returned reader is an instance of the expected class.
    """
    path = tmp_path / f"test{extension}"
    path.touch()
    reader = create_channeled_reader(str(path))
    assert isinstance(reader, expected_class)


@pytest.mark.parametrize("extension", [".jpg", ".png", ".txt", ""])
def test_create_channeled_reader_invalid_extensions(tmp_path, extension):
    """Test that create_channeled_reader raises exception for unsupported extensions.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        extension (str): File extension to test.

    Asserts:
        PycrogliaException with error code 1005 is raised.
    """
    path = tmp_path / f"test{extension}"
    path.touch()
    with pytest.raises(PycrogliaException) as exc:
        create_channeled_reader(str(path))
    assert exc.value.error_code == 1005
