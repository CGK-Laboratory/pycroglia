import pytest

from numpy import array_equal
from pathlib import Path
from pycroglia.core.tests.integration.helpers import read_hdf5_results
from pycroglia.core.files import TiffReader

# Test cases folder
TEST_CASES_PATH = Path(__file__).parent / "cases"

# TIFF files test cases's folder
TEST_CASES_TIFF_PATH = TEST_CASES_PATH / "tiff"
TEST_CASE_TIFF = TEST_CASES_TIFF_PATH / "testcase.tif"


@pytest.mark.parametrize(
    "test_case, ch, chi",
    [
        ("ch-1-chi-1.h5", 1, 1),
        ("ch-2-chi-1.h5", 2, 1),
        ("ch-2-chi-2.h5", 2, 2),
        ("ch-5-chi-2.h5", 5, 2),
    ],
)
def test_tiff_reader(test_case, ch, chi):
    """Integration test for TiffReader with various channel configurations.

    Verifies that TiffReader correctly processes TIFF files by comparing
    its output against precomputed reference results stored in HDF5 files.

    The test covers multiple combinations of channel parameters:
    - Different channel intervals (ch).
    - Different channel interest indices (chi).

    Test data flow:
    1. Load reference results from HDF5 file.
    2. Read and process TIFF file using TiffReader.
    3. Compare actual results with reference data.

    Args:
        test_case: Name of HDF5 file containing reference results
        ch: Channel interval parameter for TiffReader
        chi: Channel interest index parameter for TiffReader
    """
    path = str(TEST_CASES_TIFF_PATH / test_case)

    expected = read_hdf5_results(file=path)

    reader = TiffReader(str(TEST_CASE_TIFF))
    results = reader.read(ch, chi)

    # Ensure both arrays are (z, y, x)
    assert results.shape == expected.data.shape
    assert array_equal(expected.data.transpose(0, 2, 1), results)
