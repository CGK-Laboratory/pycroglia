import numpy as np

from pycroglia.ui.widgets.imagefilters.results import FilterResults


def test_filter_results_as_dict():
    """Test that FilterResults.as_dict returns the correct dictionary."""

    dummy_img = np.ones((2, 2, 2), dtype=np.uint8)
    fr = FilterResults(
        file_path="file.tif",
        gray_filter_value=1.5,
        min_size=10,
        small_object_filtered_img=dummy_img,
    )
    result = fr.as_dict()
    assert result["file_path"] == "file.tif"
    assert result["gray_filter_value"] == 1.5
    assert result["min_size"] == 10
    assert np.array_equal(result["small_object_filtered_img"], dummy_img)
