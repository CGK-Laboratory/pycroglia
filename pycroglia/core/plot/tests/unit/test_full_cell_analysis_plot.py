import numpy as np
from pycroglia.core.full_cell_analysis import FullCellAnalysis
from pycroglia.core.plot.full_cell_analysis import FullCellAnalysisPlot
from scipy.io import loadmat
from pathlib import Path

TEST_DIR = Path(__file__).parent  # folder where this test lives
FILES_DIR = TEST_DIR / "files"  # adjust if files/ is elsewhere

def indices_to_mask(
        cell_indices: np.ndarray, img_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.zeros(img_shape, dtype=bool)
    mask.ravel()[cell_indices] = True
    return mask.astype(np.uint8)

def test_plot(tmp_path: Path):
    """Test FullCellAnalysisPlot correctly saves 3D convex hull figures.

    Asserts:
        - Each generated figure is saved as a non-empty image file.
        - The number of saved files matches the number of plotted figures.
        - Output files follow the expected sequential naming pattern.
    """
    full_mg = loadmat(FILES_DIR/"fullmg.mat")["FullMg"].ravel()
    img_shape = (39,1024,1024)
    masks = []
    for cell in full_mg:
        indices = cell.ravel().astype(int) - 1  # MATLAB 1-based → Python 0-based
        mask = indices_to_mask(indices, (img_shape[2], img_shape[1], img_shape[0]))
        masks.append(mask)

    voxscale = 1.0000e-03
    fca = FullCellAnalysis(masks, voxscale)
    results = fca.compute()

    plotter = FullCellAnalysisPlot(results, masks)

    save_base = tmp_path / "cell_plot"
    saved_paths = plotter.save(save_base, fmt="png")
    
    assert len(saved_paths) == len(plotter.figs)
    for path in saved_paths:
        assert path.exists(), f"Expected {path} to exist."
        assert path.suffix == ".png"
        assert path.stat().st_size > 0  # non-empty file

    expected_files = [tmp_path / f"cell_plot_{i+1}.png" for i in range(len(plotter.figs))]
    assert saved_paths == expected_files
