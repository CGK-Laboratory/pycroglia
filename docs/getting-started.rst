Getting Started
===============

This page walks you through the quickest paths to install and run Pycroglia.

----

Requirements
------------

* Python **3.11** or later
* A modern GPU is *not* required — all computation runs on CPU.
* Tested on Linux, macOS, and Windows.

----

Installation
------------

Option 1 — Pre-built executables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the platform-specific binary from the
`GitHub Releases <https://github.com/CGK-Laboratory/pycroglia/releases>`_ page.
No Python installation is needed.

Option 2 — uvx (recommended for end-users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `uv <https://docs.astral.sh/uv/>`_ and then run:

.. code-block:: bash

   uvx pycroglia

This installs an isolated copy and launches the GUI in one step.

Option 3 — pip
~~~~~~~~~~~~~~

.. code-block:: bash

   pip install pycroglia
   pycroglia

Option 4 — From source
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/CGK-Laboratory/pycroglia.git
   cd pycroglia
   uv run python -m pycroglia

Run the test suite:

.. code-block:: bash

   uv run pytest

----

Launching the GUI
-----------------

All installation options above launch the same PyQt6 wizard interface.
After the window opens, follow the five-step workflow:

1. **File Selection** — Add one or more TIFF or LSM files.
2. **Filter Editor** — Tune the per-slice Otsu threshold, remove small objects, and apply morphological erosion.
3. **Segmentation Editor** — Inspect labeled cells, split merged cells via GMM, and undo steps with rollback.
4. **Cell Selection** — Remove border cells, filter by voxel size, and verify the final cell set.
5. **Results Dashboard** — Set the physical voxel scale, run the full morphological analysis in parallel, preview 3D renderings, and export results.

----

Library Mode
------------

You can import ``pycroglia.core`` directly for scripted or batch workflows
without the GUI:

.. code-block:: python

   import tifffile
   import numpy as np
   from pycroglia.core.labeled_cells import LabeledCells, SkimageLabelingStrategy
   from pycroglia.core.branch_analysis import BranchAnalysis

   # 1. Load a binary 3-D image (Z, Y, X)
   img = tifffile.imread("my_cell.tif").astype(np.uint8)

   # 2. Label connected components
   cells = LabeledCells(img, SkimageLabelingStrategy())

   # 3. Analyse the first cell
   mask = cells.get_cell(1)
   centroid = np.array([mask.shape[0] / 2, mask.shape[1] / 2, mask.shape[2] / 2])

   analysis = BranchAnalysis(
       cell=mask,
       centroid=centroid,
       scale=0.207,    # µm/px XY
       zscale=1.0,     # µm/slice Z
       zslices=mask.shape[0],
   )
   results = analysis.compute()
   print("Branch count :", results["num_branchpoints"])
   print("Avg length   :", results["avg_branch_length"], "µm")

----

Next Steps
----------

* Read :doc:`workflow` for a detailed explanation of each pipeline stage.
* Read :doc:`architecture` for the system design and component diagram.
* Browse the :doc:`api/modules` reference for full API documentation.
