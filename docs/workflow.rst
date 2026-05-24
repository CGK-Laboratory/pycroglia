Workflow and UI Guide
======================

The Pycroglia Graphical User Interface (GUI) is built as a step-by-step wizard.
This design ensures a logical progression from raw image data to final morphological analysis,
minimizing user error and providing immediate visual feedback at every stage.

The wizard is orchestrated by the ``ConfigurableMainStack`` and ``PageManager`` components,
which handle navigation (Next/Back) and state transfer between the major steps.

1. File Selection
-----------------

The entry point of the application. 

**UI Elements:**
* **File List:** Displays the list of selected files.
* **Add/Remove Buttons:** Browse and select TIFF or LSM files to process.

**Connection:**
The selected file paths are gathered and passed forward as state to the *Filter Editor* page.

2. Filter Editor
----------------

Here, users apply initial pre-processing to clean up the raw microscopy images.
This step is critical for separating foreground (cells) from background.

**UI Elements:**
* **Channel Selector:** If a multi-channel image is loaded (like an LSM file), you select which channel contains the cells of interest.
* **Per-Slice Otsu Thresholding:** A slider allows you to manually adjust the automatically calculated Otsu threshold. This applies 2D thresholding slice-by-slice, improving robustness against uneven Z-illumination.
* **Small Object Removal:** A spin box to define the minimum volume (in voxels) to keep. Smaller disconnected components are discarded as noise.
* **Morphological Erosion:** Tools to define a structuring element (e.g., 2D Diamond, radius) to erode the binary mask, helping to separate touching cells.
* **Image Viewer:** Provides real-time visual feedback of the filtering operations.

**Connection:**
The resulting binary masks and filter configurations for each file are sent to the *Segmentation Editor* page.

3. Segmentation Editor
----------------------

This page isolates individual cells from the filtered binary mask using connected-component labeling.

**UI Elements:**
* **Cell Viewer:** Displays the labeled components. Each color represents a distinct connected component.
* **GMM Splitting Tool:** If two cells are physically merged in the binary mask, you can select the combined object and apply a Gaussian Mixture Model (GMM) clustering to split it based on its morphology and intensity profile.
* **Action History/Rollback:** Allows you to undo splitting or merging operations, restoring the previous segmentation state.
* **Size/Property Tables:** Lists all detected cells with their initial voxel volumes.

**Connection:**
The fully separated and labeled cell masks are passed to the *Cell Selection* page.

4. Cell Selection
-----------------

A quality control step to finalize which cells will be analyzed.

**UI Elements:**
* **Filter Panel:** Global filters to exclude cells that touch the image borders (incomplete cells) or fall outside a specified size range.
* **Interactive Cell List:** Allows manual deselection of specific cells that are artifacts or otherwise unsuited for analysis.
* **Preview Viewer:** Renders the 3D footprint of the selected cells.

**Connection:**
The finalized list of individual cell masks is transmitted to the *Results Dashboard* for computation.

5. Results Dashboard
--------------------

The final stage where quantitative analysis is executed and visualized.

**UI Elements:**
* **Configuration Panel:** Input the physical dimensions of the image voxels (XY scale in µm, Z scale in µm). This is crucial for calculating accurate physical volumes and branch lengths.
* **Run Button:** Triggers the ``MetricsDAG`` parallel computation engine to analyze all selected cells.
* **Summary Table:** Displays aggregated metrics for the entire image (e.g., total territorial volume, average centroid distance).
* **Per-Cell Table:** Details metrics for every individual cell (volume, territorial volume, ramification index, branch count, endpoints, max/min/avg branch lengths).
* **3D PyVista Renderers:** Interactive 3D plots showing the cell surface, the skeleton, branch points, and convex hull.
* **Export Panel:** Buttons to save the data to Excel (.xlsx), JSON, or export the 3D geometries (OBJ, PLY, VTP).

**Under the Hood:**
When computation is triggered, the dashboard relies on a `QPool` facade to execute CPU-intensive tasks (skeletonization, convex hull calculation, distance transforms) concurrently across available threads without freezing the Qt GUI.
