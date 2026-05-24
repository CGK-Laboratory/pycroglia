System Architecture
===================

Pycroglia is designed with a strict separation between its computational core and its PyQt6 graphical user interface. This separation enables both a responsive GUI and an accessible library mode for script-based execution.

Core Module (``pycroglia.core``)
--------------------------------

The core module houses all image processing algorithms, geometry calculations, and parallel execution infrastructure. It has no dependencies on PyQt.

Compute Infrastructure
~~~~~~~~~~~~~~~~~~~~~~

To handle intensive operations like 3D skeletonization and Euclidean distance transforms without blocking the UI, Pycroglia uses a flexible multiprocessing architecture.

* **``Computable`` Interface:** All major analytical tasks (e.g., ``BranchAnalysis``, ``Centroids``, ``FullCellAnalysis``, ``TerritorialVolume``) inherit from a ``Computable`` abstract base class, exposing a ``compute() -> dict`` method.
* **Pool Facade:** The ``Pool`` class acts as a unified frontend for job submission. It abstracts the underlying execution backend.
* **Backends:**
    * ``QPool``: Uses Qt's ``QThreadPool`` and ``QRunnable``. Ideal for GUI execution as it natively integrates with Qt's event loop and signals.
    * ``MPPool``: Uses Python's standard ``multiprocessing.Pool``. Used when Pycroglia is imported as a library without a Qt QApplication.

Metrics DAG
~~~~~~~~~~~

Because some metrics depend on the results of others (e.g., branch lengths require the centroid to be calculated first to determine distance order), the computation is orchestrated by a ``MetricsDAG`` (Directed Acyclic Graph). 

The DAG submits independent tasks (Centroid, Territorial Volume, Full Cell Analysis) to the pool simultaneously. Once the Centroid task completes, it dynamically submits the dependent ``BranchAnalysis`` tasks. Progress is tracked via atomic ``QCounter`` barriers.

Algorithm Highlights
~~~~~~~~~~~~~~~~~~~~

* **Skeletonization (``slimskel3d``):** Pycroglia contains a native Python/NumPy implementation of the Lee-Kashyap-Chu 3D thinning algorithm, matching the exact behavior of the MATLAB `Skeleton3D` implementation used in *CellSelect-3DMorph*. It recursively erodes border voxels while preserving Euler topology and endpoints.
* **Branch Analysis:** Once skeletonized, the core uses 3x3x3 26-connected convolutional kernels to identify branch points and endpoints. Branches are isolated by path-finding from endpoints to the cell centroid. Arc length is calculated using Euclidean distance between ordered voxels.
* **I/O (``pycroglia.core.io``):** 
    * Handles parsing of TIFF and LSM stacks.
    * Outputs exporters (Excel, JSON) map the heterogeneous ``Computable`` result dictionaries to structured output formats.

UI Module (``pycroglia.ui``)
----------------------------

The UI is built on PyQt6 and follows a structured model-view-controller (MVC) pattern adapted for wizard-style navigation.

* **Widgets:** Custom PyQt widgets (found in ``pycroglia.ui.widgets``) handle display logic. E.g., ``FileSelectionEditor``, ``FilterEditorStack``, ``ResultsDashboardStack``.
* **Controllers / State:** Pure state managers (e.g., ``SegmentationEditorState``, ``ResultsDashboardState``) handle the business logic of UI interactions, decoupled from the views. They emit ``pyqtSignal``s when data changes, which the views observe to update themselves.
* **Wizard Navigation:** A ``PageManager`` wraps major widgets into ``BasePage`` objects. It orchestrates the flow of data forward. For instance, when clicking "Next" on the Filter page, the ``PageManager`` extracts the binary masks using ``get_state()`` and injects them into the Segmentation page using ``set_data()``.
