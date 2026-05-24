.. pycroglia documentation master file

Pycroglia
=========

.. image:: https://img.shields.io/pypi/v/pycroglia
   :target: https://pypi.org/project/pycroglia/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/CGK-Laboratory/pycroglia/blob/main/LICENSE
   :alt: License

**A Python toolkit for quantitative 3D morphology analysis of cells from fluorescence microscopy images.**

Originally based on the MATLAB tool *CellSelect-3DMorph*, Pycroglia reconstructs individual
cells voxel-by-voxel and extracts quantitative morphological descriptors.  It is built with a
robust, extensible architecture and supports both a **PyQt6 graphical interface** and a **library
mode** for automated/scripted workflows.

----

Key Features at a Glance
-------------------------

* **Multi-format I/O** — reads TIFF and LSM image stacks.
* **Interactive filtering** — per-slice Otsu threshold, small-object removal, morphological erosion.
* **Cell segmentation** — connected-component labeling with optional GMM-based cell splitting.
* **3D skeletonization** — Lee-Kashyap-Chu thinning (*slimskel3d*) or scikit-image fallback.
* **Morphological metrics** — volume, centroid, territorial volume, branch lengths, ramification index.
* **Parallel computation** — Qt thread pool and multiprocessing backends behind a unified ``Pool`` façade, orchestrated by a DAG-based scheduler.
* **Rich export** — Excel (XLSX), JSON, and 3D geometry (OBJ / PLY / VTP / VTK / VTI).
* **Wizard GUI** — step-by-step workflow: *File Selection → Filters → Segmentation → Cell Selection → Results Dashboard*.

----

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting-started
   workflow
   architecture
   CONTRIBUTING

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/modules
