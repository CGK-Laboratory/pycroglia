# Pycroglia

**A Python-based toolkit for quantitative 3D microglia morphology analysis**

Pycroglia is a modern, open-source port of **CellSelect-3DMorph**, a
MATLAB-based tool originally designed to isolate and analyze cell
morphology from 3D fluorescence microscopy images.  By reconstructing
individual cells voxel by voxel, Pycroglia enables researchers to
extract quantitative morphological descriptors such as **cell
volume**, **territorial volume**, **ramification index**, **branch
length**, **number of branches**, and **endpoints**, among others.  It
builds upon the logic of the original MATLAB scripts but introduces a
robust and extensible Python architecture, supporting both GUI and
library modes for interactive and automated workflows.

---

## Installation and Usage

### 1. Prerequisites
This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency and environment management.  
Follow the official installation instructions for your operating system.

---

### 2. Clone the Repository

```bash
git clone https://github.com/CGK-Laboratory/pycroglia.git
cd pycroglia
```
---

### 3. Running the Application

#### Launch the Graphical Interface

To start the full GUI version of Pycroglia:

Finally while standing on the root of the project run the following (if the objective is to run the GUI)

```
uv run .
```
#### Use Pycroglia as a Library

If you want to work within a *Jupyter Notebook*, launch a notebook server connected to the project’s virtual environment:
```bash
uv run --with jupyter jupyter lab
```

### 4. Tests
Run all repository tests using *pytest*.

```
uv run pytest .
```

## Contributing
If you are interested in contributing to the project follow the following guidelines
[CONTRIBUTING](https://github.com/CGK-Laboratory/pycroglia/blob/main/docs/CONTRIBUTING.md)
