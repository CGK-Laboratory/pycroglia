![Pycroglia Banner](images/banner.png)

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
Pycroglia is available on [PyPI](https://pypi.org/project/pycroglia/)
and can be installed or executed using
[uv](https://github.com/astral-sh/uv) or the standard `pip` tool.

### Prerequisites

* [uv](https://docs.astral.sh/uv/getting-started/installation/) (Recommended): 
* Python 3.10 or later


### Option 1 — Install with pip

You can install Pycroglia using pip directly from PyPI:

```bash
pip install pycroglia
```
and to run it 

```bash
pycroglia
```

### Option 2 — Install with uv

If you prefer to use `uv`, which provides faster and isolated package management:

```bash
uv pip install pycroglia
```
and to run it 

```bash
pycroglia
```


### Option 3 — Run directly (recommended)

You can run Pycroglia without installing it globally, using `uvx`:

```bash
uvx pycroglia
```
This automatically downloads and runs the latest released version from PyPI in an isolated environment.

You can also specify a particular version:
```bash
uvx pycroglia==0.0.2
```

### Option 4 — From source

If you cloned the repository and want to run it locally:

```bash
git clone https://github.com/CGK-Laboratory/pycroglia/pycroglia.git
cd pycroglia
uv run main.py
```
and for running the test suite

```bash
uv run pytest
```

#### Use Pycroglia from a Jupyter Notebook

If you want to work within a *Jupyter Notebook*, launch a notebook
server connected to the project’s virtual environment:

```bash
uv run --with jupyter jupyter lab
```

## Contributing
If you are interested in contributing to the project follow the following guidelines
[CONTRIBUTING](https://github.com/CGK-Laboratory/pycroglia/blob/main/docs/CONTRIBUTING.md)
