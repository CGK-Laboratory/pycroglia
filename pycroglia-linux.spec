# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for pycroglia (Linux).
# Build with: uv run pyinstaller pycroglia-linux.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[str(Path(".").resolve())],
    binaries=[],
    datas=[
        ("pycroglia/assets", "assets"),
    ],
    hiddenimports=[
        # PyQt6 / PyQtGraph internals that PyInstaller may miss
        "PyQt6.sip",
        "PyQt6.QtPrintSupport",
        "PyQt6.QtSvg",
        "pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt6",
        "pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt6",
        "pyqtgraph.imageview.ImageViewTemplate_pyqt6",
        # VTK / PyVista
        "vtkmodules",
        "vtkmodules.all",
        "vtkmodules.util.numpy_support",
        "vtkmodules.util.data_model",
        "pyvista",
        # SciPy / scikit-image sub-modules
        "scipy._lib.messagestream",
        "scipy.special._ufuncs",
        "skimage.filters.rank.core_cy_3d",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="pycroglia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    icon=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
