# Root entry-point shim for PyInstaller.
# PyInstaller analyses this top-level script; all real logic lives in
# pycroglia/__main__.py so the installed package continues to work normally.
from pycroglia.__main__ import main

if __name__ == "__main__":
    main()
