def test_import():
    import pycroglia
    assert hasattr(pycroglia, "__version__")

if __name__ == "__main__":
    test_import()
