def test_import():
    import pycroglia
    assert hasattr(pycroglia, "core")

if __name__ == "__main__":
    test_import()
