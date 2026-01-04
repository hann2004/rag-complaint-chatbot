import importlib


def test_import_task1_processing():
    # Smoke test: module should import without side effects raising errors
    mod = importlib.import_module("task1_processing")
    assert hasattr(mod, "TARGET_PRODUCTS")
    assert len(mod.TARGET_PRODUCTS) > 0
