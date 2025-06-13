import sys, os
sys.path.insert(0, os.path.abspath("src"))

from encoding import ColumnNameManager


def test_column_mapping():
    mgr = ColumnNameManager()
    mgr.add("a", ["a__enc1", "a__enc2"])
    assert mgr.original_to_encoded("a") == ["a__enc1", "a__enc2"]
    assert mgr.encoded_to_original("a__enc2") == "a"
    assert "a__enc1" in mgr.original_to_encoded("a")
