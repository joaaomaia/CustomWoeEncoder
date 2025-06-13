import sys, os
sys.path.insert(0, os.path.abspath("src"))

import pandas as pd
from encoding import EncodingManager, FilesystemRegistry


def test_registry_roundtrip(tmp_path):
    df = pd.DataFrame({"feat": ["a", "b", "a", "c"], "y": [0, 1, 0, 1]})
    manager = EncodingManager("woe", categorical_cols=["feat"], drop_original=False)
    manager.fit(df[["feat"]], df["y"])
    reg = FilesystemRegistry(tmp_path)
    reg.save("v1", manager.encoder, {"ok": True}, missing_handler=manager.missing_handler)
    enc, mh, meta = reg.load("v1")
    assert hasattr(enc, "transform")
    assert isinstance(mh, type(manager.missing_handler))
