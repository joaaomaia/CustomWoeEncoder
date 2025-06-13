import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from encoding import MissingHandler, FilesystemRegistry


def test_nan_preserved():
    df = pd.DataFrame({"a": [1, np.nan, 2]})
    mh = MissingHandler()
    out = mh.fit_transform(df)
    assert out["a"].isna().sum() == 1
    assert mh.mapping["a"] == "kept_as_nan"


def test_sentinel_replacement():
    df = pd.DataFrame({"a": [1, np.nan, 2]})
    mh = MissingHandler(sentinel=-999)
    out = mh.fit_transform(df)
    assert out["a"].iloc[1] == -999
    assert mh.mapping["a"] == "filled_with_-999"


def test_registry_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, np.nan]})
    mh = MissingHandler()
    mh.fit(df)
    reg = FilesystemRegistry(tmp_path)
    reg.save("v1", encoder="dummy", metadata=mh.to_dict(), missing_handler=mh)
    enc, loaded_mh, meta = reg.load("v1")
    assert meta["mapping"] == mh.to_dict()["mapping"]
    assert np.isnan(meta["sentinel"]) and np.isnan(mh.sentinel)
    assert isinstance(loaded_mh, MissingHandler)
    assert np.isnan(loaded_mh.sentinel)

