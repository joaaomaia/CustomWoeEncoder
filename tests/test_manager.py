import sys, os
sys.path.insert(0, os.path.abspath("src"))

import pandas as pd
from encoding import EncodingManager
from encoding.encoders import WOEGuard


def test_woe_exposed():
    df = pd.DataFrame({"feat": ["a", "b", "a", "c"], "y": [0, 1, 0, 1]})
    manager = EncodingManager("woe", categorical_cols=["feat"], drop_original=False)
    Xt = manager.fit_transform(df[["feat"]], df["y"])

    enc = WOEGuard(["feat"])
    enc.fit(df[["feat"]], df["y"])
    direct = enc.transform(df[["feat"]])

    pd.testing.assert_frame_equal(Xt.reset_index(drop=True), direct.reset_index(drop=True))
