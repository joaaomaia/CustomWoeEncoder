"""Simple memory benchmark comparing WOE and OneHot encoders."""

import numpy as np
import pandas as pd
import psutil
from encoding import EncodingManager


def bench(n_rows: int = 1000, n_unique: int = 1000):
    df = pd.DataFrame({
        "feat": np.random.choice([f"cat_{i}" for i in range(n_unique)], size=n_rows),
        "y": np.random.randint(0, 2, size=n_rows),
    })

    mem_before = psutil.Process().memory_info().rss
    woe = EncodingManager("woe", categorical_cols=["feat"])
    woe.fit_transform(df[["feat"]], df["y"])
    mem_woe = psutil.Process().memory_info().rss - mem_before

    mem_before = psutil.Process().memory_info().rss
    oh = EncodingManager("onehot", columns=["feat"])
    oh.fit_transform(df[["feat"]], df["y"])
    mem_onehot = psutil.Process().memory_info().rss - mem_before

    print(f"WoE memory delta: {mem_woe} bytes")
    print(f"OneHot memory delta: {mem_onehot} bytes")


if __name__ == "__main__":
    bench()
