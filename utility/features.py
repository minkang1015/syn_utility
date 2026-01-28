from __future__ import annotations
from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd


def make_ts_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    ts_cols: Sequence[str],
    dates: Iterable[int] = (1, 2, 4),
    date_col: str = "Date",
    sort: bool = True,
    prefix: str = "Prev",
    min_periods: int = 1,
    add_shift: bool = True,
    add_rolling_mean: bool = True,
    add_rolling_std: bool = False,
    add_rolling_min: bool = False,
    add_rolling_max: bool = False,
    add_growth: bool = True,
    growth_type: str = "pct",  # {"pct","log","diff"}
    growth_clip: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()
    required = list(group_cols) + [date_col] + list(ts_cols)
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if sort:
        out = out.sort_values(list(group_cols) + [date_col], kind="mergesort")

    for col in ts_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    g = out.groupby(list(group_cols), sort=False)

    for col in ts_cols:
        for k in dates:
            k = int(k)
            if k <= 0:
                raise ValueError(f"dates must be positive integers; got {k}")

            if add_shift:
                out[f"{prefix}_{k}_period_{col}"] = g[col].shift(k).fillna(-999)

            if add_rolling_mean:
                out[f"{prefix}_{k}_periods_{col}_Mean"] = g[col].transform(
                    lambda x: x.shift(1).rolling(window=k, min_periods=min_periods).mean().fillna(-999)
                )
            if add_rolling_std:
                out[f"{prefix}_{k}_periods_{col}_Std"] = g[col].transform(
                    lambda x: x.shift(1).rolling(window=k, min_periods=min_periods).std().fillna(-999)
                )
            if add_rolling_min:
                out[f"{prefix}_{k}_periods_{col}_Min"] = g[col].transform(
                    lambda x: x.shift(1).rolling(window=k, min_periods=min_periods).min().fillna(-999)
                )
            if add_rolling_max:
                out[f"{prefix}_{k}_periods_{col}_Max"] = g[col].transform(
                    lambda x: x.shift(1).rolling(window=k, min_periods=min_periods).max().fillna(-999)
                )

            if add_growth:
                prev1 = g[col].shift(1)
                prev1k = g[col].shift(1 + k)

                if growth_type == "pct":
                    denom = prev1k.replace(0, np.nan)
                    growth = (prev1 - prev1k) / denom
                    name = f"{prefix}_{k}_periods_{col}_GrowthPct"
                elif growth_type == "log":
                    a = prev1.where(prev1 > 0)
                    b = prev1k.where(prev1k > 0)
                    growth = np.log(a) - np.log(b)
                    name = f"{prefix}_{k}_periods_{col}_GrowthLog"
                elif growth_type == "diff":
                    growth = (prev1 - prev1k)
                    name = f"{prefix}_{k}_periods_{col}_GrowthDiff"
                else:
                    raise ValueError(f"Unknown growth_type: {growth_type}")

                if growth_clip is not None:
                    growth = growth.clip(lower=-growth_clip, upper=growth_clip)

                out[name] = growth.fillna(-999)

    return out.reset_index(drop=True) if sort else out
