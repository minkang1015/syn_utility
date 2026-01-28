from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def _validate_datetime(df: pd.DataFrame, date_col: str, name: str):
    assert date_col in df.columns, f"{name}: '{date_col}' not in columns."
    assert (is_datetime64_any_dtype(df[date_col]) or is_datetime64tz_dtype(df[date_col])), (
        f"{name}: '{date_col}' must be datetime dtype. current={df[date_col].dtype}"
    )
    assert df[date_col].notna().all(), f"{name}: '{date_col}' contains NaT."


def _snap_k(total: int, ratio_req: float, clamp01: bool = True) -> tuple[int, float]:
    """
    Snap ratio to nearest feasible k/total using rounding.
    If clamp01=True, ratio_req is clamped into [0,1].
    """
    r = float(ratio_req)
    if clamp01:
        r = max(0.0, min(1.0, r))
    k = int(np.round(r * total))
    k = max(0, min(total, k))
    return k, (k / total) if total > 0 else 0.0


def temporal_split_tmtr(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    id_col: str,
    group_cols: Optional[list[str]],
    date_col: str,
    test_ratio: float = 0.2,
    syn_ratio: float = 0.0,
    seed: int = 0,
    granularity: int = 10,
):
    """
    Data split for TMTR evaluation:
      - train total unique IDs: total_id = floor(min(#real_ids,#syn_ids)/granularity)*granularity
      - syn_id_n = round(syn_ratio * total_id), real_id_n = total_id - syn_id_n
      - test: REAL only, temporal holdout
    Returns: train_df, test_df, cutoff_date, syn_ratio_used
    """
    assert 0 < test_ratio < 1
    group_cols = group_cols or []
    granularity = int(granularity)

    for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
        assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
        for gc in group_cols:
            assert gc in df.columns, f"{name}: '{gc}' not in columns."
        _validate_datetime(df, date_col, name)

    rng = np.random.default_rng(seed)

    grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
    common_groups = real_df[grouping_cols].drop_duplicates().merge(
        syn_df[grouping_cols].drop_duplicates(), on=grouping_cols, how="inner"
    )
    if len(common_groups) == 0:
        raise ValueError("No common (group_cols + date_col) combinations between real and synthetic.")

    real_df = real_df.merge(common_groups, on=grouping_cols, how="inner")
    syn_df = syn_df.merge(common_groups, on=grouping_cols, how="inner")

    uniq_dates = np.sort(real_df[date_col].unique())
    nD = len(uniq_dates)
    n_testD = max(1, int(round(nD * test_ratio)))
    cutoff = uniq_dates[-n_testD]

    real_train_pool = real_df[real_df[date_col] < cutoff]
    real_test = real_df[real_df[date_col] >= cutoff]
    syn_train_pool = syn_df[syn_df[date_col] < cutoff]

    if len(real_train_pool) == 0:
        raise ValueError("real_train_pool empty after cutoff.")
    if len(syn_train_pool) == 0:
        raise ValueError("syn_train_pool empty before cutoff (tmtr requires syn IDs available).")

    real_ids = pd.Index(real_train_pool[id_col].unique())
    syn_ids = pd.Index(syn_train_pool[id_col].unique())

    if granularity > 1:
        total_id = (min(len(real_ids), len(syn_ids)) // granularity) * granularity
    else:
        total_id = min(len(real_ids), len(syn_ids))

    if total_id <= 0:
        raise ValueError(f"Cannot form total_id (real_ids={len(real_ids)}, syn_ids={len(syn_ids)}).")

    syn_id_n, syn_ratio_used = _snap_k(total_id, syn_ratio, clamp01=True)
    real_id_n = total_id - syn_id_n

    real_choice = rng.choice(real_ids.to_numpy(), size=real_id_n, replace=False) if real_id_n > 0 else []
    syn_choice = rng.choice(syn_ids.to_numpy(), size=syn_id_n, replace=False) if syn_id_n > 0 else []

    real_train = real_train_pool[real_train_pool[id_col].isin(real_choice)] if real_id_n > 0 else real_train_pool.iloc[0:0]
    syn_train = syn_train_pool[syn_train_pool[id_col].isin(syn_choice)] if syn_id_n > 0 else syn_train_pool.iloc[0:0]

    train = pd.concat([real_train, syn_train], ignore_index=True).sort_values([id_col, date_col], kind="mergesort")
    real_test = real_test.sort_values([id_col, date_col], kind="mergesort")

    return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, syn_ratio_used


def temporal_split_tatr(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    id_col: str,
    group_cols: Optional[list[str]],
    date_col: str,
    test_ratio: float = 0.2,
    aug_ratio: float = 0.0,
    seed: int = 0,
    granularity: int = 10,
):
    """
    Data split for TATR evaluation:
      - real train: all
      - base_n = floor(#syn_ids/granularity)*granularity
      - syn_id_n = round(aug_ratio * base_n) (clamped to available syn IDs)
      - test: REAL only, temporal holdout
    Returns: train_df, test_df, cutoff_date, aug_ratio_used (= syn_id_n/base_n)
    """
    assert 0 < test_ratio < 1
    assert aug_ratio >= 0.0
    assert aug_ratio <= 1.0
    group_cols = group_cols or []
    granularity = int(granularity)

    for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
        assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
        for gc in group_cols:
            assert gc in df.columns, f"{name}: '{gc}' not in columns."
        _validate_datetime(df, date_col, name)

    rng = np.random.default_rng(seed)

    grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
    common_groups = real_df[grouping_cols].drop_duplicates().merge(
        syn_df[grouping_cols].drop_duplicates(), on=grouping_cols, how="inner"
    )
    if len(common_groups) == 0:
        raise ValueError("No common (group_cols + date_col) combinations between real and synthetic.")

    real_df = real_df.merge(common_groups, on=grouping_cols, how="inner")
    syn_df = syn_df.merge(common_groups, on=grouping_cols, how="inner")

    uniq_dates = np.sort(real_df[date_col].unique())
    nD = len(uniq_dates)
    n_testD = max(1, int(round(nD * test_ratio)))
    cutoff = uniq_dates[-n_testD]

    real_train = real_df[real_df[date_col] < cutoff]
    real_test = real_df[real_df[date_col] >= cutoff]
    syn_train_pool = syn_df[syn_df[date_col] < cutoff]

    if len(real_train) == 0:
        raise ValueError("real_train empty after cutoff.")
    if len(syn_train_pool) == 0:
        train = real_train.sort_values([id_col, date_col], kind="mergesort")
        real_test = real_test.sort_values([id_col, date_col], kind="mergesort")
        return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, 0.0

    syn_ids = pd.Index(syn_train_pool[id_col].unique())

    if granularity > 1:
        base_n = (len(syn_ids) // granularity) * granularity
    else:
        base_n = len(syn_ids)
    if base_n <= 0:
        base_n = len(syn_ids)

    syn_id_n = int(np.round(float(aug_ratio) * base_n))
    syn_id_n = max(0, min(len(syn_ids), syn_id_n))
    aug_ratio_used = (syn_id_n / base_n) if base_n > 0 else 0.0

    syn_choice = rng.choice(syn_ids.to_numpy(), size=syn_id_n, replace=False) if syn_id_n > 0 else []
    syn_train = syn_train_pool[syn_train_pool[id_col].isin(syn_choice)] if syn_id_n > 0 else syn_train_pool.iloc[0:0]

    train = pd.concat([real_train, syn_train], ignore_index=True).sort_values([id_col, date_col], kind="mergesort")
    real_test = real_test.sort_values([id_col, date_col], kind="mergesort")

    return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, aug_ratio_used
