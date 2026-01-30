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
    filtering: bool = False,
):
    """
    Data split for TMTR evaluation.
    filtering=True: Filters SYN data to keep only (group + date) pairs that exist in REAL data.
                    REAL data is kept 100% intact (Train & Test).
    """
    assert 0 < test_ratio < 1
    syn_ratio = max(0.0, min(1.0, float(syn_ratio)))
    group_cols = group_cols or []
    granularity = int(granularity)

    for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
        assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
        for gc in group_cols:
            assert gc in df.columns, f"{name}: '{gc}' not in columns."
        _validate_datetime(df, date_col, name)

    rng = np.random.default_rng(seed)

    uniq_dates = np.sort(real_df[date_col].unique())
    nD = len(uniq_dates)
    n_testD = max(1, int(round(nD * test_ratio)))
    cutoff = uniq_dates[-n_testD]
    
    real_train_pool = real_df[real_df[date_col] < cutoff]
    real_test = real_df[real_df[date_col] >= cutoff]
    syn_train_pool_raw = syn_df[syn_df[date_col] < cutoff]

    if filtering:
        grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
        
        valid_groups = real_train_pool[grouping_cols].copy()
        valid_groups[date_col] = pd.to_datetime(valid_groups[date_col]).dt.normalize()
        valid_groups = valid_groups.drop_duplicates()
        
        syn_train_pool_tmp = syn_train_pool_raw.copy()
        syn_train_pool_tmp['_compare_date'] = pd.to_datetime(syn_train_pool_tmp[date_col]).dt.normalize()
        
        syn_train_pool = syn_train_pool_tmp.merge(
            valid_groups.rename(columns={date_col: '_compare_date'}),
            on=(group_cols + ['_compare_date']) if group_cols else ['_compare_date'],
            how="inner"
        ).drop(columns=['_compare_date'])
        
        if len(syn_train_pool) == 0 and syn_ratio > 0:
            raise ValueError("Filtering (Daily Resolution) removed all synthetic training data.")
    else:
        syn_train_pool = syn_train_pool_raw

    if len(real_train_pool) == 0:
        raise ValueError("real_train_pool empty after cutoff.")
    if len(real_test) == 0:
        raise ValueError("real_test is empty (Check cutoff logic).")

    real_ids = pd.Index(real_train_pool[id_col].unique())
    syn_ids = pd.Index(syn_train_pool[id_col].unique())
    
    if granularity > 1:
        total_id = (min(len(real_ids), len(syn_ids)) // granularity) * granularity
    else:
        total_id = min(len(real_ids), len(syn_ids))

    if total_id <= 0:
        if not filtering and len(syn_ids) == 0 and syn_ratio == 0:
             total_id = len(real_ids)
             if granularity > 1:
                 total_id = (total_id // granularity) * granularity
        
        if total_id <= 0:
             raise ValueError(f"Cannot form total_id (real={len(real_ids)}, syn={len(syn_ids)}).")

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
    filtering: bool = False,
):
    """
    Data split for TATR evaluation.
    filtering=True: Filters SYN data to keep only (group + date) pairs that exist in REAL data.
    """
    assert 0 < test_ratio < 1
    assert aug_ratio >= 0.0
    group_cols = group_cols or []
    granularity = int(granularity)

    for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
        assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
        for gc in group_cols:
            assert gc in df.columns, f"{name}: '{gc}' not in columns."
        _validate_datetime(df, date_col, name)

    rng = np.random.default_rng(seed)

    uniq_dates = np.sort(real_df[date_col].unique())
    nD = len(uniq_dates)
    n_testD = max(1, int(round(nD * test_ratio)))
    cutoff = uniq_dates[-n_testD]

    real_train = real_df[real_df[date_col] < cutoff]
    real_test = real_df[real_df[date_col] >= cutoff]
    
    syn_train_pool_raw = syn_df[syn_df[date_col] < cutoff]

    if filtering:
        grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
        valid_groups = real_train[grouping_cols].copy()
        valid_groups[date_col] = pd.to_datetime(valid_groups[date_col]).dt.normalize()
        valid_groups = valid_groups.drop_duplicates()
        
        syn_train_pool_tmp = syn_train_pool_raw.copy()
        syn_train_pool_tmp['_compare_date'] = pd.to_datetime(syn_train_pool_tmp[date_col]).dt.normalize()
        
        syn_train_pool = syn_train_pool_tmp.merge(
            valid_groups.rename(columns={date_col: '_compare_date'}),
            on=(group_cols + ['_compare_date']) if group_cols else ['_compare_date'],
            how="inner"
        ).drop(columns=['_compare_date'])
        
    else:
        syn_train_pool = syn_train_pool_raw

    if len(real_train) == 0:
        raise ValueError("real_train empty after cutoff.")
    if len(real_test) == 0:
        raise ValueError("real_test is empty (Check cutoff logic).")

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