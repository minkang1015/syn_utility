# from __future__ import annotations
# from typing import Optional
# import numpy as np
# import pandas as pd
# from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# def _validate_datetime(df: pd.DataFrame, date_col: str, name: str):
#     assert date_col in df.columns, f"{name}: '{date_col}' not in columns."
#     assert (is_datetime64_any_dtype(df[date_col]) or is_datetime64tz_dtype(df[date_col])), (
#         f"{name}: '{date_col}' must be datetime dtype. current={df[date_col].dtype}"
#     )
#     assert df[date_col].notna().all(), f"{name}: '{date_col}' contains NaT."


# def _snap_k(total: int, ratio_req: float, clamp01: bool = True) -> tuple[int, float]:
#     """
#     Snap ratio to nearest feasible k/total using rounding.
#     If clamp01=True, ratio_req is clamped into [0,1].
#     """
#     r = float(ratio_req)
#     if clamp01:
#         r = max(0.0, min(1.0, r))
#     k = int(np.round(r * total))
#     k = max(0, min(total, k))
#     return k, (k / total) if total > 0 else 0.0


# def temporal_split_tmtr(
#     real_df: pd.DataFrame,
#     syn_df: pd.DataFrame,
#     id_col: str,
#     group_cols: Optional[list[str]],
#     date_col: str,
#     test_ratio: float = 0.2,
#     syn_ratio: float = 0.0,
#     seed: int = 0,
#     granularity: int = 10,
#     filtering: bool = False,
# ):
#     """
#     Data split for TMTR evaluation.
#     """
#     assert 0 < test_ratio < 1
#     syn_ratio = max(0.0, min(1.0, float(syn_ratio)))
#     group_cols = group_cols or []
#     granularity = int(granularity)

#     for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
#         assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
#         for gc in group_cols:
#             assert gc in df.columns, f"{name}: '{gc}' not in columns."
#         _validate_datetime(df, date_col, name)

#     rng = np.random.default_rng(seed)


#     uniq_dates = np.sort(real_df[date_col].unique())
#     nD = len(uniq_dates)
#     n_testD = max(1, int(round(nD * test_ratio)))
#     cutoff = uniq_dates[-n_testD]
    
#     if filtering:
#         grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
        
#         common_groups = real_df[grouping_cols].drop_duplicates().merge(
#             syn_df[grouping_cols].drop_duplicates(), on=grouping_cols, how="inner"
#         )
#         if len(common_groups) == 0:
#             raise ValueError("No common (group_cols + date_col) combinations in FULL dataset.")

#         real_df_filtered = real_df.merge(common_groups, on=grouping_cols, how="inner")
#         syn_df_filtered = syn_df.merge(common_groups, on=grouping_cols, how="inner")
        
#         real_train_pool = real_df_filtered[real_df_filtered[date_col] < cutoff]
#         real_test = real_df_filtered[real_df_filtered[date_col] >= cutoff]
#         syn_train_pool = syn_df_filtered[syn_df_filtered[date_col] < cutoff]

#     else:
#         real_train_pool = real_df[real_df[date_col] < cutoff]
#         real_test = real_df[real_df[date_col] >= cutoff]
#         syn_train_pool = syn_df[syn_df[date_col] < cutoff]

#     if len(real_train_pool) == 0:
#         raise ValueError("real_train_pool empty after cutoff.")
    
#     if len(real_test) == 0:
#         raise ValueError("real_test is empty. Filtering removed all test data (check if synthetic data covers test dates).")

#     real_ids = pd.Index(real_train_pool[id_col].unique())
#     syn_ids = pd.Index(syn_train_pool[id_col].unique())
    
#     if granularity > 1:
#         total_id = (min(len(real_ids), len(syn_ids)) // granularity) * granularity
#     else:
#         total_id = min(len(real_ids), len(syn_ids))

#     if total_id <= 0:
#         if not filtering and len(syn_ids) == 0 and syn_ratio == 0:
#              total_id = len(real_ids)
#              if granularity > 1:
#                  total_id = (total_id // granularity) * granularity
        
#         if total_id <= 0:
#              raise ValueError(f"Cannot form total_id (real={len(real_ids)}, syn={len(syn_ids)}).")

#     syn_id_n, syn_ratio_used = _snap_k(total_id, syn_ratio, clamp01=True)
#     real_id_n = total_id - syn_id_n

#     real_choice = rng.choice(real_ids.to_numpy(), size=real_id_n, replace=False) if real_id_n > 0 else []
#     syn_choice = rng.choice(syn_ids.to_numpy(), size=syn_id_n, replace=False) if syn_id_n > 0 else []

#     real_train = real_train_pool[real_train_pool[id_col].isin(real_choice)] if real_id_n > 0 else real_train_pool.iloc[0:0]
#     syn_train = syn_train_pool[syn_train_pool[id_col].isin(syn_choice)] if syn_id_n > 0 else syn_train_pool.iloc[0:0]

#     train = pd.concat([real_train, syn_train], ignore_index=True).sort_values([id_col, date_col], kind="mergesort")
#     real_test = real_test.sort_values([id_col, date_col], kind="mergesort")

#     return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, syn_ratio_used


# def temporal_split_tatr(
#     real_df: pd.DataFrame,
#     syn_df: pd.DataFrame,
#     id_col: str,
#     group_cols: Optional[list[str]],
#     date_col: str,
#     test_ratio: float = 0.2,
#     aug_ratio: float = 0.0,
#     seed: int = 0,
#     granularity: int = 10,
#     filtering: bool = False,
# ):
#     """
#     Data split for TATR evaluation.
#     """
#     assert 0 < test_ratio < 1
#     assert aug_ratio >= 0.0
#     group_cols = group_cols or []
#     granularity = int(granularity)

#     for name, df in [("real_df", real_df), ("syn_df", syn_df)]:
#         assert id_col in df.columns, f"{name}: '{id_col}' not in columns."
#         for gc in group_cols:
#             assert gc in df.columns, f"{name}: '{gc}' not in columns."
#         _validate_datetime(df, date_col, name)

#     rng = np.random.default_rng(seed)

#     uniq_dates = np.sort(real_df[date_col].unique())
#     nD = len(uniq_dates)
#     n_testD = max(1, int(round(nD * test_ratio)))
#     cutoff = uniq_dates[-n_testD]

#     if filtering:
#         grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
#         common_groups = real_df[grouping_cols].drop_duplicates().merge(
#             syn_df[grouping_cols].drop_duplicates(), on=grouping_cols, how="inner"
#         )
#         if len(common_groups) == 0:
#              raise ValueError("No common (group_cols + date_col) combinations in FULL dataset.")

#         real_df_filtered = real_df.merge(common_groups, on=grouping_cols, how="inner")
#         syn_df_filtered = syn_df.merge(common_groups, on=grouping_cols, how="inner")

#         real_train = real_df_filtered[real_df_filtered[date_col] < cutoff]
#         real_test = real_df_filtered[real_df_filtered[date_col] >= cutoff]
#         syn_train_pool = syn_df_filtered[syn_df_filtered[date_col] < cutoff]
#     else:
#         real_train = real_df[real_df[date_col] < cutoff]
#         real_test = real_df[real_df[date_col] >= cutoff]
#         syn_train_pool = syn_df[syn_df[date_col] < cutoff]

#     if len(real_train) == 0:
#         raise ValueError("real_train empty after cutoff.")
#     if len(real_test) == 0:
#         raise ValueError("real_test is empty. Filtering removed all test data.")

#     if len(syn_train_pool) == 0:
#         train = real_train.sort_values([id_col, date_col], kind="mergesort")
#         real_test = real_test.sort_values([id_col, date_col], kind="mergesort")
#         return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, 0.0

#     syn_ids = pd.Index(syn_train_pool[id_col].unique())

#     if granularity > 1:
#         base_n = (len(syn_ids) // granularity) * granularity
#     else:
#         base_n = len(syn_ids)
    
#     if base_n <= 0:
#         base_n = len(syn_ids)

#     syn_id_n = int(np.round(float(aug_ratio) * base_n))
#     syn_id_n = max(0, min(len(syn_ids), syn_id_n))
#     aug_ratio_used = (syn_id_n / base_n) if base_n > 0 else 0.0

#     syn_choice = rng.choice(syn_ids.to_numpy(), size=syn_id_n, replace=False) if syn_id_n > 0 else []
#     syn_train = syn_train_pool[syn_train_pool[id_col].isin(syn_choice)] if syn_id_n > 0 else syn_train_pool.iloc[0:0]

#     train = pd.concat([real_train, syn_train], ignore_index=True).sort_values([id_col, date_col], kind="mergesort")
#     real_test = real_test.sort_values([id_col, date_col], kind="mergesort")

#     return train.reset_index(drop=True), real_test.reset_index(drop=True), cutoff, aug_ratio_used

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
        
        valid_groups = real_train_pool[grouping_cols].drop_duplicates()
        syn_train_pool = syn_train_pool_raw.merge(valid_groups, on=grouping_cols, how="inner")
        
        # (만약 필터링 후 Syn 데이터가 0개가 되면 경고나 에러를 낼 수 있음
        if len(syn_train_pool) == 0 and syn_ratio > 0:
            raise ValueError("Filtering removed all synthetic training data.")
    else:
        syn_train_pool = syn_train_pool_raw

    # --- 이하 공통 로직 ---

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
             # Syn 데이터가 너무 적거나 필터링으로 다 날아간 경우
             # 실험을 계속하기 위해 total_id가 0이면 에러 처리
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

    # 1. Real Data (항상 원본 유지)
    real_train = real_df[real_df[date_col] < cutoff]
    real_test = real_df[real_df[date_col] >= cutoff]
    
    # 2. Syn Data 준비
    syn_train_pool_raw = syn_df[syn_df[date_col] < cutoff]

    # 3. Filtering 적용 (Syn 데이터만 정제)
    if filtering:
        grouping_cols = (group_cols + [date_col]) if group_cols else [date_col]
        
        # Real Train에 있는 조합만 유효
        valid_groups = real_train[grouping_cols].drop_duplicates()
        
        # Syn 필터링
        syn_train_pool = syn_train_pool_raw.merge(valid_groups, on=grouping_cols, how="inner")
    else:
        syn_train_pool = syn_train_pool_raw

    if len(real_train) == 0:
        raise ValueError("real_train empty after cutoff.")
    if len(real_test) == 0:
        raise ValueError("real_test is empty (Check cutoff logic).")

    # Augmentation 로직
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