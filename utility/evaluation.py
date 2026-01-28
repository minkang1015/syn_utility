from __future__ import annotations
import pandas as pd
from utility.datasets import DatasetInfo
from utility.data_split import temporal_split_tmtr, temporal_split_tatr
from utility.model import fit_downstream_model

def evaluate(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    ratios: list[float],
    info: DatasetInfo,
    seed: int,
    test_ratio: float = 0.2,
    mode: str = "tmtr",
    granularity: int = 10,
) -> pd.DataFrame:
    rows = []
    ratio_name = "syn_ratio" if mode == "tmtr" else "aug_ratio"

    for r_req in ratios:
        if mode == "tmtr":
            train, test, cutoff, r_used = temporal_split_tmtr(
                real_df=real_df,
                syn_df=syn_df,
                id_col=info.id_col,
                group_cols=info.group_cols,
                date_col=info.date_col,
                test_ratio=test_ratio,
                syn_ratio=float(r_req),
                seed=seed,
                granularity=granularity,
            )
        elif mode == "tatr":
            train, test, cutoff, r_used = temporal_split_tatr(
                real_df=real_df,
                syn_df=syn_df,
                id_col=info.id_col,
                group_cols=info.group_cols,
                date_col=info.date_col,
                test_ratio=test_ratio,
                aug_ratio=float(r_req),
                seed=seed,
                granularity=granularity,
            )
        else:
            raise ValueError("mode must be one of: tmtr, tatr")

        metrics = fit_downstream_model(
            train=train,
            test=test,
            target_col=info.target_col,
            date_col=info.date_col,
            id_col=info.id_col,
            task=info.task,
            seed=seed,
        )

        rows.append({
            ratio_name: float(r_used),
            "cutoff": cutoff,
            **metrics,
        })

    return pd.DataFrame(rows).sort_values(ratio_name).reset_index(drop=True)
