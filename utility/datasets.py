from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
from utility.utils import read_json, encode_tables
from utility.features import make_ts_features


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    target_col: str
    id_col: str
    date_col: str
    group_cols: Optional[list[str]]
    task: str 


def load_rossmann_flat(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [7, 14]
    base = data_root / ("original" if split == "original" else "synthetic") / "rossmann_subsampled"
    if split != "original":
        base = base / method / str(run) / str(sample)

    hist_cols = [
        "Store", "Date", "Customers",
        "Open", "Promo", "SchoolHoliday",
        "DayOfWeek", "StateHoliday",
    ]
    store_cols = [
        "Store",
        "StoreType", "Assortment", "PromoInterval",
        "CompetitionDistance", "Promo2",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear",
    ]

    hist = pd.read_csv(base / "historical.csv", usecols=hist_cols, parse_dates=["Date"])
    store = pd.read_csv(base / "store.csv", usecols=store_cols)

    meta = read_json(data_root / "original" / "rossmann_subsampled" / "metadata.json")

    hist_ts = make_ts_features(
        hist,
        group_cols=["Store"],
        ts_cols=["Customers"],
        dates=dates,
        date_col="Date",
        add_shift=False,
        add_rolling_mean=True,
        add_growth=True,
        growth_type="pct",
    )

    store_enc = encode_tables(store.fillna(0), meta["tables"]["store"])
    hist_enc = encode_tables(hist_ts, meta["tables"]["historical"])
    final = hist_enc.merge(store_enc, on="Store", how="left").fillna(0)

    info = DatasetInfo(
        name="rossmann_subsampled",
        target_col="Customers",
        id_col="Store",
        date_col="Date",
        group_cols=None,
        task="regression",
    )
    return final, info


def load_walmart_flat(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [1, 2]
    base = data_root / ("original" if split == "original" else "synthetic") / "walmart_subsampled"
    if split != "original":
        base = base / method / str(run) / str(sample)

    depts_cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    stores_cols = ["Store", "Type", "Size"]
    feats_cols = [
        "Store", "Date",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "IsHoliday",
    ]

    depts = pd.read_csv(base / "depts.csv", usecols=depts_cols, parse_dates=["Date"])
    stores = pd.read_csv(base / "stores.csv", usecols=stores_cols)
    feats = pd.read_csv(base / "features.csv", usecols=feats_cols, parse_dates=["Date"])

    meta = read_json(data_root / "original" / "walmart_subsampled" / "metadata.json")

    stores_enc = encode_tables(stores.fillna(0), meta["tables"]["stores"])
    feats_enc = encode_tables(feats, meta["tables"]["features"])
    depts_enc = encode_tables(depts, meta["tables"]["depts"])

    feats_ts = make_ts_features(
        feats_enc,
        group_cols=["Store"],
        ts_cols=["Temperature", "Fuel_Price", "CPI", "Unemployment"],
        dates=dates,
        date_col="Date",
        add_shift=True,
        add_rolling_mean=True,
        add_growth=True,
        growth_type="pct",
    )

    depts_ts = make_ts_features(
        depts_enc,
        group_cols=["Store", "Dept"],
        ts_cols=["Weekly_Sales"],
        dates=dates,
        date_col="Date",
        add_shift=False,
        add_rolling_mean=True,
        add_growth=True,
        growth_type="pct",
    )

    tmp = depts_ts.merge(feats_ts, on=["Store", "Date", "IsHoliday"], how="left")
    final = tmp.merge(stores_enc, on="Store", how="left").fillna(0)

    info = DatasetInfo(
        name="walmart_subsampled",
        target_col="Weekly_Sales",
        id_col="Store",
        date_col="Date",
        group_cols=["Dept"],
        task="regression",
    )
    return final, info


DATASET_LOADERS = {
    "rossmann_subsampled": load_rossmann_flat,
    "walmart_subsampled": load_walmart_flat,
}


def get_dataset_loader(name: str):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]
