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

def load_ptbxl_flat(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [5, 10, 15, 20]
    base = data_root / ("original" if split == "original" else "synthetic") / "ptbxl"
    if split != "original":
        base = base / method / str(run) / str(sample)

    df_meta = pd.read_csv(base / "ptbxl_database.csv")
    df_meta = df_meta.drop(columns=['report'])
    df_records = pd.read_csv(base / "records.csv")
    meta = read_json(data_root / "original" / "ptbxl" / "metadata.json")
    

    df_meta_encoded = encode_tables(df_meta, meta['tables']['ptbxl_database'])
    df_records_encoded = encode_tables(df_records, meta['tables']['records'])
    
    ts_cols = [f"lead_{i}" for i in range(12)]
    df_features = make_ts_features(
        df=df_records_encoded,
        group_cols=["ecg_id"],
        ts_cols=ts_cols,
        date_col="t",
        dates=dates,
        add_shift=False,
        add_rolling_mean=True,
        add_rolling_std=True,
        add_growth=True,
        growth_type="diff"
    )
    
    df_features_agg = df_features.groupby("ecg_id").mean(numeric_only=True).reset_index()
    final = pd.merge(df_meta_encoded, df_features_agg, on="ecg_id", how="inner")

    info = DatasetInfo(
        name="ptbxl",
        target_col="heart_axis",
        id_col="ecg_id",
        date_col="recording_date",
        group_cols=None,
        task="classification",
    )
    return final, info

def load_freddiemac_flat(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [1, 2]
    base = data_root / ("original" if split == "original" else "synthetic") / "freddiemac"
    if split != "original":
        base = base / method / str(run) / str(sample)

    hist = pd.read_csv(base / "hist.csv")
    orig = pd.read_csv(base / "orig.csv")
    meta = read_json(data_root / "original" / "freddiemac" / "metadata.json")

    # 1. Encoding
    hist_enc = encode_tables(hist, meta['tables']['hist'])
    orig_enc = encode_tables(orig, meta['tables']['orig'])

    # 2. Time-series features
    ts_cols = ['CURRENT INTEREST RATE', 'CURRENT NON-INTEREST BEARING UPB', 'ESTIMATED LOAN TO VALUE (ELTV)']
    hist_feats = make_ts_features(
        hist_enc, 
        group_cols=['LOAN SEQUENCE NUMBER'], 
        ts_cols=ts_cols, 
        dates=dates, 
        date_col='MONTHLY REPORTING PERIOD'
    )

    # 3. Merge and Target Creation (DLQ_T+1)
    df = hist_feats.merge(orig_enc, on='LOAN SEQUENCE NUMBER', how='left')
    df = df.sort_values(['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD'], kind="mergesort")
    
    g = df.groupby('LOAN SEQUENCE NUMBER', sort=False)
    d = pd.to_numeric(df['CURRENT LOAN DELINQUENCY STATUS'], errors="coerce").fillna(0)
    df["DLQ_BIN_T"] = (d > 0).astype("int8")
    df["DLQ_T+1"] = g["DLQ_BIN_T"].shift(-1).astype("Int8")  
    
    final = df.dropna(subset=["DLQ_T+1"]).copy()
    final["DLQ_T+1"] = final["DLQ_T+1"].astype("int8")

    info = DatasetInfo(
        name="freddiemac",
        target_col="DLQ_T+1",
        id_col="LOAN SEQUENCE NUMBER",
        date_col="MONTHLY REPORTING PERIOD",
        group_cols=None,
        task="classification",
    )
    return final, info

def load_fanniemae_flat(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [1, 2]
    base = data_root / ("original" if split == "original" else "synthetic") / "fanniemae"
    if split != "original":
        base = base / method / str(run) / str(sample)

    df = pd.read_csv(base / "fanniemae.csv")
    meta = read_json(data_root / "original" / "fanniemae" / "metadata_filtered.json")

    # 1. Encoding
    df_enc = encode_tables(df, meta['tables']['crt'])

    # 2. Time-series features
    ts_cols = ['Original Loan to Value Ratio (LTV)', 'Debt-To-Income (DTI)', 'Original Combined Loan to Value Ratio (CLTV)']
    df_feats = make_ts_features(
        df_enc, 
        group_cols=['Loan Identifier'], 
        ts_cols=ts_cols, 
        dates=dates, 
        date_col='Monthly Reporting Period'
    )

    # 3. Target Creation (DLQ_T+1)
    df_feats = df_feats.sort_values(['Loan Identifier', 'Monthly Reporting Period'], kind="mergesort")
    g = df_feats.groupby('Loan Identifier', sort=False)
    
    d = pd.to_numeric(df_feats['Current Loan Delinquency Status'], errors="coerce").fillna(0)
    df_feats["DLQ_BIN_T"] = (d > 0).astype("int8")
    df_feats["DLQ_T+1"] = g["DLQ_BIN_T"].shift(-1).astype("Int8")  

    final = df_feats.dropna(subset=["DLQ_T+1"]).copy()
    final["DLQ_T+1"] = final["DLQ_T+1"].astype("int8")

    info = DatasetInfo(
        name="fanniemae",
        target_col="DLQ_T+1",
        id_col="Loan Identifier",
        date_col="Monthly Reporting Period",
        group_cols=None,
        task="classification",
    )
    return final, info


DATASET_LOADERS = {
    "rossmann_subsampled": load_rossmann_flat,
    "walmart_subsampled": load_walmart_flat,
    "ptbxl": load_ptbxl_flat,
    "freddiemac": load_freddiemac_flat,
    "fanniemae": load_fanniemae_flat,
}


def get_dataset_loader(name: str):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]
