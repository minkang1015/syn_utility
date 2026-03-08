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

def load_rossmann(
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

    store_enc = encode_tables(store.fillna(0), meta["tables"]["store"])
    hist_enc = encode_tables(hist, meta["tables"]["historical"])
    final = hist_enc.merge(store_enc, on="Store", how="left").fillna(0)

    final = final.dropna()
    
    info = DatasetInfo(
        name="rossmann_subsampled",
        target_col="Customers",
        id_col="Store",
        date_col="Date",
        group_cols=None,
        task="regression",
    )
    return final, info

def load_walmart(
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

    tmp = depts_enc.merge(feats_enc, on=["Store", "Date", "IsHoliday"], how="left")
    final = tmp.merge(stores_enc, on="Store", how="left").fillna(0)

    final = final.dropna()
    
    info = DatasetInfo(
        name="walmart_subsampled",
        target_col="Weekly_Sales",
        id_col="Store",
        date_col="Date",
        group_cols=["Dept"],
        task="regression",
    )
    return final, info

def load_ptbxl(
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
    df_records = df_records.drop(columns=['record_id'])
    meta = read_json(data_root / "original" / "ptbxl" / "metadata.json")
    
    df_meta_encoded = encode_tables(df_meta, meta['tables']['ptbxl_database'])
    df_records_encoded = encode_tables(df_records, meta['tables']['records'])
    
    superclass = (
        df_records_encoded.groupby("ecg_id")["diagnostic_superclass"]
        .apply(lambda s: int(not (s == 15).any()))
        .rename("diagnostic_superclass")
        .reset_index()
    )

    df_features_agg = df_records_encoded.groupby("ecg_id").mean(numeric_only=True).reset_index()
    df_features_agg.drop(columns=['diagnostic_superclass'], inplace=True)
    tmp = pd.merge(df_meta_encoded, df_features_agg, on="ecg_id", how="inner")
    final = pd.merge(tmp, superclass, on="ecg_id", how="inner")

    final = final.dropna()
    
    info = DatasetInfo(
        name="ptbxl",
        target_col="diagnostic_superclass",
        id_col="ecg_id",
        date_col="recording_date",
        group_cols=None,
        task="classification",
    )
    return final, info

def load_freddiemac(
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

    hist_enc = encode_tables(hist, meta['tables']['hist'])
    orig_enc = encode_tables(orig, meta['tables']['orig'])

    df = hist_enc.merge(orig_enc, on='LOAN SEQUENCE NUMBER', how='left')
    df = df.sort_values(['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD'], kind="mergesort")
    
    g = df.groupby('LOAN SEQUENCE NUMBER', sort=False)
    d = pd.to_numeric(df['CURRENT LOAN DELINQUENCY STATUS'], errors="coerce").fillna(0)
    df["DLQ_BIN_T"] = (d > 0).astype("int8")
    df["DLQ_T+1"] = g["DLQ_BIN_T"].shift(-1).astype("Int8")  
    
    final = df.dropna(subset=["DLQ_T+1"]).copy()
    final["DLQ_T+1"] = final["DLQ_T+1"].astype("int8")

    final = final.dropna()
    
    info = DatasetInfo(
        name="freddiemac",
        target_col="DLQ_T+1",
        id_col="LOAN SEQUENCE NUMBER",
        date_col="MONTHLY REPORTING PERIOD",
        group_cols=None,
        task="classification",
    )
    return final, info

def load_fanniemae(
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

    df_enc = encode_tables(df, meta['tables']['crt'])

    ts_cols = ['Original Loan to Value Ratio (LTV)', 'Debt-To-Income (DTI)', 'Original Combined Loan to Value Ratio (CLTV)']
    df_feats = make_ts_features(
        df_enc, 
        group_cols=['Loan Identifier'], 
        ts_cols=ts_cols, 
        dates=dates, 
        date_col='Monthly Reporting Period'
    )

    df_feats = df_feats.sort_values(['Loan Identifier', 'Monthly Reporting Period'], kind="mergesort")
    g = df_feats.groupby('Loan Identifier', sort=False)
    
    d = pd.to_numeric(df_feats['Current Loan Delinquency Status'], errors="coerce").fillna(0)
    df_feats["DLQ_BIN_T"] = (d > 0).astype("int8")
    df_feats["DLQ_T+1"] = g["DLQ_BIN_T"].shift(-1).astype("Int8")  

    final = df_feats.dropna(subset=["DLQ_T+1"]).copy()
    final["DLQ_T+1"] = final["DLQ_T+1"].astype("int8")

    final = final.dropna()
    
    info = DatasetInfo(
        name="fanniemae",
        target_col="DLQ_T+1",
        id_col="Loan Identifier",
        date_col="Monthly Reporting Period",
        group_cols=None,
        task="classification",
    )
    return final, info

def load_berka(
    data_root: Path,
    split: str,
    method: str = "",
    run: str = "1",
    sample: str = "sample1",
    dates: list[int] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    dates = dates or [10, 30, 90]
    base = data_root / ("original" if split == "original" else "synthetic") / "berka"
    if split != "original":
        base = base / method / str(run) / str(sample)

    loan     = pd.read_csv(base / "loan.csv")
    trans    = pd.read_csv(base / "trans.csv")
    order    = pd.read_csv(base / "order.csv")
    account  = pd.read_csv(base / "account.csv")
    disp     = pd.read_csv(base / "disp.csv")
    client   = pd.read_csv(base / "client.csv")
    district = pd.read_csv(base / "district.csv")
    card     = pd.read_csv(base / "card.csv")

    meta = read_json(data_root / "original" / "berka" / "metadata.json")

    # 3) encode_tables 활용 (모든 테이블 인코딩)
    loan_enc     = encode_tables(loan, meta["tables"]["loan"])
    trans_enc    = encode_tables(trans, meta["tables"]["trans"])
    order_enc    = encode_tables(order, meta["tables"]["order"])
    account_enc  = encode_tables(account, meta["tables"]["account"])
    disp_enc     = encode_tables(disp, meta["tables"]["disp"])
    client_enc   = encode_tables(client, meta["tables"]["client"])
    district_enc = encode_tables(district, meta["tables"]["district"])
    card_enc     = encode_tables(card, meta["tables"]["card"])

    trans_ts = make_ts_features(
        df=trans_enc,
        group_cols=["account_id"],
        ts_cols=["amount", "balance"],
        dates=dates,
        date_col="trans_date",
        add_shift=False,
        add_rolling_mean=True,
        add_growth=True,
        growth_type="diff"
    )

    disp_owner = (disp_enc.sort_values(["account_id", "disp_type", "disp_id"])
                    .drop_duplicates("account_id", keep="first")[["account_id", "client_id", "disp_id"]])

    base = (loan_enc.merge(account_enc, on="account_id", how="left")
                .merge(disp_owner, on="account_id", how="left")
                .merge(client_enc, on="client_id", how="left", suffixes=("", "_cli"))
                .merge(district_enc, on="district_id", how="left", suffixes=("", "_dist")))

    base["account_age_days"] = base["loan_date"] - base["account_date"]
    base["y"] = base["status"].astype(int)

    feat_order = (order_enc.groupby("account_id", as_index=False)
                      .agg(
                          order_cnt=("order_id", "size"),
                          order_amt_sum=("amount", "sum"),
                          order_amt_mean=("amount", "mean"),
                          order_bank_to_nuniq=("bank_to", "nunique"),
                          order_ksymbol_nuniq=("k_symbol", "nunique"),
                      ))
    base = base.merge(feat_order, on="account_id", how="left")

    cj = (base[["loan_id", "disp_id", "loan_date"]]
          .merge(card_enc, on="disp_id", how="left"))
    cj = cj[cj["issued"] < cj["loan_date"]].copy()
    cj["card_age_days"] = cj["loan_date"] - cj["issued"]

    feat_card = (cj.groupby("loan_id", as_index=False)
                   .agg(
                       has_card=("card_id", lambda s: 1),
                       card_cnt=("card_id", "size"),
                       card_type_nuniq=("card_type", "nunique"),
                       card_age_min=("card_age_days", "min"),
                   ))
    base = base.merge(feat_card, on="loan_id", how="left")

    loan_key = base[["loan_id", "account_id", "loan_date"]].sort_values("loan_date")
    trans_ts = trans_ts.sort_values("trans_date")
    
    trans_final = pd.merge_asof(
        loan_key,
        trans_ts,
        by="account_id",
        left_on="loan_date",
        right_on="trans_date",
        direction="backward"
    )
    
    tj = trans_enc.merge(loan_key, on="account_id", how="inner")
    tj = tj[tj["trans_date"] < tj["loan_date"]].copy()
    
    df90 = tj[tj["trans_date"] >= (tj["loan_date"] - 90)].copy()
    cnt90 = df90.groupby(["loan_id", "trans_type"])["amount"].size().unstack(fill_value=0)
    cnt90.columns = [f"trans_type_cnt__{int(c)}__d90" for c in cnt90.columns]
    
    base = base.merge(trans_final.drop(columns=["account_id", "loan_date"]), on="loan_id", how="left")
    base = base.merge(cnt90.reset_index(), on="loan_id", how="left")

    agg_cols = [c for c in base.columns if c.startswith(("order_", "has_card", "card_", "Prev_", "trans_"))]
    base[agg_cols] = base[agg_cols].fillna(0)
    
    final = base.dropna(subset=["y"]).fillna(0)

    info = DatasetInfo(
        name="berka",
        target_col="y",
        id_col="loan_id",
        date_col="loan_date",
        group_cols=None,
        task="classification",
    )
    return final, info



DATASET_LOADERS = {
    "rossmann_subsampled": load_rossmann,
    "walmart_subsampled": load_walmart,
    "ptbxl": load_ptbxl,
    "freddiemac": load_freddiemac,
    "fanniemae": load_fanniemae,
}

def get_dataset_loader(name: str):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]