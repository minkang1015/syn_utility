from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip() != ""]

def parse_ratios_from_bin(bin_size: float, ratio_max: float = 1.0) -> list[float]:
    bin_size = float(bin_size)
    ratio_max = float(ratio_max)
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    out = []
    x = 0.0
    tol = bin_size / 1000
    while x <= ratio_max + tol:
        out.append(round(float(x), 10))
        x += bin_size
    if abs(out[-1] - ratio_max) > 1e-6:
        out.append(round(float(ratio_max), 10))
    return out


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cast_numeric(series: pd.Series, rep: str | None):
    s = pd.to_numeric(series, errors="coerce")
    if rep and rep.startswith("Int"):
        return s.round().astype("Int32")
    return s


def boolean_to_int(series: pd.Series):
    if series.dtype == object:
        s = series.astype(str).str.lower().map({"true": 1, "false": 0})
        return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")


def categorical_to_code(series: pd.Series):
    codes, _ = pd.factorize(series, sort=True)
    return codes.astype("int16")


def encode_tables(df: pd.DataFrame, table_meta: dict) -> pd.DataFrame:
    cols_meta = table_meta.get("columns", {})
    for col, spec in cols_meta.items():
        if col not in df.columns:
            continue
        sdtype = spec.get("sdtype")
        if sdtype == "numerical":
            df[col] = cast_numeric(df[col], spec.get("computer_representation"))
        elif sdtype == "boolean":
            df[col] = boolean_to_int(df[col])
        elif sdtype == "categorical":
            df[col] = categorical_to_code(df[col])
        elif sdtype == "id":
            continue
    return df