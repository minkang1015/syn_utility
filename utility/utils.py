from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import re
from pandas.api.types import (
    is_numeric_dtype, is_bool_dtype,
    is_datetime64_any_dtype, is_datetime64tz_dtype,
    is_period_dtype, is_timedelta64_dtype,
    is_object_dtype, is_string_dtype, is_categorical_dtype,
)

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

def nondatetime_to_datetime(series: pd.Series, fmt: str) -> pd.Series:
    DIGIT_WIDTH = {"%Y": 4, "%y": 2, "%m": 2, "%d": 2, "%H": 2, "%M": 2, "%S": 2}

    if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series) or is_period_dtype(series):
        raise TypeError(f"Series dtype is already datetime/period: {series.dtype}. Use .dt.to_timestamp() or skip parsing.")
    if is_timedelta64_dtype(series):
        raise TypeError("Timedelta dtype is not supported for datetime parsing in this function.")
    if is_bool_dtype(series):
        raise TypeError("Boolean dtype is not supported for datetime parsing in this function.")

    if not (is_numeric_dtype(series) or is_object_dtype(series) or is_string_dtype(series) or is_categorical_dtype(series)):
        raise TypeError(f"Unsupported dtype for datetime parsing: {series.dtype}")

    tokens = re.findall(r"%[A-Za-z]", fmt)
    if not tokens:
        raise ValueError(f"Invalid fmt='{fmt}'. Must include strftime directives like %Y, %m, %d.")

    unsupported = [t for t in tokens if t not in DIGIT_WIDTH]
    if unsupported:
        raise ValueError(f"Unsupported directive(s) in fmt='{fmt}': {unsupported}. "
                         f"Supported: {sorted(DIGIT_WIDTH.keys())}")

    s = series
    if is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round().astype("Int64").astype("string")
    else:
        s = s.astype("string")
    s = s.str.strip()

    if re.sub(r"%[A-Za-z]", "", fmt) == "":
        width = sum(DIGIT_WIDTH[t] for t in tokens)
        s = s.str.replace(r"\D", "", regex=True).str.zfill(width)

    return pd.to_datetime(s, format=fmt, errors="coerce")

def encode_tables(df: pd.DataFrame, table_meta: dict) -> pd.DataFrame:
    df = df.copy()
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
        elif sdtype == "datetime":
            if is_datetime64_any_dtype(df[col]) or is_datetime64tz_dtype(df[col]) or is_period_dtype(df[col]):
                continue
            fmt = spec.get("datetime_format")
            if not fmt:
                raise ValueError(f"datetime_format missing for datetime column: {col}")
            df[col] = nondatetime_to_datetime(df[col], fmt)
        elif sdtype == "id":
            continue
    return df
