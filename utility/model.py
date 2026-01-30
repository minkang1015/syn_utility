from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

def fit_downstream_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    date_col: str,
    id_col: str,
    task: str,
    seed: int,
) -> dict:
    drop_cols = [c for c in [target_col, date_col, id_col] if c and c in train.columns]
    train_X = train.drop(columns=drop_cols, errors="ignore")
    train_y = train[target_col]
    test_X = test.drop(columns=drop_cols, errors="ignore")
    test_y = test[target_col]

    dt_cols = train_X.select_dtypes(include=["datetime64[ns]"]).columns
    for c in dt_cols:
        train_X[c] = train_X[c].dt.year * 100 + train_X[c].dt.month
        test_X[c]  = test_X[c].dt.year  * 100 + test_X[c].dt.month
    
    if task == "classification" and train_y.nunique() < 3:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            n_jobs=-1,
        )
        
        model.fit(train_X, train_y)
        proba = model.predict_proba(test_X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(test_y, proba)) if test_y.nunique() > 1 else np.nan,
            "accuracy": float(accuracy_score(test_y, pred)),
            "f1": float(f1_score(test_y, pred, zero_division=0)),
            "recall": float(recall_score(test_y, pred, zero_division=0)),
            "precision": float(precision_score(test_y, pred, zero_division=0)),
        }
        
    elif task == "classification" and train_y.nunique() >= 3:
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                random_state=seed,
                n_jobs=-1,
                num_class=train_y.nunique()
            )
            model.fit(train_X, train_y)
            
            proba = model.predict_proba(test_X)
            pred = model.predict(test_X)        
            try:
                auc_val = float(roc_auc_score(test_y, proba, multi_class='ovr', average='weighted'))
            except ValueError:
                auc_val = np.nan
            return {
                "auc": auc_val,
                "accuracy": float(accuracy_score(test_y, pred)),
                "f1": float(f1_score(test_y, pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(test_y, pred, average="weighted", zero_division=0)),
                "precision": float(precision_score(test_y, pred, average="weighted", zero_division=0)),
            }

    if task == "regression":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        return {
            "mse": float(mean_squared_error(test_y, pred_y)),
            "mae": float(mean_absolute_error(test_y, pred_y)),
            "mape": float(mean_absolute_percentage_error(test_y, pred_y)),
        }

    raise ValueError(f"Unknown task: {task}")