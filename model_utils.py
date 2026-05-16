from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "home_win"
DEFAULT_DROP_COLS = {
    "game_id",
    "gameday",
    "home_score",
    "away_score",
    TARGET_COL,
}
MARKET_COLS = {"spread_line", "total_line"}


def get_feature_columns(matchups: pd.DataFrame, exclude_market: bool = False) -> list[str]:
    drop_cols = set(DEFAULT_DROP_COLS)
    if exclude_market:
        drop_cols.update(MARKET_COLS)
    return [col for col in matchups.columns if col not in drop_cols]


def split_feature_types(matchups: pd.DataFrame, feature_cols: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(matchups[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_model_pipeline(
    matchups: pd.DataFrame,
    feature_cols: list[str],
    max_iter: int = 2000,
) -> Pipeline:
    numeric_cols, categorical_cols = split_feature_types(matchups, feature_cols)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )
