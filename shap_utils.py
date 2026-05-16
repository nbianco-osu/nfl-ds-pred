from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def feature_names(model_pipeline) -> list[str]:
    preprocessor = model_pipeline.named_steps["preprocess"]
    return [name.replace("num__", "").replace("cat__", "") for name in preprocessor.get_feature_names_out()]


def transformed_matrix(model_pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    values = model_pipeline.named_steps["preprocess"].transform(rows)
    return pd.DataFrame(values, columns=feature_names(model_pipeline), index=rows.index)


def positive_class_shap_values(explainer, transformed_rows: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(transformed_rows)
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def make_tree_explainer(model_pipeline):
    estimator = model_pipeline.named_steps["model"]
    return shap.TreeExplainer(estimator)


def explain_rows(artifact: dict, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_pipeline = artifact["model"]
    transformed = transformed_matrix(model_pipeline, rows[artifact["feature_cols"]])
    explainer = make_tree_explainer(model_pipeline)
    values = positive_class_shap_values(explainer, transformed)
    shap_frame = pd.DataFrame(values, columns=transformed.columns, index=rows.index)
    return shap_frame, transformed


def global_importance(artifact: dict, rows: pd.DataFrame, max_rows: int = 750, random_state: int = 42) -> pd.DataFrame:
    sample = rows.sample(min(max_rows, len(rows)), random_state=random_state) if len(rows) > max_rows else rows
    shap_frame, transformed = explain_rows(artifact, sample)
    importance = (
        shap_frame.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance["mean_value"] = importance["feature"].map(transformed.mean().to_dict())
    return importance


def local_top_features(
    artifact: dict,
    row: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    shap_frame, transformed = explain_rows(artifact, row)
    first_index = shap_frame.index[0]
    result = pd.DataFrame(
        {
            "feature": shap_frame.columns,
            "shap_value": shap_frame.loc[first_index].values,
            "feature_value": transformed.loc[first_index].values,
        }
    )
    result["abs_shap"] = result["shap_value"].abs()
    result["impact"] = np.where(result["shap_value"].ge(0), "Helps home team", "Helps away team")
    return result.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
