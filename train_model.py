from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from model_utils import TARGET_COL, build_model_pipeline, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save an NFL matchup win model.")
    parser.add_argument("--input", type=Path, default=Path("data/nfl_matchups_1999_2025.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-name", default="home_win_logreg.joblib")
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=None,
        help="Season used as the test set. Defaults to the latest season in the dataset.",
    )
    parser.add_argument(
        "--exclude-market",
        action="store_true",
        help="Drop spread_line and total_line so the model does not use betting market features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matchups = pd.read_csv(args.input)
    holdout_season = args.holdout_season or int(matchups["season"].max())

    train = matchups[matchups["season"] < holdout_season].copy()
    test = matchups[matchups["season"] == holdout_season].copy()
    if train.empty or test.empty:
        raise ValueError("Need at least one training season and one holdout season.")

    feature_cols = get_feature_columns(matchups, exclude_market=args.exclude_market)
    model = build_model_pipeline(matchups, feature_cols)
    model.fit(train[feature_cols], train[TARGET_COL])

    probabilities = model.predict_proba(test[feature_cols])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "holdout_season": holdout_season,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(test[TARGET_COL], predictions)),
        "log_loss": float(log_loss(test[TARGET_COL], probabilities)),
        "roc_auc": float(roc_auc_score(test[TARGET_COL], probabilities)),
        "exclude_market": bool(args.exclude_market),
    }

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_input": str(args.input),
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / args.model_name
    metrics_path = model_path.with_suffix(".metrics.json")
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Holdout season: {holdout_season}")
    print(f"Rows: train={len(train):,}, test={len(test):,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Log loss: {metrics['log_loss']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
