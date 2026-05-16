from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from model_utils import TARGET_COL, build_model_pipeline, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoML-style model search for NFL matchup prediction.")
    parser.add_argument("--input", type=Path, default=Path("data/nfl_matchups_1999_2025.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-name", default="home_win_automl.joblib")
    parser.add_argument("--holdout-season", type=int, default=None)
    parser.add_argument("--validation-season", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-market", action="store_true")
    parser.add_argument(
        "--search-jobs",
        type=int,
        default=1,
        help="Parallel jobs for the outer AutoML search. Keep at 1 on Windows to avoid process explosion.",
    )
    parser.add_argument(
        "--complexity",
        choices=["standard", "deep"],
        default="standard",
        help="Use a larger search space for slower, more complex tuning.",
    )
    return parser.parse_args()


def search_space(random_state: int, complexity: str) -> list[dict[str, object]]:
    if complexity == "standard":
        return [
            {
                "model": [LogisticRegression(max_iter=3000, random_state=random_state)],
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            {
                "model": [RandomForestClassifier(random_state=random_state, n_jobs=-1)],
                "model__n_estimators": [200, 400, 800],
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
            {
                "model": [ExtraTreesClassifier(random_state=random_state, n_jobs=-1)],
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
            {
                "model": [HistGradientBoostingClassifier(random_state=random_state)],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_iter": [100, 200, 400],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ]

    return [
        {
            "model": [LogisticRegression(max_iter=3000, random_state=random_state)],
            "model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [RandomForestClassifier(random_state=random_state, n_jobs=-1)],
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [4, 6, 8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 4, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.25, 0.5, 0.75],
            "model__class_weight": [None, "balanced_subsample"],
        },
        {
            "model": [ExtraTreesClassifier(random_state=random_state, n_jobs=-1)],
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [4, 6, 8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 4, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.25, 0.5, 0.75],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [HistGradientBoostingClassifier(random_state=random_state)],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_iter": [100, 200, 300],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [10, 20, 30, 50],
            "model__l2_regularization": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
        },
    ]


def make_time_split(train: pd.DataFrame, validation_season: int) -> PredefinedSplit:
    split = [-1 if season < validation_season else 0 for season in train["season"]]
    if 0 not in split or -1 not in split:
        raise ValueError("Validation split must contain both training and validation rows.")
    return PredefinedSplit(split)


def main() -> None:
    args = parse_args()
    matchups = pd.read_csv(args.input)
    holdout_season = args.holdout_season or int(matchups["season"].max())
    validation_season = args.validation_season or holdout_season - 1

    train = matchups[matchups["season"] < holdout_season].copy()
    test = matchups[matchups["season"] == holdout_season].copy()
    if train.empty or test.empty:
        raise ValueError("Need at least one training season and one holdout season.")

    feature_cols = get_feature_columns(matchups, exclude_market=args.exclude_market)
    base_pipeline = build_model_pipeline(matchups, feature_cols)
    cv = make_time_split(train, validation_season)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=search_space(args.random_state, args.complexity),
        n_iter=args.n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=args.random_state,
        n_jobs=args.search_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(train[feature_cols], train[TARGET_COL])

    probabilities = search.best_estimator_.predict_proba(test[feature_cols])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "holdout_season": holdout_season,
        "validation_season": validation_season,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "best_cv_neg_log_loss": float(search.best_score_),
        "accuracy": float(accuracy_score(test[TARGET_COL], predictions)),
        "log_loss": float(log_loss(test[TARGET_COL], probabilities)),
        "roc_auc": float(roc_auc_score(test[TARGET_COL], probabilities)),
        "best_params": {key: str(value) for key, value in search.best_params_.items()},
        "exclude_market": bool(args.exclude_market),
        "complexity": args.complexity,
    }
    artifact = {
        "model": search.best_estimator_,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_input": str(args.input),
        "automl": True,
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / args.model_name
    metrics_path = model_path.with_suffix(".metrics.json")
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved AutoML model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Best CV neg log loss: {metrics['best_cv_neg_log_loss']:.3f}")
    print(f"Holdout season: {holdout_season}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Log loss: {metrics['log_loss']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
