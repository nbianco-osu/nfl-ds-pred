"""Add eight noise-aware GPR candidates without modifying production artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from gpr_models import CalibratedGPRClassifier, FEATURES, KERNELS
from model_utils import get_feature_columns


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p >= 0.5)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/nfl_matchups_1999_2026_advanced.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--holdout-season", type=int, default=2026)
    parser.add_argument("--max-train-rows", type=int, default=1000)
    args = parser.parse_args()
    data = pd.read_csv(args.input).sort_values(["season", "week", "game_id"])
    data = data.loc[data.home_score.notna() & data.away_score.notna() & data.home_score.ne(data.away_score)]
    if data.game_id.duplicated().any():
        raise ValueError("Duplicate games")
    if not data.home_win.eq((data.home_score > data.away_score).astype(int)).all():
        raise ValueError("Inconsistent outcome labels")
    # Keep paired team columns for the shared schedule feature builder. The
    # estimator itself uses only the fixed numeric contrasts in FEATURES.
    cols = get_feature_columns(data, exclude_market=True)
    train = data.loc[data.season < args.holdout_season - 1]
    validation = data.loc[data.season == args.holdout_season - 1]
    history = data.loc[data.season < args.holdout_season]
    test = data.loc[data.season == args.holdout_season]
    full = data.loc[data.season <= args.holdout_season]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Need training, validation, and holdout seasons")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"created_at": stamp, "status": "experimental; no automatic promotion",
              "validation_season": args.holdout_season - 1, "holdout_season": args.holdout_season,
              "holdout_rows": len(test), "models": [],
              "limitations": "Binary-label Gaussian regression with sigmoid calibration, not GPC. "
              "Recent-window fits; WhiteKernel observation noise is not a random winner flip. "
              "2026 retrospective evaluation is small, not a prospective performance claim. "
              "Unavailable current features retain historical fallback."}
    predictions = test[["game_id", "home_win"]].copy()
    for name in KERNELS:
        trials = []
        for noise in (0.1, 0.5):
            for scale in (3.0, 8.0):
                candidate = CalibratedGPRClassifier(name, noise, scale, args.max_train_rows)
                candidate.fit(train[cols], train.home_win)
                loss = log_loss(validation.home_win, candidate.predict_proba(validation[cols]), labels=[0, 1])
                trials.append({"noise_level": noise, "length_scale": scale, "validation_log_loss": float(loss)})
        best = min(trials, key=lambda row: row["validation_log_loss"])
        model = CalibratedGPRClassifier(name, best["noise_level"], best["length_scale"], args.max_train_rows)
        model.fit(history[cols], history.home_win)
        p = model.predict_proba(test[cols])[:, 1]
        predictions[name] = p
        result = {"name": name, "holdout_season": args.holdout_season, "test_rows": len(test),
                  **metrics(test.home_win, p), "search_trials": trials, "selected": best,
                  "evaluation": "Pre-refit retrospective holdout; tuning on preceding season only"}
        model.fit(full[cols], full.home_win)
        result.update(final_regression_rows=model.training_rows_, final_calibration_rows=model.calibration_rows_,
                      final_training_end=model.training_end_, calibration_start=model.calibration_start_,
                      kernel=str(model.regressor_.kernel_))
        path = args.model_dir / f"home_win_gpr_{name}.joblib"
        artifact = {"model": model, "feature_cols": cols, "target_col": "home_win", "metrics": result,
                    "trained_at": stamp, "model_version": f"gpr_{name}_{stamp}", "training_input": str(args.input),
                    "training_through_season": args.holdout_season, "experimental": True}
        if path.exists():
            archive = args.model_dir / "archive" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / path.name)
        joblib.dump(artifact, path, compress=3)
        reloaded = joblib.load(path)
        np.testing.assert_allclose(model.predict_proba(test[cols]), reloaded["model"].predict_proba(test[cols]))
        path.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report["models"].append(result)
        print(f"{name}: 2026 log loss {result['log_loss']:.4f}; saved {path}", flush=True)
    report["saved_model_count_excluding_smoke"] = len([p for p in args.model_dir.glob("*.joblib") if "smoke" not in p.name])
    (args.model_dir / "gpr_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(args.model_dir / "gpr_holdout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
