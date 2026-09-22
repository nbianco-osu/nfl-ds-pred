"""Export saved evaluation reports, without refitting or changing live picks."""
import json
from pathlib import Path

PRODUCTION = "home_win_advanced_automl_no_market_deep.joblib"
LABELS = {
    "home_win_advanced_automl_no_market_deep.joblib": "Advanced AutoML - deep, no market",
    "home_win_advanced_automl_no_market.joblib": "Advanced AutoML - no market",
    "home_win_advanced_automl.joblib": "Advanced AutoML - with market",
    "home_win_advanced_logreg_no_market.joblib": "Advanced logistic regression - no market",
    "home_win_advanced_logreg.joblib": "Advanced logistic regression - with market",
    "home_win_automl_no_market.joblib": "Basic AutoML - no market",
    "home_win_automl.joblib": "Basic AutoML - with market",
    "home_win_logreg.joblib": "Basic logistic regression - with market",
}
KERNEL_LABELS = {"rbf": "RBF", "matern_05": "Matern 0.5", "matern_15": "Matern 1.5",
                 "matern_25": "Matern 2.5", "rational_quadratic": "Rational Quadratic",
                 "linear": "Linear", "rbf_linear": "RBF + Linear", "matern_linear": "Matern 1.5 + Linear"}


def export(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    original = json.loads((root / "models/retraining_2026.json").read_text())
    gpr = json.loads((root / "models/gpr_comparison.json").read_text())
    rows = []
    for name, result in original.items():
        if name not in LABELS:
            continue
        rows.append({"id": name, "name": LABELS[name], "family": "Original",
                     "status": "Production" if name == PRODUCTION else "Candidate",
                     "market_inputs": "no_market" not in name, "kernel": None,
                     "noise_variance": None, "length_scale": None,
                     "training": f"{result['final_training_rows']:,} games in final fit",
                     **{key: result[key] for key in ("holdout_season", "test_rows", "accuracy", "log_loss", "roc_auc")}})
    for result in gpr["models"]:
        rows.append({"id": f"home_win_gpr_{result['name']}.joblib",
                     "name": "GPR - " + KERNEL_LABELS[result["name"]], "family": "GPR",
                     "status": "Experimental", "market_inputs": False,
                     "kernel": KERNEL_LABELS[result["name"]],
                     "noise_variance": result["selected"]["noise_level"],
                     "length_scale": None if result["name"] == "linear" else result["selected"]["length_scale"],
                     "training": f"{result['final_regression_rows']:,} regression + {result['final_calibration_rows']:,} calibration games",
                     **{key: result[key] for key in ("holdout_season", "test_rows", "accuracy", "log_loss", "roc_auc")}})
    if len(rows) != 16 or len({row['id'] for row in rows}) != 16:
        raise ValueError("Expected eight original and eight unique GPR models")
    output = {"gpr_trained_at": gpr["created_at"], "models": rows,
              "note": "Pre-refit retrospective evaluation. Not live pregame accuracy; no automatic promotion. "
              "Market-input and no-market models use different information. GPR uses a bounded recent-game window. "
              "Missing current features retain historical fallbacks."}
    path = root / "public_site/data/model_comparison.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    return output


if __name__ == "__main__":
    export()
