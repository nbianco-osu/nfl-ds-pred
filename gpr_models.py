"""Experimental Gaussian regression scores with chronological sigmoid calibration.

Binary-label regression is an approximation, not Gaussian process classification.
WhiteKernel models observation noise; it does not randomly flip saved picks.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


KERNELS = ("rbf", "matern_05", "matern_15", "matern_25", "rational_quadratic",
           "linear", "rbf_linear", "matern_linear")
FEATURES = (
    "diff_team_win_pct", "diff_team_point_diff_per_game", "diff_team_last_5_point_diff",
    "diff_team_rest_days", "diff_team_pbp_off_epa_per_play_avg",
    "diff_team_pbp_off_success_rate_avg", "diff_team_pbp_off_explosive_rate_avg",
    "diff_team_pbp_off_turnover_rate_avg", "diff_team_pbp_def_epa_allowed_per_play_avg",
    "diff_team_pbp_def_takeaway_rate_avg", "diff_team_pbp_prev_season_off_epa_per_play",
    "diff_team_pbp_prev_season_def_epa_allowed_per_play", "diff_team_qb_qb_passing_epa_avg",
    "diff_team_qb_qb_passing_cpoe_avg", "diff_team_qb_prev_season_qb_passing_epa",
    "diff_team_injury_out_count", "diff_team_injury_questionable_count",
    "diff_team_injury_qb_out", "diff_team_roster_active_count", "diff_team_roster_avg_years_exp",
)


def make_kernel(name, length_scale, noise, dimensions):
    linear = (1.0 / dimensions) * DotProduct(sigma_0=1.0, sigma_0_bounds="fixed")
    kernels = {
        "rbf": RBF(length_scale),
        "matern_05": Matern(length_scale, nu=0.5),
        "matern_15": Matern(length_scale, nu=1.5),
        "matern_25": Matern(length_scale, nu=2.5),
        "rational_quadratic": RationalQuadratic(length_scale, alpha=1.0),
        "linear": linear,
        "rbf_linear": RBF(length_scale) + linear,
        "matern_linear": Matern(length_scale, nu=1.5) + linear,
    }
    return kernels[name] + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")


class CalibratedGPRClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, kernel_name="rbf", noise_level=0.1, length_scale=5.0,
                 max_train_rows=1000, calibration_rows=256):
        self.kernel_name = kernel_name
        self.noise_level = noise_level
        self.length_scale = length_scale
        self.max_train_rows = max_train_rows
        self.calibration_rows = calibration_rows

    def fit(self, X, y):
        if self.max_train_rows < 2 or self.calibration_rows < 2 or self.noise_level <= 0:
            raise ValueError("Positive noise and at least two training/calibration rows required")
        frame = X.copy().reset_index(drop=True)
        frame["_target"] = np.asarray(y)
        frame = frame.sort_values(["season", "week"], kind="stable")
        if not set(frame._target.unique()).issubset({0, 1}) or frame._target.nunique() != 2:
            raise ValueError("Both binary outcomes are required")
        if len(frame) <= self.calibration_rows:
            raise ValueError("Insufficient rows for chronological calibration")
        boundary = frame.iloc[-self.calibration_rows]
        calibration = (frame.season > boundary.season) | (
            frame.season.eq(boundary.season) & frame.week.ge(boundary.week))
        train = frame.loc[~calibration].tail(self.max_train_rows)
        cal = frame.loc[calibration]
        if len(train) < 2 or cal._target.nunique() != 2:
            raise ValueError("Need earlier training games and both calibration outcomes")
        self.feature_cols_ = list(FEATURES)
        self.preprocessor_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        a = self.preprocessor_.fit_transform(train[self.feature_cols_])
        b = self.preprocessor_.transform(cal[self.feature_cols_])
        self.regressor_ = GaussianProcessRegressor(
            kernel=make_kernel(self.kernel_name, self.length_scale, self.noise_level, a.shape[1]),
            alpha=1e-8, optimizer=None, normalize_y=True, random_state=42)
        self.regressor_.fit(a, train._target)
        self.calibrator_ = LogisticRegression(C=1.0, random_state=42)
        self.calibrator_.fit(self.regressor_.predict(b).reshape(-1, 1), cal._target)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.training_rows_ = len(train)
        self.calibration_rows_ = len(cal)
        self.training_end_ = [int(train.season.iloc[-1]), int(train.week.iloc[-1])]
        self.calibration_start_ = [int(cal.season.iloc[0]), int(cal.week.iloc[0])]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "calibrator_")
        values = self.preprocessor_.transform(X[self.feature_cols_])
        scores = self.regressor_.predict(values)
        return self.calibrator_.predict_proba(scores.reshape(-1, 1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
