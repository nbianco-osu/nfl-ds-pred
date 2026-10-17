"""Production-shaped, fail-closed ensemble; release approval is deliberately separate."""
import numpy as np
from scipy.optimize import minimize


def logit(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def fit_weights(values, y, classification=False, cap=.35):
    values = np.asarray(values, dtype=float)
    y = np.asarray(y, dtype=float)
    n = values.shape[-1]
    if n < 3 or cap*n < 1 or values.shape[:-1] != y.shape or not np.isfinite(values).all() or not np.isfinite(y).all():
        raise ValueError('Invalid ensemble matrix or cap')
    if classification and (not np.isin(y, [0, 1]).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError('Invalid probability labels or members')
    uniform = np.ones(n)/n
    errors = (values-y[..., None]).reshape(-1, n)
    covariance = np.atleast_2d(np.cov(errors, rowvar=False))
    covariance /= max(float(np.trace(covariance)), 1e-9)

    def objective(w):
        prediction = values@w
        if classification:
            p = np.clip(prediction, 1e-6, 1-1e-6)
            risk = -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
        else:
            risk = np.mean((prediction-y)**2)/100
        return risk + .1*np.sum((w-uniform)**2) + .05*(w@covariance@w)

    result = minimize(objective, uniform, method='SLSQP', bounds=[(0, cap)]*n,
                      constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                      options={'maxiter': 1000, 'ftol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Constrained weight optimization failed')
    w = np.maximum(0, result.x); w /= w.sum()
    if w.max() > cap+1e-7:
        raise ValueError('Member concentration limit violated')
    return w


class RobustMatchupEnsemble:
    """One artifact supplies winner, paired scores, ranges and disagreement."""
    def __init__(self, winner_models, winner_weights, score_models, score_weights,
                 calibrator, residuals, feature_cols, model_version):
        self.winner_models = winner_models
        self.winner_weights = np.asarray(winner_weights)
        self.score_models = score_models
        self.score_weights = np.asarray(score_weights)
        self.calibrator = calibrator
        self.residuals = np.asarray(residuals)
        self.feature_cols = list(feature_cols)
        self.model_version = model_version
        self.classes_ = np.array([0, 1])
        if self.residuals.ndim != 2 or self.residuals.shape[1] != 2 or not len(self.residuals) or not np.isfinite(self.residuals).all():
            raise ValueError('Paired finite calibration residuals required')
        for models, weights in [(winner_models, self.winner_weights), (score_models, self.score_weights)]:
            if len(models) != len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise ValueError('Invalid member weights')

    def _inputs(self, frame):
        if frame.columns.duplicated().any() or not set(self.feature_cols) <= set(frame.columns):
            raise ValueError('Missing or duplicate input columns')
        x = frame[self.feature_cols].astype(float)
        if np.isinf(x.to_numpy()).any() or x.isna().mean(axis=1).gt(.4).any():
            raise ValueError('Nonfinite or excessively missing inputs; use an explicitly approved fallback')
        return x

    def _winner_members(self, x):
        p = np.column_stack([m.predict_proba(x)[:, 1] for m in self.winner_models])
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid member probabilities')
        return p

    def _score_members(self, x):
        scores = np.stack([np.maximum(0, m.predict(x)) for m in self.score_models], axis=-1)
        if scores.shape != (len(x), 2, len(self.score_models)) or not np.isfinite(scores).all():
            raise ValueError('Invalid member scores')
        return scores

    def predict_proba(self, frame):
        pooled = self._winner_members(self._inputs(frame))@self.winner_weights
        p = self.calibrator.predict_proba(logit(pooled).reshape(-1, 1))[:, 1]
        if not np.isfinite(p).all(): raise ValueError('Invalid calibrated probabilities')
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.column_stack([1-p, p])

    def predict(self, frame):
        score = self._score_members(self._inputs(frame))@self.score_weights
        return np.maximum(0, score+self.residuals.mean(axis=0))

    def predict_details(self, frame):
        x = self._inputs(frame)
        members = self._winner_members(x)
        base = self._score_members(x)@self.score_weights
        # Preserve within-game home/away residual dependence; no independent draws.
        draws = np.maximum(0, base[:, None, :]+self.residuals[None, :, :])
        total = draws.sum(axis=2); margin = draws[:, :, 0]-draws[:, :, 1]
        return dict(model_version=self.model_version, home_probability=self.predict_proba(frame)[:, 1],
                    scores=self.predict(frame), winner_member_sd=members.std(axis=1),
                    score_lower=np.quantile(draws, .1, axis=1), score_upper=np.quantile(draws, .9, axis=1),
                    total_lower=np.quantile(total, .1, axis=1), total_upper=np.quantile(total, .9, axis=1),
                    margin_lower=np.quantile(margin, .1, axis=1), margin_upper=np.quantile(margin, .9, axis=1),
                    score_implied_home_win=(margin > 0).mean(axis=1),
                    note='Historical paired-residual 80% marginal ranges, not guaranteed joint coverage. '
                         'Score-implied probability and calibrated classifier probability are separate estimates.')
