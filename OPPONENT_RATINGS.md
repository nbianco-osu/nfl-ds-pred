# Opponent-Adjusted EPA

This is an open, reproducible research alternative, not FTN DVOA. No proprietary
data, DVOA formula, or restricted source is used.

Fit `EPA = intercept + offense effect + defense allowance effect` jointly using
ridge regression. Opponent-pair means are weighted by play count, equivalent to
the per-play squared-error objective up to a constant. The intercept is unpenalized;
team effects have a fixed 200-play ridge penalty, shrinking small samples toward
zero. This choice has not been tuned or validated for predictive improvement.
NFLverse EPA already models game situation; we do not fit a second situation
model or claim DVOA's success definitions or adjustments.

- Offensive rating: EPA/play above the fitted baseline against average opposition.
- Defensive rating: negative defense allowance effect; positive is better.
- Net rating: offense plus defense EPA prevented. This is not a percentage.
- Sack differential: sacks made minus allowed per observed game; rate differential
  uses each side's dropbacks separately.
- Big-play differential: gains minus gains allowed per observed game. Passes of
  at least 20 yards and runs of at least 10 count; rates use scrimmage plays.

Only pass/run plays from completed regular-season games are used. Missing EPA
does not become zero; rate/count fields with missing source observations remain
missing. Game counts are observed PBP coverage, not proof of complete play capture.
Missing completed game IDs are reported. Kneels, spikes and no-play records are
excluded by play type. No home-field coefficient, special teams rating, recency
weighting or explicit garbage-time filter is included.

Every matchup's features freeze before the first date of its week and exclude
all same-week games. Ratings restart each season. Week 1 and unobserved teams
remain missing, not neutral observations. Latest rankings use all completed
games and must never be joined onto historical matchups. Archived source data
may have revisions: a pregame calculation cutoff is not evidence the data was
available at that time.

Run `.\.venv\Scripts\python.exe opponent_ratings.py --season 2026`.
Outputs in `data/`: `opponent_features_2026.csv` (272 matchup rows),
`opponent_rankings_2026.csv` (latest team ratings), and
`opponent_ratings_2026.json` (weekly ratings, source hashes and coverage).
The staged weekly refresh regenerates these research outputs. Production models,
archived picks and the hosted dashboard are unchanged by this calculation.

Next evaluation: compare score-form + EPA against the same model plus adjusted
ratings on identical chronological folds. Do not promote on an in-sample ranking.
