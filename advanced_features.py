from __future__ import annotations

import nflreadpy as nfl
import pandas as pd


ROLLING_WINDOWS = (3, 5)


def _safe_load(name: str, seasons: list[int], min_season: int | None = None, max_season: int | None = None) -> pd.DataFrame:
    load_seasons = seasons
    if min_season is not None:
        load_seasons = [season for season in load_seasons if season >= min_season]
    if max_season is not None:
        load_seasons = [season for season in load_seasons if season <= max_season]
    if not load_seasons:
        print(f"Skipping {name}: no requested seasons are in its supported range")
        return pd.DataFrame()
    try:
        frame = getattr(nfl, name)(load_seasons).to_pandas()
        print(f"Loaded {name}: {len(frame):,} rows")
        return frame
    except Exception as exc:
        print(f"Skipping {name}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _game_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    base_cols = [col for col in ["game_id", "season", "week", "gameday"] if col in games.columns]
    home = games[base_cols + ["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    home["is_home"] = 1
    away = games[base_cols + ["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    away["is_home"] = 0
    rows = pd.concat([home, away], ignore_index=True)
    rows["gameday"] = pd.to_datetime(rows["gameday"], errors="coerce")
    return rows.sort_values(["team", "season", "week", "game_id"]).reset_index(drop=True)


def _pregame_rollups(
    per_game: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
    season_reset: bool = True,
) -> pd.DataFrame:
    if per_game.empty:
        return pd.DataFrame(columns=["game_id", "team", "season", "week"])

    per_game = per_game.sort_values(["team", "season", "week", "game_id"]).copy()
    group_cols = ["team", "season"] if season_reset else ["team"]
    grouped = per_game.groupby(group_cols, group_keys=False)
    out = per_game[["game_id", "team", "season", "week"]].copy()
    prior_games = grouped.cumcount()
    denom = prior_games.mask(prior_games.eq(0))

    for col in value_cols:
        shifted = grouped[col].shift(1)
        out[f"{prefix}_{col}_avg"] = grouped[col].cumsum().groupby([per_game[c] for c in group_cols]).shift(fill_value=0) / denom
        for window in ROLLING_WINDOWS:
            out[f"{prefix}_{col}_last_{window}"] = (
                shifted.groupby([per_game[c] for c in group_cols])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=list(range(len(group_cols))), drop=True)
            )

    return out.fillna(0)


def _prior_season_features(per_game: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    if per_game.empty:
        return pd.DataFrame(columns=["season", "team"])

    season_summary = per_game.groupby(["team", "season"], as_index=False)[value_cols].mean()
    season_summary["season"] = season_summary["season"] + 1
    return season_summary.rename(columns={col: f"{prefix}_prev_season_{col}" for col in value_cols})


def make_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    rows = _game_team_rows(games)
    grouped = rows.groupby("team", group_keys=False)
    rows["previous_game_date"] = grouped["gameday"].shift(1)
    rows["previous_game_season"] = grouped["season"].shift(1)
    rows["team_rest_days"] = (rows["gameday"] - rows["previous_game_date"]).dt.days
    rows.loc[rows["previous_game_season"].ne(rows["season"]), "team_rest_days"] = 7
    rows["team_rest_days"] = rows["team_rest_days"].clip(lower=4, upper=14).fillna(7)
    return rows[["game_id", "team", "season", "week", "team_rest_days"]]


def make_pbp_features(games: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp = _safe_load("load_pbp", seasons)
    if pbp.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep = pbp[
        pbp["game_id"].isin(games["game_id"])
        & pbp["posteam"].notna()
        & pbp["epa"].notna()
        & ~pbp["play_type"].isin(["no_play", "qb_kneel", "qb_spike"])
    ].copy()
    keep["success"] = keep["epa"].gt(0).astype(int)
    keep["explosive_play"] = (
        (keep["pass_attempt"].fillna(0).eq(1) & keep["yards_gained"].fillna(0).ge(20))
        | (keep["rush_attempt"].fillna(0).eq(1) & keep["yards_gained"].fillna(0).ge(10))
    ).astype(int)
    keep["turnover"] = (
        keep.get("interception", 0).fillna(0).eq(1) | keep.get("fumble_lost", 0).fillna(0).eq(1)
    ).astype(int)

    off = (
        keep.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_explosive_rate=("explosive_play", "mean"),
            off_turnover_rate=("turnover", "mean"),
            off_pass_rate=("pass_attempt", "mean"),
            off_pass_epa=("epa", lambda s: s[keep.loc[s.index, "pass_attempt"].fillna(0).eq(1)].mean()),
            off_rush_epa=("epa", lambda s: s[keep.loc[s.index, "rush_attempt"].fillna(0).eq(1)].mean()),
        )
        .rename(columns={"posteam": "team"})
    )
    defense_source = keep.copy()
    defense_source["def_epa_allowed_per_play"] = defense_source["epa"]
    defense_source["def_success_allowed_rate"] = defense_source["success"]
    defense_source["def_explosive_allowed_rate"] = defense_source["explosive_play"]
    defense_source["def_takeaway_rate"] = defense_source["turnover"]
    defense = (
        defense_source.groupby(["game_id", "defteam"], as_index=False)
        .agg(
            def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
            def_success_allowed_rate=("def_success_allowed_rate", "mean"),
            def_explosive_allowed_rate=("def_explosive_allowed_rate", "mean"),
            def_takeaway_rate=("def_takeaway_rate", "mean"),
        )
        .rename(columns={"defteam": "team"})
    )
    per_game = off.merge(defense, on=["game_id", "team"], how="outer")
    per_game = per_game.merge(_game_team_rows(games)[["game_id", "team", "season", "week"]], on=["game_id", "team"], how="left")

    value_cols = [col for col in per_game.columns if col not in {"game_id", "team", "season", "week"}]
    return _pregame_rollups(per_game, value_cols, "team_pbp"), _prior_season_features(per_game, value_cols, "team_pbp")


def make_qb_features(games: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = _safe_load("load_player_stats", seasons)
    if stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    qb = stats[
        stats["season"].isin(seasons)
        & stats["season_type"].eq("REG")
        & stats["position"].eq("QB")
        & stats["attempts"].fillna(0).gt(0)
    ].copy()
    if qb.empty:
        return pd.DataFrame(), pd.DataFrame()

    qb = qb.sort_values(["season", "week", "team", "attempts"], ascending=[True, True, True, False])
    starters = qb.groupby(["season", "week", "team"], as_index=False).first()
    starters = starters.rename(columns={"team": "team", "player_id": "qb_player_id"})
    value_cols = [
        col
        for col in [
            "attempts",
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "sacks_suffered",
            "passing_epa",
            "passing_cpoe",
        ]
        if col in starters.columns
    ]
    per_game = _game_team_rows(games)[["game_id", "team", "season", "week"]].merge(
        starters[["season", "week", "team", "qb_player_id"] + value_cols],
        on=["season", "week", "team"],
        how="left",
    )
    per_game = per_game.rename(columns={col: f"qb_{col}" for col in value_cols})
    qb_value_cols = [f"qb_{col}" for col in value_cols]

    per_game = per_game.sort_values(["team", "season", "week", "game_id"])
    per_game["qb_prev_starter_same_as_prev_game"] = (
        per_game.groupby(["team", "season"])["qb_player_id"].shift(1)
        == per_game.groupby(["team", "season"])["qb_player_id"].shift(2)
    ).fillna(False).astype(int)
    qb_value_cols.append("qb_prev_starter_same_as_prev_game")
    return _pregame_rollups(per_game.fillna(0), qb_value_cols, "team_qb"), _prior_season_features(per_game.fillna(0), qb_value_cols, "team_qb")


def make_injury_features(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    injuries = _safe_load("load_injuries", seasons, min_season=2009, max_season=2025)
    if injuries.empty:
        return pd.DataFrame()

    injuries = injuries[injuries["game_type"].eq("REG")].copy()
    status = injuries["report_status"].fillna("").str.lower()
    injuries["injury_out"] = status.str.contains("out|doubtful|injured reserve|reserve").astype(int)
    injuries["injury_questionable"] = status.str.contains("questionable").astype(int)
    injuries["injury_qb_out"] = (injuries["position"].eq("QB") & injuries["injury_out"].eq(1)).astype(int)
    injuries["injury_skill_out"] = (
        injuries["position"].isin(["QB", "RB", "FB", "WR", "TE"]) & injuries["injury_out"].eq(1)
    ).astype(int)
    injuries["injury_line_out"] = (
        injuries["position"].isin(["C", "G", "T", "OL", "DE", "DT", "NT", "DL", "LB", "OLB", "ILB"])
        & injuries["injury_out"].eq(1)
    ).astype(int)
    return (
        injuries.groupby(["season", "week", "team"], as_index=False)
        .agg(
            team_injury_report_count=("gsis_id", "count"),
            team_injury_out_count=("injury_out", "sum"),
            team_injury_questionable_count=("injury_questionable", "sum"),
            team_injury_qb_out=("injury_qb_out", "max"),
            team_injury_skill_out_count=("injury_skill_out", "sum"),
            team_injury_line_out_count=("injury_line_out", "sum"),
        )
    )


def make_roster_features(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    rosters = _safe_load("load_rosters_weekly", seasons, min_season=2002, max_season=2025)
    if rosters.empty:
        return pd.DataFrame()

    rosters = rosters[rosters["game_type"].eq("REG")].copy()
    rosters["years_exp"] = pd.to_numeric(rosters["years_exp"], errors="coerce")
    status = rosters["status"].fillna("").str.lower()
    rosters["roster_active"] = status.str.contains("active").astype(int)
    rosters["roster_rookie"] = rosters["years_exp"].fillna(0).eq(0).astype(int)
    rosters["roster_qb_count"] = rosters["position"].eq("QB").astype(int)
    rosters["roster_ol_count"] = rosters["position"].isin(["C", "G", "T", "OL"]).astype(int)
    rosters["roster_skill_count"] = rosters["position"].isin(["QB", "RB", "FB", "WR", "TE"]).astype(int)
    return (
        rosters.groupby(["season", "week", "team"], as_index=False)
        .agg(
            team_roster_size=("gsis_id", "count"),
            team_roster_active_count=("roster_active", "sum"),
            team_roster_avg_years_exp=("years_exp", "mean"),
            team_roster_rookie_count=("roster_rookie", "sum"),
            team_roster_qb_count=("roster_qb_count", "sum"),
            team_roster_ol_count=("roster_ol_count", "sum"),
            team_roster_skill_count=("roster_skill_count", "sum"),
        )
    )


def merge_team_feature_block(matchups: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    if team_features.empty:
        return matchups

    home = team_features.add_prefix("home_").rename(
        columns={"home_team": "home_team", "home_season": "season", "home_week": "week", "home_game_id": "game_id"}
    )
    away = team_features.add_prefix("away_").rename(
        columns={"away_team": "away_team", "away_season": "season", "away_week": "week", "away_game_id": "game_id"}
    )
    home_keys = [key for key in ["game_id", "season", "week", "home_team"] if key in home.columns and key in matchups.columns]
    away_keys = [key for key in ["game_id", "season", "week", "away_team"] if key in away.columns and key in matchups.columns]
    out = matchups.merge(home, on=home_keys, how="left")
    out = out.merge(away, on=away_keys, how="left")
    return out


def add_diff_columns(matchups: pd.DataFrame) -> pd.DataFrame:
    home_cols = [col for col in matchups.columns if col.startswith("home_")]
    blocked_suffixes = {"score", "win"}
    diff_data = {}
    for home_col in home_cols:
        suffix = home_col.removeprefix("home_")
        if suffix in blocked_suffixes:
            continue
        away_col = "away_" + suffix
        if away_col in matchups.columns and pd.api.types.is_numeric_dtype(matchups[home_col]):
            diff_col = "diff_" + suffix
            if diff_col not in matchups.columns:
                diff_data[diff_col] = matchups[home_col] - matchups[away_col]
    if not diff_data:
        return matchups
    return pd.concat([matchups, pd.DataFrame(diff_data, index=matchups.index)], axis=1).copy()


def add_advanced_features(matchups: pd.DataFrame, games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    print("Building rest features")
    matchups = merge_team_feature_block(matchups, make_rest_features(games))

    print("Building play-by-play EPA features")
    pbp_rollups, pbp_prev = make_pbp_features(games, seasons)
    matchups = merge_team_feature_block(matchups, pbp_rollups)
    matchups = merge_team_feature_block(matchups, pbp_prev)

    print("Building QB features")
    qb_rollups, qb_prev = make_qb_features(games, seasons)
    matchups = merge_team_feature_block(matchups, qb_rollups)
    matchups = merge_team_feature_block(matchups, qb_prev)

    print("Building injury features")
    matchups = merge_team_feature_block(matchups, make_injury_features(games, seasons))

    print("Building roster features")
    matchups = merge_team_feature_block(matchups, make_roster_features(games, seasons))

    return add_diff_columns(matchups)
