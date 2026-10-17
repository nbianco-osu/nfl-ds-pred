"""Explicit live dependency expansion for the R60 differential feature contract."""
import pandas as pd


def dependency_columns(columns):
    expanded = list(columns)
    for column in columns:
        if column.startswith('diff_team_'):
            suffix = column.removeprefix('diff_')
            expanded.extend(['home_'+suffix, 'away_'+suffix])
    return list(dict.fromkeys(expanded))


def build_live_features(root, schedule, upcoming, feature_cols, season):
    from shadow_next import features
    if upcoming.game_id.duplicated().any():
        raise ValueError('Duplicate upcoming games')
    # The legacy builder drops unrequested home/away columns before forming diffs.
    report = {'features': {'charted_updated': dependency_columns(feature_cols)}}
    frame = features(root, schedule, upcoming, report, season)
    if not frame.index.equals(pd.Index(upcoming.game_id, name=frame.index.name)):
        raise ValueError('Feature rows do not match requested games')
    return frame.loc[:, feature_cols].copy()
