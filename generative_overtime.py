"""2026 regular-season overtime stopping logic over coarse possession events.

Rule 16: https://operations.nfl.com/rules-officiating/2026-nfl-rulebook
Not a play-level engine: no penalties, onside recoveries, timeouts or live-clock
extensions. Scoring and durations use regulation drives as an explicit fallback.
"""
import numpy as np


def resolve(events, first=0):
    """Events contain duration, offense points, defense points; each ends a possession."""
    if first not in (0, 1):
        raise ValueError('Invalid receiver')
    scores = np.zeros(2, dtype=int)
    side, clock, possessions, sudden = first, 600, 0, False
    for duration, own, defense in events:
        if duration < 1 or own not in (0, 2, 3, 6, 7, 8) or defense not in (0, 2, 6, 7, 8):
            raise ValueError('Invalid overtime event')
        if duration > clock:
            break
        clock -= duration
        # A winning TD has no unnecessary conversion. A reply TD may need its try.
        if defense >= 6:
            defense = 6
        if own >= 6 and (sudden or (possessions >= 1 and scores[side]+6 > scores[1-side]) or clock == 0):
            own = 6
        scores[side] += own
        scores[1-side] += defense
        possessions += 1
        if defense or (sudden and own):
            break
        if possessions >= 2 and scores[0] != scores[1]:
            break
        if clock == 0:
            break
        if possessions == 1 and own == 0 or possessions >= 2:
            sudden = True
        side = 1-side
    return scores


def finalize(paths, model, home, away, seed=42):
    """Retain regulation paths and add a distinct final-score endpoint."""
    rng = np.random.default_rng(seed)
    final = paths[:, -1].copy()
    d = model['drives']
    pools = []
    for offense, defense in [(home, away), (away, home)]:
        weights = np.ones(len(d))/len(d)
        local = (d.offense.eq(offense) | d.defense.eq(defense)).to_numpy(dtype=float)
        if local.sum():
            weights = .65*weights+.35*local/local.sum()
        pools.append(weights)
    events = d[['duration', 'own', 'opp']].to_numpy(dtype=int)
    for i in np.flatnonzero(final[:, 0] == final[:, 1]):
        first = int(rng.integers(2))
        def stream():
            for j in range(600):
                yield events[rng.choice(len(events), p=pools[(first+j)%2])]
        final[i] += resolve(stream(), first)
    return final
