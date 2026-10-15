"""R47 lightweight due-window check; refresh only uncaptured live windows."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import pandas as pd
from pregame_inputs import kickoff,timestamp
from research_operations import read
from score_markets import pregame


def due(root,season,now=None):
    now=timestamp(now or datetime.now(timezone.utc))
    captured=read(root,'audit/checkpoints.json',{})
    result=[]
    for g in pd.read_csv(root/f'data/schedule_{season}.csv').to_dict('records'):
        if not pregame(g,now):continue
        remaining=(kickoff(g)-now).total_seconds()/60
        for name,minutes,tolerance in [('24h',1440,30),('90m',90,15)]:
            if abs(remaining-minutes)<=tolerance and g['game_id']+'|'+name not in captured:
                result.append({'game_id':g['game_id'],'horizon':name})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--season',type=int)
    args=p.parse_args();now=datetime.now(timezone.utc);season=args.season or now.year-(now.month<=2)
    root=Path(__file__).resolve().parent;windows=due(root,season)
    print(json.dumps({'due':windows,'needs_refresh':bool(windows)}),flush=True)
    if windows and args.run:
        from staged_refresh import refresh
        refresh(root,season)
