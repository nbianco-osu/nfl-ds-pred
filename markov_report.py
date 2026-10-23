"""Build an isolated, offline research viewer; does not change the public dashboard."""
import argparse
import json
from pathlib import Path


def render(report, output):
    payload = json.dumps(report, allow_nan=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regulation Trajectory Research</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 16px;max-width:1100px;color:#172026;background:#fafbfc}select{max-width:100%;padding:10px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ccc}.scroll{overflow:auto}.chart{height:320px}p{line-height:1.5}h1{font-size:28px}canvas{max-width:100%}</style>
<h1>Regulation Trajectory Research</h1>
<p>Research only. These are regulation outcomes, not final-game picks. Overtime is not modeled. Historical replay is not a saved pregame forecast.</p>
<p id="coverage"></p><label for="game">Matchup </label><select id="game"></select>
<p id="summary" role="status"></p><p id="training"></p>
<h2>Projected Score Trajectory</h2><div class="chart"><canvas id="chart" aria-label="Quarter score trajectories"></canvas></div>
<div class="scroll"><table><thead><tr><th>Quarter</th><th>Home mean</th><th>Home 80% band</th><th>Away mean</th><th>Away 80% band</th><th>Home leading</th></tr></thead><tbody id="rows"></tbody></table></div>
<p>Leading probability is not the chance of ultimately winning. Bands summarize simulation variation, not all model uncertainty. Possession scores are placed at drive completion; intermediate paths are coarse.</p>
<script src="../../../public_site/vendor/chart.umd.min.js"></script>
<script>const report=__DATA__;
const el=id=>document.getElementById(id);let chart;
el('coverage').textContent=`Archived source: ${report.source_games} games, through week ${report.source_max_week}. ${report.trials} trials per component. ${report.skipped.length} games skipped for insufficient earlier data.`;
for(const g of report.games){const option=document.createElement('option');option.value=g.game_id;option.textContent=`Week ${g.week}: ${g.away_team} at ${g.home_team}`;el('game').append(option);}
function render(){const g=report.games.find(g=>g.game_id===el('game').value);if(!g)return;
const pct=x=>(100*x).toFixed(1)+'%';
el('summary').textContent=`Regulation: ${g.home_team} win ${pct(g.home_regulation_win)}, tie ${pct(g.regulation_tie)}, ${g.away_team} win ${pct(g.away_regulation_win)}.`;
el('training').textContent=`Transition fit: ${g.training_games} earlier games through week ${g.training_max_week}. Team-conditioned mixture weight ${pct(g.team_weight)}; ${g.weight_training_games} earlier out-of-fold games. ${g.weight_basis}.`;
el('rows').replaceChildren();for(const q of g.quarters){const tr=document.createElement('tr');for(const value of [q.quarter,q.home_mean.toFixed(1),q.home_interval80.map(v=>v.toFixed(1)).join(' to '),q.away_mean.toFixed(1),q.away_interval80.map(v=>v.toFixed(1)).join(' to '),pct(q.home_leading)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}el('rows').append(tr);}
chart?.destroy();if(window.Chart)chart=new Chart(el('chart'),{type:'line',data:{labels:['Q1','Q2','Q3','Q4'],datasets:[{label:g.home_team,data:g.quarters.map(q=>q.home_mean),borderColor:'#175cd3'},{label:g.away_team,data:g.quarters.map(q=>q.away_mean),borderColor:'#087443'}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{y:{beginAtZero:true,title:{display:true,text:'Mean regulation points'}}}}});}
el('game').addEventListener('change',render);render();</script></html>'''
    Path(output).write_text(html.replace('__DATA__', payload), encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    render(json.loads(args.report.read_text()),args.report.with_name('index.html'))
