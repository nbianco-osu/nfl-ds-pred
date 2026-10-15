/* R23-R27: keep weekly matchups focused; research and operations stay secondary. */
(() => {
  const el=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>Number.isFinite(v)?v.toFixed(3):'Unavailable';
  const names={poisson:'Independent Poisson',negative_binomial:'Overdispersed scores',correlated_negative_binomial:'Correlated overdispersed scores'};
  const library=el('modelComparison'),signals=el('shapBars').closest('.section-grid');
  el('researchPanel').append(library,signals);
  const season=el('winnerBars').closest('aside');
  season.classList.remove('panel');season.classList.add('season-summary');
  el('seasonPanel').append(season);
  el('matchupList').closest('.section-grid').classList.add('weekly-only');
  const panel=el('experimentsPanel');
  const section=document.createElement('section');section.className='experiment-section';
  section.innerHTML=`<h2>Score Uncertainty: Matched Research</h2><p id="uncertaintyStatus"></p>
    <label>Target<select id="uncertaintyTarget"><option value="total">Combined score</option><option value="margin">Home margin</option><option value="home">Home score</option><option value="away">Away score</option></select></label>
    <div class="table-wrap" tabindex="0" aria-label="Overdispersed score experiments"><table><thead><tr><th>Distribution</th><th>Games</th><th>CRPS</th><th>80% coverage</th><th>Width</th><th>Interval score</th></tr></thead><tbody id="uncertaintyRows"></tbody></table></div>
    <details><summary>Methods and limitations</summary><p id="uncertaintyNote"></p></details>
    <h2>2026 Season Replay</h2><p id="replayStatus"></p><label>Week<select id="replayWeek"></select></label>
    <div class="table-wrap" tabindex="0" aria-label="Retrospective and future score projections"><table><thead><tr><th>Game</th><th>Basis</th><th>Model</th><th>Expected away / home</th><th>Home win chance</th><th>Total 80% range</th></tr></thead><tbody id="replayRows"></tbody></table></div>
    <p><a href="./data/season_replay.json" download>Download all season projections</a></p>
    <h2>Opponent-Adjusted EPA</h2><p id="ratingsNote"></p><div class="table-wrap" tabindex="0" aria-label="Current team efficiency ratings"><table><thead><tr><th>Team</th><th>Games</th><th>Offense EPA</th><th>Defense EPA prevented</th><th>Net EPA</th><th>Sack differential / game</th><th>Big-play differential / game</th></tr></thead><tbody id="ratingsRows"></tbody></table></div>`;
  panel.prepend(section);
  async function read(name){const r=await fetch(`./data/${name}.json`);if(!r.ok)throw Error(name);return r.json();}
  read('uncertainty_research').then(data=>{
    el('uncertaintyStatus').textContent='Chronological research only; no model promotion or guaranteed interval coverage.';
    el('uncertaintyNote').textContent=data.note;
    const render=()=>{const key=el('uncertaintyTarget').value;
      el('uncertaintyRows').innerHTML=data.models.map(m=>`<tr><td>${esc(names[m.id])}</td><td>${m.games}</td><td>${num(m.metrics[key+'_crps'])}</td><td>${(100*m.metrics[key+'_coverage80']).toFixed(1)}%</td><td>${num(m.metrics[key+'_width80'])}</td><td>${num(m.metrics[key+'_interval_score80'])}</td></tr>`).join('');};
    el('uncertaintyTarget').addEventListener('change',render);render();
  }).catch(()=>el('uncertaintyStatus').textContent='Score uncertainty research unavailable.');
  read('season_replay').then(data=>{
    el('replayStatus').textContent=data.note;
    [...new Set(data.forecasts.map(r=>r.week))].forEach(w=>el('replayWeek').add(new Option(`Week ${w}`,w)));
    const render=()=>el('replayRows').innerHTML=data.forecasts.filter(r=>r.week===Number(el('replayWeek').value)).map(r=>`<tr><td>${esc(r.game_id)}</td><td>${esc(r.basis)}</td><td>${esc(names[r.kind])}</td><td>${r.expected_away.toFixed(1)} / ${r.expected_home.toFixed(1)}</td><td>${(r.home_win_probability*100).toFixed(1)}%</td><td>${r.total_interval80.join(' - ')}</td></tr>`).join('');
    el('replayWeek').addEventListener('change',render);render();
  }).catch(()=>el('replayStatus').textContent='Season replay unavailable.');
  read('opponent_ratings').then(data=>{
    el('ratingsNote').textContent=`${data.source_games} completed source games. Current rankings, not historical matchup inputs. Positive is better. Our own research metric, not DVOA.`;
    el('ratingsRows').innerHTML=data.latest_ratings.map(r=>`<tr><td>${esc(r.team)}</td><td>${r.games}</td>${['offense_epa','defense_epa_prevented','net_epa','sack_differential_per_game','big_play_differential_per_game'].map(k=>`<td>${num(r[k])}</td>`).join('')}</tr>`).join('');
  }).catch(()=>el('ratingsNote').textContent='Opponent ratings unavailable.');
  read('live_challengers').then(data=>{
    el('liveStatus').textContent=`${data.forecasts.length} saved forecasts across three challengers. ${data.note}`;
    el('liveRows').innerHTML=Object.entries(data.cohorts).flatMap(([v,rows])=>rows.map(r=>`<tr><td>${esc(v)}</td><td>${esc(names[r.kind])}</td><td>${r.games}</td><td>${num(r.challenger_log_loss)}</td><td>${num(r.production_log_loss)}</td><td>${esc(r.status)}</td></tr>`)).join('');
    const lookup=new Map(data.forecasts.filter(r=>r.kind==='correlated_negative_binomial').map(r=>[r.game_id,r]));
    window.matchupUncertainty=row=>{const r=lookup.get(row.game_id);return `<p class="matchup-uncertainty">${r?`Shadow total: ${r.total_interval80.join(' - ')} points (nominal 80% range). Separate research model.`:'Score uncertainty: no archived research interval.'}</p>`;};
    window.dispatchEvent(new Event('researchready'));
  }).catch(()=>el('liveStatus').textContent='Live challenger archive unavailable.');
  read('reliability').then(data=>{
    el('reliabilityStatus').textContent=`${data.status==='healthy'?'Checks passed':'Attention needed'}; checked ${data.updated_at}. Last validated refresh: ${data.last_successful_refresh??'unknown'}.`;
    el('reliabilityRows').innerHTML=data.issues.length?data.issues.map(r=>`<tr><td>${esc(r.severity)}</td><td>${esc(r.message)}</td><td>${esc(r.action)}</td></tr>`).join(''):'<tr><td colspan="3">No actionable issues detected.</td></tr>';
  }).catch(()=>el('reliabilityStatus').textContent='Reliability report unavailable; health is unknown.');
})();
