/* Read-only research summaries and complete, provenance-separated recaps. */
(() => {
 const el=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const n=v=>Number.isFinite(v)?v.toFixed(3):'Unavailable',p=v=>Number.isFinite(v)?(v*100).toFixed(1)+'%':'Unavailable';
 const friendly=v=>({none:'Full feature set',epa:'EPA',sacks:'Sacks',big_plays:'Big plays',all_new:'All opponent features',genuine_pregame:'Verified pregame',legacy_or_unverified:'Legacy / unverified',missing:'Missing'}[v]??String(v??'').replaceAll('_',' '));
 const table=(heads,id)=>`<div class="table-wrap" tabindex="0"><table><thead><tr>${heads.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody id="${id}"></tbody></table></div>`;
 const tr=values=>`<tr>${values.map(v=>`<td>${esc(v)}</td>`).join('')}</tr>`;
 const tab=document.createElement('button');tab.id='matchEvidenceTab';tab.setAttribute('role','tab');tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls','matchEvidencePanel');tab.tabIndex=-1;tab.textContent='Matchup Evidence';document.querySelector('.view-tabs').append(tab);
 const panel=document.createElement('section');panel.id='matchEvidencePanel';panel.hidden=true;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','matchEvidenceTab');
 panel.innerHTML=`<h2>Matchup Evidence</h2><p id="matchEvidenceStatus" role="status">Loading evidence.</p>
 <div class="comparison-controls"><label>Week<select id="evidenceWeek"></select></label><label>Matchup<select id="evidenceGame"></select></label><label>Observation window<select id="evidenceHorizon"><option value="initial">Initial</option><option value="24h">24 hours</option><option value="90m">90 minutes</option></select></label></div>
 <h3>Model Disagreement</h3><p id="disagreementNote"></p>${table(['Generation / target','Model count','Range width: home probability or home / away score','Standard deviation: home probability or home / away score','Pick split / margin and total range width'],'disagreementRows')}
 <h3>Joint Score Forecast</h3><p id="jointNote"></p>${table(['Away / home mean score','Home / away / tie probability','Margin 80% interval','Total 80% interval','Home cover / away cover / push','Over / under / push'],'jointRows')}
 <h3>Pregame Availability</h3><p id="availabilityNote"></p>${table(['Input','Status','Observed at','Provider published at','Used by candidate'],'availabilityRows')}
 <h2>Weekly Recap</h2><p id="recapNote"></p>${table(['Game / actual score','Winner result','Winner provenance','Winner version','Score MAE (points)','Score provenance'],'recapRows')}
 <p><a href="./data/iteration_evidence.json" download>Download matchup evidence</a></p>`;
 document.querySelector('.disclaimer').before(panel);
 const research=document.createElement('section');research.innerHTML=`<h2>Opponent-Adjusted Team Ratings</h2><p id="ratings49Note"></p>${table(['Team','Games','Offense EPA / play','Defense EPA prevented / play','Net EPA / play','Sack differential / game','Big-play differential / game'],'ratings49Rows')}
 <h2>Feature Group Ablations</h2><p id="ablation49Note"></p>${table(['Removed group','Games / decisive','Log loss','Loss change vs full','95% paired interval','Team score MAE'],'ablation49Rows')}<p id="ablation49Limit"></p>
 <h2>Matched Simple Baselines</h2><p id="baseline49Note"></p>${table(['Model','Games','Log loss','Accuracy','Loss change vs simple form','95% paired interval'],'baseline49Rows')}`;el('labPanel').append(research);
 window.dashboardFetch('./data/iteration_evidence.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(d=>{
   el('matchEvidenceStatus').textContent=`${d.season} research evidence. Production predictions are unchanged.`;
   el('disagreementNote').textContent=d.model_disagreement.note;el('jointNote').textContent=d.joint_scores.note;
   el('availabilityNote').textContent=d.input_availability.note;el('recapNote').textContent=d.weekly_recap.note;
   const weeks=d.weekly_recap.weeks;weeks.forEach(w=>el('evidenceWeek').add(new Option(`Week ${w.week}`,w.week)));
   el('evidenceWeek').value=weeks.find(w=>w.completed_games<w.scheduled_games)?.week??weeks[0]?.week;
   const render=()=>{
     const id=el('evidenceGame').value,h=el('evidenceHorizon').value;
     const groups=d.model_disagreement.groups.filter(g=>g.game_id===id&&g.horizon===h);
     el('disagreementRows').innerHTML=groups.map(g=>tr([`${g.generation} / ${friendly(g.task)}`,g.model_count,(g.range??[]).map(v=>g.task==='winner'?p(v):n(v)).join(' / ')||g.status,(g.standard_deviation??[]).map(v=>g.task==='winner'?p(v):n(v)).join(' / '),g.task==='winner'?(g.pick_split?'Models disagree on winner':'Same projected winner'):`${n(g.margin_range)} / ${n(g.total_range)} points`])).join('')||'<tr><td colspan="5">No saved models in this game and observation window.</td></tr>';
     const joint=d.joint_scores.games.find(g=>g.game_id===id);
     el('jointRows').innerHTML=joint?tr([`${n(joint.expected_away)} / ${n(joint.expected_home)}`,`${p(joint.home_win)} / ${p(joint.away_win)} / ${p(joint.tie)}`,joint.margin_interval80.map(n).join(' to '),joint.total_interval80.map(n).join(' to '),joint.spread?`${p(joint.spread.above)} / ${p(joint.spread.below)} / ${p(joint.spread.push)}`:'No archived line',joint.total?`${p(joint.total.above)} / ${p(joint.total.below)} / ${p(joint.total.push)}`:'No archived line']):'<tr><td colspan="6">No eligible saved joint forecast.</td></tr>';
     const fresh=d.input_availability.games.find(g=>g.game_id===id);
     el('availabilityRows').innerHTML=(fresh?.inputs??[]).map(r=>tr([r.input,r.status,r.observed_at??'Unknown',r.provider_published_at??'Unknown',r.used_by_candidate?'Yes':'Context only'])).join('')||'<tr><td colspan="5">No saved freshness evidence. Missing is not confirmation of availability.</td></tr>';
   };
   const week=()=>{
     const w=weeks.find(w=>String(w.week)===el('evidenceWeek').value);el('evidenceGame').replaceChildren();
     (w?.games??[]).forEach(g=>el('evidenceGame').add(new Option(`${g.away_team} at ${g.home_team}`,g.game_id)));
     el('recapRows').innerHTML=(w?.games??[]).map(g=>{
       const a=g.tasks.winner,s=g.tasks.score,grade=a.grade,cls=grade==='Correct'?'prediction-correct':grade==='Wrong'?'prediction-incorrect':'';
       return `<tr><td>${esc(g.away_team)} at ${esc(g.home_team)}<br>${g.status==='Final'?`${g.away_score} - ${g.home_score}`:'Pending'}</td><td class="${cls}">${esc(grade)}</td><td>${esc(friendly(a.cohort))}</td><td>${esc(a.generation)}</td><td>${n(s.team_mae)}</td><td>${esc(friendly(s.cohort))}</td></tr>`;
     }).join('');render();
   };
   el('evidenceWeek').addEventListener('change',week);['evidenceGame','evidenceHorizon'].forEach(id=>el(id).addEventListener('change',render));week();
   el('ratings49Note').textContent=d.opponent_adjusted.note;
   el('ratings49Rows').innerHTML=(d.opponent_adjusted.ratings?.latest_ratings??[]).map(r=>tr([r.team,r.games,n(r.offense_epa),n(r.defense_epa_prevented),n(r.net_epa),n(r.sack_differential_per_game),n(r.big_play_differential_per_game)])).join('');
   const a=d.grouped_ablation;el('ablation49Note').textContent=a.note??a.reason;
   el('ablation49Rows').innerHTML=(a.results??[]).map(r=>tr([friendly(r.removed_group),`${r.games} / ${r.decisive_games}`,n(r.log_loss),n(r.delta_log_loss_vs_full),r.ci95?.map(n).join(' to ')??'Unavailable',n(r.team_score_mae)])).join('');
   el('ablation49Limit').textContent=`Unavailable groups: ${(a.unavailable_groups??[]).map(friendly).join(', ')||'None listed'}. ${a.source_changed_since_research?'Source data changed since this frozen experiment; no automatic refit.':''}`;
   const b=d.simple_baselines??d.opponent_adjusted.simple_baselines;
   el('baseline49Note').textContent=b?.note??'Matched Elo comparison and special-teams ratings are not yet available in this report; no superiority claim is made.';
   el('baseline49Rows').innerHTML=(b?.results??[]).map(r=>tr([friendly(r.model??r.name),r.games,n(r.log_loss),p(r.accuracy),n(r.delta_log_loss_vs_simple),r.ci95?.map(n).join(' to ')??'Unavailable'])).join('');
   const st=d.opponent_adjusted.special_teams;
   if(st){const section=document.createElement('section');section.innerHTML=`<h2>Special Teams: Descriptive EPA</h2><p>${esc(st.note)}</p>${table(['Team','Games','Valid plays','Net EPA / play','Coverage'],'special49Rows')}`;research.append(section);
     el('special49Rows').innerHTML=(st.teams??[]).map(r=>tr([r.team,r.games,r.plays,n(r.epa_per_play),r.status])).join('')||'<tr><td colspan="5">Special-teams source unavailable.</td></tr>';}
 }).catch(()=>el('matchEvidenceStatus').textContent='Matchup evidence unavailable. Production predictions are unaffected.');
})();
