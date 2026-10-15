/* R41-R48: prospective evidence is distinct from retrospective experiments. */
(() => {
 const el=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const n=v=>Number.isFinite(v)?v.toFixed(3):'Unavailable';
 const name=v=>v==='ensemble_score'?'Temporal score ensemble':v==='ensemble_winner'?'Temporal winner ensemble':window.candidateDisplayName(v)+(v.endsWith('_charted_updated')?' - charted inputs':v.endsWith('_core_updated')?' - core inputs':''), pct=v=>Number.isFinite(v)?(100*v).toFixed(1)+'%':'Pending';
 const table=(head,id)=>`<div class="table-wrap" tabindex="0"><table><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody id="${id}"></tbody></table></div>`;
 const read=async file=>{const r=await fetch(`./data/${file}.json`);if(!r.ok)throw Error(file);return r.json();};
 const tab=document.createElement('button');tab.id='nextTab';tab.setAttribute('role','tab');tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls','nextPanel');tab.tabIndex=-1;tab.textContent='Shadow Forecasts';document.querySelector('.view-tabs').append(tab);
 const panel=document.createElement('section');panel.id='nextPanel';panel.hidden=true;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','nextTab');
 panel.innerHTML=`<h2>Upcoming Shadow Forecasts</h2><p id="nextStatus" role="status">Loading forecasts.</p><p id="nextNote"></p>
 <div class="comparison-controls"><label>Model<select id="nextModel"></select></label><label>Observation window<select id="nextHorizon"><option value="initial">Initial</option><option value="24h">24 hours</option><option value="90m">90 minutes</option></select></label></div>
 ${table(['Game','Projected winner / score (away - home)','Home win probability','Margin / total','80% historical residual ranges','Saved at'],'nextRows')}
 <h2>Prospective Scorecard</h2><p id="nextGradeStatus"></p>${table(['Model / window','Games','Winner accuracy / team MAE','Log loss / margin MAE','Brier / total MAE','Spread correct / wrong / push','Total correct / wrong / push'],'nextGrades')}
 <h2>Prospective Production Comparison</h2><p id="pairedNote"></p>${table(['Candidate','Production version','Window','Matched games','Log-loss difference','95% week interval'],'pairedRows')}
 <h2>Game Input Freshness</h2><label>Game<select id="freshGame"></select></label><p id="freshNote"></p>
 ${table(['Input','Status','Observed at','Age (hours)','Missing fields','Model input'],'freshRows')}
 <h2>Pregame Refresh Schedule</h2><p id="nextSchedule">Loading schedule status.</p>
 <h2>Model Release Review</h2><p id="releaseNote"></p>${table(['Candidate','Prospective games','Feature contract','Approval','Readiness'],'releaseRows')}
 <p id="rollbackNote"></p><p><a href="./data/release_review.json" download>Download release checklist</a> | <a href="./data/shadow_next.json" download>Download shadow forecasts and scorecard</a></p>`;
 document.querySelector('.disclaimer').before(panel);
 read('shadow_next').then(d=>{
   el('nextStatus').textContent=`${new Set(d.forecasts.map(r=>r.game_id)).size} upcoming games | ${new Set(d.forecasts.map(r=>r.model)).size} shadow configurations and ensembles | ${d.archived_observations} preserved observations`;
   el('nextNote').textContent=d.note;
   const models=[...new Set(d.forecasts.map(r=>r.model))].sort();models.forEach(m=>el('nextModel').add(new Option(name(m),m)));
   el('nextModel').value=models.includes('ensemble_score')?'ensemble_score':models[0]??'';
   if(!d.forecasts.some(r=>r.horizon==='initial')&&d.forecasts.length)el('nextHorizon').value=d.forecasts[0].horizon;
   const render=()=>{
     const model=el('nextModel').value,h=el('nextHorizon').value,rows=d.forecasts.filter(r=>r.model===model&&r.horizon===h);
     el('nextRows').innerHTML=rows.map(r=>{const score=r.task==='score',ranges=r.intervals?Object.entries(r.intervals).map(([k,v])=>`${name(k)}: ${v.map(x=>x.toFixed(1)).join(' to ')}`).join('; '):'Unavailable';
       return `<tr><td>${esc(r.away_team)} at ${esc(r.home_team)}<br>Week ${r.week}</td><td>${score?`${r.predicted_away_score.toFixed(1)} - ${r.predicted_home_score.toFixed(1)}`:esc(r.predicted_winner)}</td><td>${score?'Separate winner model':pct(r.home_probability)}</td><td>${score?`${(r.predicted_home_score-r.predicted_away_score).toFixed(1)} / ${(r.predicted_home_score+r.predicted_away_score).toFixed(1)}`:'Not a score model'}</td><td>${esc(ranges)}</td><td>${esc(new Date(r.forecast_at).toLocaleString())}</td></tr>`;}).join('')||'<tr><td colspan="6">No actual pregame observation in this window.</td></tr>';
     const grades=d.scorecard.filter(r=>r.model===model&&r.horizon===h);
     el('nextGradeStatus').textContent=grades.length?'Genuine saved pregame predictions only. Ties, missing lines and pushes are handled separately.':'No completed prospective games yet for this model and observation window.';
     const pairs=(d.prospective_comparisons??[]).filter(r=>r.candidate===model&&r.horizon===h);
     el('pairedNote').textContent=pairs.length?'Identical observed games and windows, separated by production version. Negative loss difference favors the candidate; small samples are not proof of improvement.':'Waiting for matched completed games against timestamped production observations.';
     el('pairedRows').innerHTML=pairs.map(r=>`<tr><td>${esc(name(r.candidate))}</td><td>${esc(r.production_version)}</td><td>${esc(r.horizon)}</td><td>${r.games}</td><td>${n(r.delta_log_loss)}</td><td>${r.ci95?r.ci95.map(n).join(' to '):'Insufficient weeks'}</td></tr>`).join('');
     el('nextGrades').innerHTML=grades.map(r=>`<tr><td>${esc(name(r.model))} / ${esc(r.horizon)}</td><td>${r.graded??r.games}</td><td>${r.task==='winner'?pct(r.accuracy):n(r.team_mae)}</td><td>${n(r.log_loss??r.margin_mae)}</td><td>${n(r.brier??r.total_mae)}</td><td>${r.task==='score'?`${r.spread_correct} / ${r.spread_wrong} / ${r.spread_pushes}`:'Not applicable'}</td><td>${r.task==='score'?`${r.total_correct} / ${r.total_wrong} / ${r.total_pushes}`:'Not applicable'}</td></tr>`).join('');
   };['nextModel','nextHorizon'].forEach(id=>el(id).addEventListener('change',render));render();
 }).catch(()=>el('nextStatus').textContent='Shadow forecast report unavailable. Production forecasts are unaffected.');
 read('feature_freshness').then(d=>{el('freshNote').textContent=d.note;d.games.forEach(g=>el('freshGame').add(new Option(g.game_id,g.game_id)));
   const render=()=>{const game=d.games.find(g=>g.game_id===el('freshGame').value);el('freshRows').innerHTML=(game?.inputs??[]).map(r=>`<tr><td>${esc(r.input)}</td><td>${esc(r.status)}</td><td>${esc(r.observed_at??'Unknown')}</td><td>${n(r.age_hours)}</td><td>${r.fields==null?'Not applicable':`${r.missing_fields} / ${r.fields}`}</td><td>${r.used_by_candidate?'Yes':'Context only'}</td></tr>`).join('');};el('freshGame').addEventListener('change',render);render();
 }).catch(()=>el('freshNote').textContent='Freshness report unavailable.');
 read('checkpoint_schedule').then(d=>el('nextSchedule').textContent=d.note).catch(()=>el('nextSchedule').textContent='Schedule status unavailable.');
 read('release_review').then(d=>{el('releaseNote').textContent=d.note;el('rollbackNote').textContent=`Local rollback manifests: ${d.rollback_bundles.join(', ')||'None'}.`;
   el('releaseRows').innerHTML=d.candidates.map(r=>`<tr><td>${esc(name(r.name))}</td><td>${r.prospective_games} / ${r.review_sample_threshold} review threshold</td><td>${r.feature_contract_compatible?'Compatible':'Integration required'}</td><td>${r.approved?'Approved':'Not approved'}</td><td>${esc(r.status)}</td></tr>`).join('');
 }).catch(()=>el('releaseNote').textContent='Release review unavailable. No activation permitted.');
 const evidence=document.createElement('section');evidence.innerHTML=`<h2>Matched Production-Recipe Comparison</h2><p id="nextComparisonNote"></p>${table(['Feature view','Candidate','Games','Candidate log loss','Reference log loss','Difference and 95% interval','Conclusion'],'nextComparison')}
 <h2>Temporal Ensemble Evaluation</h2><p id="nextEnsembleNote"></p>${table(['Target','Member','Frozen weight'],'nextWeights')}<p id="nextEnsembleResult"></p>
 <h2>Prediction Interval Evaluation</h2><p id="nextIntervalNote"></p>${table(['Target','Nominal coverage','2026 observed coverage','Half-width (points)'],'nextIntervals')}`;
 el('labPanel').append(evidence);
 read('next_evidence').then(d=>{
   el('nextComparisonNote').textContent=d.comparison_note;
   el('nextComparison').innerHTML=d.comparisons.map(r=>`<tr><td>${esc(name(r.view))}</td><td>${esc(name(r.candidate))}</td><td>${r.games}</td><td>${n(r.candidate_metrics.log_loss)}</td><td>${n(r.reference_metrics.log_loss)}</td><td>${n(r.delta_log_loss)} [${r.ci95.map(n).join(', ')}]</td><td>${esc(r.status)}</td></tr>`).join('');
   const e=d.ensemble;el('nextEnsembleNote').textContent=e.note;
   el('nextWeights').innerHTML=['winner','score'].flatMap(k=>e[k+'_members'].map((v,i)=>`<tr><td>${k}</td><td>${esc(v)}</td><td>${pct(e[k+'_weights'][i])}</td></tr>`)).join('');
   el('nextEnsembleResult').textContent=`2026 winner log loss: ensemble ${n(e.winner_metrics.log_loss)}, validation-selected individual ${n(e.winner_individual_selected_on_early2025.log_loss)}. Score total MAE: ensemble ${n(e.score_metrics.total_mae)}, selected individual ${n(e.score_individual_selected_on_early2025.total_mae)} points.`;
   const q=d.intervals;el('nextIntervalNote').textContent=q.note;
   el('nextIntervals').innerHTML=Object.entries(q.radii).map(([k,v])=>`<tr><td>${esc(name(k))}</td><td>${pct(q.nominal_coverage)}</td><td>${pct(q.holdout_coverage[k])}</td><td>${n(v)}</td></tr>`).join('');
 }).catch(()=>el('nextComparisonNote').textContent='Matched evidence unavailable.');
})();
