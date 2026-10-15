/* Completed research generations remain separate from live model performance. */
(() => {
  const section = document.createElement('section');
  section.innerHTML = `<h2>Retrained Model Candidates</h2>
    <p id="retrainingStatus" role="status">Loading retraining results.</p>
    <p>Shadow candidates only. Live forecasts and saved picks are unchanged. Results below evaluate 2026 before the final refit; they are not prospective performance or proof of improvement over production.</p>
    <div class="comparison-controls"><label>Prediction target<select id="retrainingTask"><option value="winner">Game winner</option><option value="score">Final score</option></select></label></div>
    <p id="retrainingSelection"></p>
    <div class="table-wrap" tabindex="0"><table><thead id="retrainingHead"></thead><tbody id="retrainingRows"></tbody></table></div>
    <details><summary>Training and evaluation details</summary><p id="retrainingMethod"></p><p>Settings were selected on 2025. Most final winner models fit through 2025 and calibrate on completed 2026 games; GPR uses a bounded recent window and separate calibration weeks. Score models refit on all available games. Missing data is imputed within training partitions. Prior QB usage is not a confirmed starter. Historical source coverage omits some ties.</p><p id="retrainingVersion"></p></details>
    <p><a href="./data/retraining.json" download>Download retraining report</a> | <a href="./data/retraining_holdout.csv" download>Download holdout predictions</a></p>`;
  document.getElementById('labPanel').prepend(section);
  const el = id => document.getElementById(id);
  const esc = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = v => Number.isFinite(v) ? v.toFixed(3) : 'Unavailable';
  const labels = {logistic:'Logistic regression',random_forest:'Random forest',extra_trees:'Extra trees',hist_boost:'Histogram gradient boosting',gradient_boost:'Gradient boosting',adaboost:'AdaBoost',sgd_logistic:'SGD logistic regression',gaussian_nb:'Gaussian naive Bayes',shrinkage_lda:'Shrinkage linear discriminant analysis',regularized_qda:'Regularized quadratic discriminant analysis',neighbors:'Nearest neighbors',rbf_svm:'RBF support vector machine',linear_svm:'Linear support vector machine',ridge_classifier:'Ridge classifier',passive_aggressive:'Passive-aggressive classifier',neural_network:'Neural network',bagged_trees:'Bagged decision trees',bagged_logistic:'Bagged logistic regression',elastic_net:'Elastic-net logistic regression',spline_logistic:'Spline logistic regression',rbf_features:'Random RBF features',nystroem_logistic:'Nystroem logistic regression',nearest_centroid:'Nearest centroid',polynomial_svm:'Polynomial support vector machine',RandomForestRegressor:'Random forest',ExtraTreesRegressor:'Extra trees',Ridge:'Ridge regression',HistGradientBoostingRegressor:'Histogram gradient boosting'};
  function name(id) {
    if(id.startsWith('nn_')) { const p=id.split('_'); return `Small neural network (${p[2].replaceAll('x', ' + ')} units, regularization ${p[3]})`; }
    if(/^(ridge|poisson)_score_/.test(id)) {const p=id.split('_');return `${p[0]==='ridge'?'Ridge':'Poisson'} regression (regularization ${p[2]})`;}
    if(id.startsWith('gpr_'))return 'Noisy GPR: '+id.slice(4).replaceAll('_',' ').replace('rbf','RBF').replace('matern','Matern');
    const base=id.replace(/_(core|charted)_updated$/,'').replace(/^score_/,'');
    return labels[base] || base.replaceAll('_',' ');
  }
  window.candidateDisplayName = name;
  fetch('./data/retraining.json').then(r=>{if(!r.ok)throw Error('Unavailable');return r.json();}).then(d=>{
    el('retrainingStatus').textContent=`${d.models.length} saved configurations | Results through ${d.training_through} | ${d.current_season_games} completed 2026 games (${d.winner_holdout_games} decisive).`;
    el('retrainingMethod').textContent=d.note;
    el('retrainingVersion').textContent=`Generation: ${d.version}. Training-source rows: ${d.games}. scikit-learn ${d.sklearn_version}.`;
    const render=()=>{
      const task=el('retrainingTask').value, winner=task==='winner', selected=d.validation_selected[task];
      el('retrainingSelection').textContent=`Selected on 2025 validation: ${name(selected)}${selected.includes('charted')?' with charted inputs':''}. Rows ordered by 2025 validation ${winner?'log loss':'team-score error'}, not 2026 results. Lower loss/error is better.`;
      el('retrainingHead').innerHTML='<tr>'+['Model','Features','2025 validation',...(winner?['2026 accuracy','2026 log loss','2026 Brier score']:['2026 home MAE','2026 away MAE','2026 margin MAE','2026 total MAE'])].map(h=>`<th>${h}</th>`).join('')+'</tr>';
      const key=winner?'validation_log_loss':'validation_team_mae';
      el('retrainingRows').innerHTML=d.models.filter(m=>m.task===task).sort((a,b)=>a[key]-b[key]).map(m=>{
        const h=m.holdout, values=winner?[`${(h.accuracy*100).toFixed(1)}%`,number(h.log_loss),number(h.brier)]:[h.home_mae,h.away_mae,h.margin_mae,h.total_mae].map(number);
        return `<tr><td>${esc(name(m.name))}${m.name===selected?' (validation selected)':''}</td><td>${m.name.includes('charted')?'Core + charted + new metrics':'Core + new metrics'}</td><td>${number(m[key])}${winner?'':' points'}</td>${values.map(v=>`<td>${v}</td>`).join('')}</tr>`;
      }).join('');
    };
    el('retrainingTask').addEventListener('change',render);render();
  }).catch(()=>{el('retrainingStatus').textContent='Retraining report unavailable. Live forecasts are unaffected.';});
})();
