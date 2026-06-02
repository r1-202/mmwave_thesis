/* ============================================================================
   5G mmWave Research Portal — chart rendering script
   All data flows from DATA (built by build_dashboard_data.py).
   Headline metrics (RMSE/R²/MAE/latency) come straight from ml_results.json.
   ============================================================================ */
(function(){
'use strict';

const ML = DATA.ml, KP = DATA.kp;
const C = {
  cyan:'#2fe6c8', cyan2:'#19d5ff', blue:'#5b8cff', purple:'#9d7bff',
  pink:'#ff5ca8', amber:'#ffb13d', green:'#3ee08f', red:'#ff5d6c',
  orange:'#ff8a4c', mut:'#8ba0c4', line:'#1b2945', panel:'#0e1628', white:'#e8eefb'
};
const FONT = {color:'#cdd9f0', family:'Manrope,sans-serif', size:12};
const baseLay = () => ({
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:FONT,
  margin:{l:64, r:22, t:14, b:54},
  xaxis:{gridcolor:C.line, zerolinecolor:C.line, linecolor:C.line, ticks:'outside',
         tickcolor:C.line, title:{font:{size:12.5,color:C.mut}}},
  yaxis:{gridcolor:C.line, zerolinecolor:C.line, linecolor:C.line, ticks:'outside',
         tickcolor:C.line, title:{font:{size:12.5,color:C.mut}}},
  legend:{orientation:'v', x:1.02, y:1, font:{size:11},
          bgcolor:'rgba(14,22,40,.7)', bordercolor:C.line, borderwidth:1},
  hoverlabel:{bgcolor:'#0e1628', bordercolor:C.cyan2, font:{family:'JetBrains Mono,monospace',size:11.5,color:C.white}}
});
const SCENE = {
  xaxis:{gridcolor:C.line, backgroundcolor:'rgba(0,0,0,0)', color:'#9fb2d4', showbackground:true, title:{font:{size:11,color:C.mut}}},
  yaxis:{gridcolor:C.line, backgroundcolor:'rgba(0,0,0,0)', color:'#9fb2d4', showbackground:true, title:{font:{size:11,color:C.mut}}},
  zaxis:{gridcolor:C.line, backgroundcolor:'rgba(0,0,0,0)', color:'#9fb2d4', showbackground:true, title:{font:{size:11,color:C.mut}}},
  bgcolor:'rgba(0,0,0,0)',
  camera:{eye:{x:1.55, y:1.55, z:1.0}}
};
const CFG  = {responsive:true, displayModeBar:'hover', displaylogo:false,
              modeBarButtonsToRemove:['sendDataToCloud','lasso2d','select2d','toggleSpikelines'],
              toImageButtonOptions:{format:'png', filename:'5G_mmWave_chart', scale:2}};
const CFG3 = {...CFG};
const fmt = n => n>=1000 ? n.toLocaleString() : n;
const MODEL_LBL = {
  RandomForest:'RandomForest', StackingEnsemble:'Stacking', XGBoost:'XGBoost',
  LightGBM:'LightGBM', GradientBoost:'GradientBoost', SVR:'SVR',
  Traditional_ITU:'ITU-R P.838-3', Traditional_Crane:'Crane (1980)'
};
const isTrad = k => k.startsWith('Traditional');
const MCOL = k => isTrad(k) ? C.purple : C.cyan2;

/* ============================================================================
   KPI cards (hero strip)
   ============================================================================ */
(function(){
  const best = ML.models[ML.best_model], o = KP.overall;
  const trad = ML.models.Traditional_ITU.rmse;
  const impr = ((trad - best.rmse)/trad*100).toFixed(1);
  const cards = [
    ['38 GHz',          C.cyan,   'Carrier',       'Real CoMMon link'],
    [fmt(ML.n_rows),    C.green,  'Real Rows',     'Measured attenuation'],
    ['6',               C.purple, 'ML Models',     'RF·XGB·LGB·GBM·SVR·Stack'],
    [impr+'%',          C.red,    'Lower RMSE',    'ML vs ITU-R'],
    [best.rmse.toFixed(3), C.cyan2, 'Best RMSE (dB)', ML.best_model],
    [best.r2.toFixed(3),   C.green, 'Best R²',     'Real measured data'],
    [o.bf_gain_db+' dB',   C.amber, 'BF Gain',     '8×8 hybrid array'],
    ['10',                 C.pink,  'LB Cities',   '5G coverage map']
  ];
  document.getElementById('kpis').innerHTML = cards.map(c =>
    `<div class="kpi" style="--c:${c[1]}"><div class="v">${c[0]}</div>
     <div class="l">${c[2]}</div><div class="s">${c[3]}</div></div>`).join('');
  document.getElementById('rowsTxt').textContent = fmt(ML.n_rows);
  document.getElementById('rowsPill').textContent = fmt(ML.n_rows)+' rows';
})();

/* ============================================================================
   Leaderboard table
   ============================================================================ */
(function(){
  const order = ['RandomForest','StackingEnsemble','XGBoost','LightGBM',
                 'GradientBoost','SVR','Traditional_ITU','Traditional_Crane'];
  const medal = ['#ffd84d','#cfd8e8','#e0a072'];
  const maxr  = Math.max(...order.map(k => ML.models[k] ? ML.models[k].rmse : 0));
  document.getElementById('lbody').innerHTML = order.filter(k => ML.models[k]).map((k,i) => {
    const m  = ML.models[k];
    const bw = Math.max(4, (1 - m.rmse/maxr) * 100);
    const rc = i<3 ? `style="background:${medal[i]};color:#1a1205"`
                   : 'style="background:var(--panel2);color:var(--mut)"';
    return `<tr><td><div class="rank" ${rc}>${i+1}</div></td>
      <td class="mdl">${MODEL_LBL[k]}</td>
      <td><span class="tt ${isTrad(k)?'trad':''}">${isTrad(k)?'PHYSICS':'ML'}</span></td>
      <td class="mono">${m.rmse.toFixed(4)}</td>
      <td class="mono">${m.mae.toFixed(4)}</td>
      <td><div class="bar" style="width:${bw}%;${isTrad(k)?'background:linear-gradient(90deg,var(--purple),var(--pink))':''}"></div></td>
      <td class="mono">${m.latency_us.toFixed(3)} µs</td>
      <td class="mono" style="color:${m.r2>0.8?'var(--green)':m.r2>0.5?'var(--amber)':'var(--red)'}">${m.r2.toFixed(4)}</td></tr>`;
  }).join('');
})();

/* ============================================================================
   SECTION 02 — Per-metric Traditional vs ML comparisons
   ============================================================================ */
(function(){
  const ORD = DATA.models_order;
  const labels = ORD.map(k => MODEL_LBL[k]);
  const colors = ORD.map(k => MCOL(k));

  // --- Prediction Accuracy: grouped RMSE + MAE
  const lay1 = baseLay();
  lay1.xaxis.title.text = 'Model';
  lay1.yaxis.title.text = 'Error (dB) — lower is better';
  lay1.barmode = 'group';
  lay1.margin = {l:64, r:22, t:14, b:90};
  lay1.xaxis.tickangle = -28;
  lay1.legend = {orientation:'h', y:-0.34, x:0.5, xanchor:'center'};
  Plotly.newPlot('cAcc', [
    {type:'bar', name:'RMSE (dB)', x:labels, y:ORD.map(k=>ML.models[k].rmse),
     marker:{color:C.cyan2, line:{color:C.cyan,width:1}},
     text:ORD.map(k=>ML.models[k].rmse.toFixed(3)), textposition:'outside',
     hovertemplate:'<b>%{x}</b><br>RMSE: %{y:.4f} dB<extra></extra>'},
    {type:'bar', name:'MAE (dB)', x:labels, y:ORD.map(k=>ML.models[k].mae),
     marker:{color:C.purple, line:{color:'#7c5ce0',width:1}},
     text:ORD.map(k=>ML.models[k].mae.toFixed(3)), textposition:'outside',
     hovertemplate:'<b>%{x}</b><br>MAE: %{y:.4f} dB<extra></extra>'}
  ], lay1, CFG);

  // --- R²
  const lay2 = baseLay();
  lay2.xaxis.title.text = 'Model';
  lay2.yaxis.title.text = 'R² Score';
  lay2.yaxis.range = [-0.3, 1.0];
  lay2.margin = {l:64, r:22, t:14, b:90};
  lay2.xaxis.tickangle = -28;
  lay2.showlegend = false;
  Plotly.newPlot('cR2', [{
    type:'bar', x:labels, y:ORD.map(k=>ML.models[k].r2),
    marker:{color:ORD.map(k=>ML.models[k].r2>0.8?C.green:ML.models[k].r2>0.5?C.amber:C.red),
            line:{color:'#11203d',width:1}},
    text:ORD.map(k=>ML.models[k].r2.toFixed(3)), textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
  }], lay2, CFG);

  // --- BER (log)
  const lay3 = baseLay();
  lay3.xaxis.title.text = 'Model';
  lay3.yaxis.title.text = 'BER @ 11 dB operating SNR';
  lay3.yaxis.type = 'log';
  lay3.yaxis.exponentformat = 'power';
  lay3.margin = {l:74, r:22, t:14, b:90};
  lay3.xaxis.tickangle = -28;
  lay3.showlegend = false;
  Plotly.newPlot('cBER', [{
    type:'bar', x:labels, y:ORD.map(k=>DATA.model_ber[k].ber),
    marker:{color:colors, line:{color:'#11203d',width:1}},
    text:ORD.map(k=>DATA.model_ber[k].ber.toExponential(2)), textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>Eff. SNR: '+ORD.map(k=>DATA.model_ber[k].snr_eff.toFixed(2)).join('|')+' dB<br>BER: %{y:.2e}<extra></extra>'
  }], lay3, CFG);

  // --- Latency (log µs)
  const lay4 = baseLay();
  lay4.xaxis.title.text = 'Model';
  lay4.yaxis.title.text = 'Inference latency (µs / sample)';
  lay4.yaxis.type = 'log';
  lay4.margin = {l:74, r:22, t:14, b:90};
  lay4.xaxis.tickangle = -28;
  lay4.showlegend = false;
  const lats = ORD.map(k => DATA.latency[k]);
  Plotly.newPlot('cLat', [{
    type:'bar', x:labels, y:lats,
    marker:{color:colors, line:{color:'#11203d',width:1}},
    text:lats.map(v => v.toFixed(2)+' µs'), textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>Latency: %{y} µs<extra></extra>'
  }], lay4, CFG);

  // --- Spectral Efficiency: No BF vs Hybrid BF
  const o = KP.overall;
  const lay5 = baseLay();
  lay5.xaxis.title.text = 'Beamforming configuration';
  lay5.yaxis.title.text = 'Mean spectral efficiency (bps/Hz)';
  lay5.showlegend = false;
  Plotly.newPlot('cSE', [{
    type:'bar', x:['No Beamforming','8×8 Hybrid BF'], y:[o.mean_SE_noBF, o.mean_SE_BF],
    marker:{color:[C.mut, C.green], line:{color:'#11203d',width:1}},
    text:[o.mean_SE_noBF.toFixed(2)+' bps/Hz', o.mean_SE_BF.toFixed(2)+' bps/Hz'], textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>SE: %{y:.3f} bps/Hz<extra></extra>'
  }], lay5, CFG);

  // --- SNR: No BF vs BF
  const lay6 = baseLay();
  lay6.xaxis.title.text = 'Beamforming configuration';
  lay6.yaxis.title.text = 'Mean received SNR (dB)';
  lay6.showlegend = false;
  Plotly.newPlot('cSNR', [{
    type:'bar', x:['No Beamforming','8×8 Hybrid BF'], y:[o.mean_snr_noBF, o.mean_snr_BF],
    marker:{color:[C.mut, C.cyan2], line:{color:'#11203d',width:1}},
    text:[o.mean_snr_noBF.toFixed(1)+' dB', o.mean_snr_BF.toFixed(1)+' dB'], textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>SNR: %{y:.2f} dB<extra></extra>'
  }], lay6, CFG);

  // Insight cards under comparisons
  const best = ML.models[ML.best_model], itu = ML.models.Traditional_ITU;
  const impr = ((itu.rmse - best.rmse)/itu.rmse*100).toFixed(1);
  const fi = ML.feature_importance;
  const topF = Object.keys(fi).sort((a,b)=>fi[b]-fi[a]).slice(0,3)
    .map(k => `${k} (${(fi[k]*100).toFixed(1)}%)`).join(', ');
  document.getElementById('insights').innerHTML = `
    <div class="ins green"><h5>🏆 Best ML Model</h5><p><b>${ML.best_model}</b> achieved RMSE = ${best.rmse.toFixed(4)} dB and R² = ${best.r2.toFixed(3)} on real measured attenuation, outperforming all other models tested.</p></div>
    <div class="ins cyan"><h5>📉 ${impr}% RMSE Reduction</h5><p>ML reduced prediction error by ${impr}% versus ITU-R P.838-3 (RMSE ${itu.rmse.toFixed(3)} dB → ${best.rmse.toFixed(3)} dB), demonstrating the value of data-driven prediction.</p></div>
    <div class="ins purple"><h5>📡 Hybrid Beamforming</h5><p>OMP-based hybrid BF achieves <b>${DATA.bf_vs_att.hybrid_ratio_pct}%</b> of fully-digital spectral efficiency while using only 4 RF chains instead of 64.</p></div>
    <div class="ins amber"><h5>🌧 Key Weather Features</h5><p>The top drivers of measured attenuation are <b>${topF}</b>, confirming the physics — rain dominates over gaseous and humidity effects.</p></div>`;
})();

/* ============================================================================
   SECTION 03 — Attenuation & dataset analysis
   ============================================================================ */

// --- Main: Measured vs ITU-R vs Crane vs rain rate
(function(){
  const s = KP.sweep;
  const lay = baseLay();
  lay.xaxis.title.text = 'Rain rate (mm/hr)';
  lay.yaxis.title.text = 'Rain-induced attenuation (dB)';
  lay.legend = {orientation:'h', y:-0.18, x:0.5, xanchor:'center', font:{size:12}};
  lay.hovermode = 'x unified';
  Plotly.newPlot('mainChart', [
    {x:s.rain, y:s.real, name:'Real measured (38 GHz link)', mode:'markers+lines',
     marker:{color:C.green, size:9, line:{color:'#1a3826',width:1}},
     line:{color:C.green, width:3}, hovertemplate:'%{y:.3f} dB<extra>Measured</extra>'},
    {x:s.rain, y:s.itu38, name:'ITU-R P.838-3 (predicted)', mode:'lines+markers',
     marker:{color:C.cyan2, size:5}, line:{color:C.cyan2, width:2.5, dash:'dot'},
     hovertemplate:'%{y:.3f} dB<extra>ITU-R</extra>'},
    {x:s.rain, y:s.crane38, name:'Crane 1980 (predicted)', mode:'lines+markers',
     marker:{color:C.amber, size:5}, line:{color:C.amber, width:2.5, dash:'dash'},
     hovertemplate:'%{y:.3f} dB<extra>Crane</extra>'}
  ], lay, CFG);
})();

// --- Predicted vs Actual scatter
(function(){
  const ps = DATA.predscatter;
  const maxv = Math.max(...ps.true, ...ps.rf) * 1.05;
  const lay = baseLay();
  lay.xaxis.title.text = 'True attenuation (dB)';
  lay.yaxis.title.text = 'Predicted attenuation (dB)';
  lay.yaxis.range = [-0.1, maxv]; lay.xaxis.range = [-0.1, maxv];
  Plotly.newPlot('cPred', [
    {type:'scatter', mode:'markers', x:ps.true, y:ps.rf, name:'RandomForest',
     marker:{size:5, color:ps.true, colorscale:'Viridis', opacity:0.75,
             showscale:true, colorbar:{title:{text:'True (dB)', font:{size:10}}, thickness:10, len:0.6}},
     hovertemplate:'True: %{x:.3f} dB<br>Pred: %{y:.3f} dB<extra></extra>'},
    {type:'scatter', mode:'lines', x:[0, maxv], y:[0, maxv], name:'Perfect',
     line:{color:C.red, dash:'dash', width:2}, hoverinfo:'skip'}
  ], lay, CFG);
})();

// --- All models vs ground truth (line plot sorted)
(function(){
  const s = DATA.sorted, idx = s.true.map((_,i)=>i);
  const lay = baseLay();
  lay.xaxis.title.text = 'Test sample index (sorted by true attenuation)';
  lay.yaxis.title.text = 'Attenuation (dB)';
  lay.legend = {orientation:'h', y:-0.18, x:0.5, xanchor:'center', font:{size:11}};
  Plotly.newPlot('cSorted', [
    {x:idx, y:s.true,  name:'Ground Truth',     mode:'lines', line:{color:'#ffffff', width:2.5}},
    {x:idx, y:s.itu,   name:'ITU-R P.838-3',    mode:'lines', line:{color:C.orange, width:1.2, dash:'dash'}, opacity:.85},
    {x:idx, y:s.crane, name:'Crane (1980)',     mode:'lines', line:{color:C.amber, width:1.2, dash:'dashdot'}, opacity:.85},
    {x:idx, y:s.rf,    name:'RandomForest (ML)', mode:'lines', line:{color:C.green, width:1.6}}
  ], lay, CFG);
})();

// --- Residual analysis
(function(){
  const r = DATA.residual_clean;
  const lay = baseLay();
  lay.xaxis.title.text = 'True attenuation (dB)';
  lay.yaxis.title.text = 'Residual = predicted − true (dB)';
  lay.shapes = [{type:'line', x0:0, x1:Math.max(...r.true)*1.05, y0:0, y1:0,
                 line:{color:C.red, dash:'dash', width:2}}];
  Plotly.newPlot('cResid', [{
    type:'scatter', mode:'markers', x:r.true, y:r.res,
    marker:{size:4.5, color:r.res, colorscale:[[0,C.red],[0.5,'#9a9a9a'],[1,C.green]],
            cmin:-1.5, cmax:1.5, opacity:0.75,
            showscale:true, colorbar:{title:{text:'Residual', font:{size:10}}, thickness:10, len:0.6}},
    hovertemplate:'True: %{x:.3f} dB<br>Residual: %{y:.3f} dB<extra></extra>'
  }], lay, CFG);
})();

// --- Histogram
(function(){
  const lay = baseLay();
  lay.xaxis.title.text = '38 GHz measured attenuation (dB)';
  lay.yaxis.title.text = 'Count (samples)';
  lay.yaxis.type = 'log';
  lay.bargap = 0.03;
  lay.showlegend = false;
  Plotly.newPlot('cHist', [{
    type:'bar', x:KP.hist.edges.slice(0,-1), y:KP.hist.counts,
    marker:{color:C.cyan2, line:{color:C.cyan, width:.5}},
    hovertemplate:'%{x:.2f} dB<br>Count: %{y}<extra></extra>'
  }], lay, CFG);
})();

// --- Feature importance
(function(){
  const fi = ML.feature_importance;
  const niceNames = {Temperature:'Temperature', Humidity:'Humidity',
    'Wind Speed':'Wind Speed', Precipitation:'Precipitation (Rain)', WC:'Weather Condition'};
  const ks = Object.keys(fi).sort((a,b) => fi[a] - fi[b]);
  const lay = baseLay();
  lay.xaxis.title.text = 'Feature importance (Mean Decrease in Impurity)';
  lay.margin = {l:135, r:48, t:14, b:54};
  lay.showlegend = false;
  Plotly.newPlot('cFI', [{
    type:'bar', orientation:'h',
    y:ks.map(k => niceNames[k] || k),
    x:ks.map(k => fi[k]),
    marker:{color:ks.map((_,i,a) =>
      i === a.length-1 ? C.pink : i === a.length-2 ? C.purple : C.blue),
      line:{color:'#11203d', width:1}},
    text:ks.map(k => (fi[k]*100).toFixed(1)+'%'), textposition:'outside',
    hovertemplate:'<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
  }], lay, CFG);
})();

// --- Mean attenuation by condition
(function(){
  const bc = KP.by_condition, conds = Object.keys(bc);
  const cmap = {Sunny:C.amber, Cloudy:C.cyan2, Rainy:C.blue};
  const lay = baseLay();
  lay.xaxis.title.text = 'Weather condition';
  lay.yaxis.title.text = 'Mean measured attenuation (dB)';
  lay.showlegend = false;
  Plotly.newPlot('cCond', [{
    type:'bar', x:conds, y:conds.map(c => bc[c].att),
    marker:{color:conds.map(c => cmap[c] || C.cyan), line:{color:'#11203d',width:1}},
    text:conds.map(c => bc[c].att.toFixed(3)+' dB  (n='+(bc[c].n).toLocaleString()+')'),
    textposition:'outside',
    hovertemplate:'<b>%{x}</b><br>Mean attenuation: %{y:.3f} dB<extra></extra>'
  }], lay, CFG);
})();

/* ============================================================================
   SECTION 04 — Link Budget & Beamforming
   ============================================================================ */

// --- BER vs SNR + model operating points
(function(){
  const b = DATA.ber_curves;
  const traces = [
    {x:b.snr, y:b.qpsk,  name:'QPSK (theory)',   mode:'lines', line:{color:C.cyan2,  width:2.5}},
    {x:b.snr, y:b.qam16, name:'16-QAM (theory)', mode:'lines', line:{color:C.amber,  width:2, dash:'dash'}},
    {x:b.snr, y:b.qam64, name:'64-QAM (theory)', mode:'lines', line:{color:C.pink,   width:2, dash:'dot'}}
  ];
  const ORD = DATA.models_order;
  ORD.forEach(k => {
    const v = DATA.model_ber[k];
    traces.push({x:[v.snr_eff], y:[v.ber], name:MODEL_LBL[k], mode:'markers',
      marker:{size:11, color:isTrad(k)?C.purple:C.green, symbol:isTrad(k)?'diamond':'circle',
              line:{color:C.white, width:1.5}},
      hovertemplate:'<b>'+MODEL_LBL[k]+'</b><br>Op. SNR: %{x:.2f} dB<br>BER: %{y:.2e}<extra></extra>'});
  });
  const lay = baseLay();
  lay.xaxis.title.text = 'Eb/N0 or SNR (dB)';
  lay.yaxis.title.text = 'Bit Error Rate (log scale)';
  lay.yaxis.type = 'log'; lay.yaxis.range = [-12, 0]; lay.yaxis.exponentformat='power';
  lay.legend = {orientation:'v', x:1.02, y:1, font:{size:10}, bgcolor:'rgba(14,22,40,.7)', bordercolor:C.line, borderwidth:1};
  Plotly.newPlot('cBERsnr', traces, lay, CFG);
})();

// --- SE vs SNR (Shannon + Hybrid + Analog)
(function(){
  const s = DATA.se_snr;
  const lay = baseLay();
  lay.xaxis.title.text = 'SNR (dB)';
  lay.yaxis.title.text = 'Spectral efficiency (bps/Hz)';
  lay.legend = {orientation:'h', y:-0.2, x:0.5, xanchor:'center'};
  Plotly.newPlot('cSEsnr', [
    {x:s.snr, y:s.digital, name:'Shannon Capacity',          mode:'lines', line:{color:C.white, width:2.5}},
    {x:s.snr, y:s.hybrid,  name:'Hybrid BF (92%)',           mode:'lines', line:{color:C.green, width:2.5, dash:'dash'}},
    {x:s.snr, y:s.analog,  name:'Analog BF (75%)',           mode:'lines', line:{color:C.red,   width:2.5, dash:'dashdot'}}
  ], lay, CFG);
})();

// --- BF SE vs attenuation
(function(){
  const b = DATA.bf_vs_att;
  const lay = baseLay();
  lay.xaxis.title.text = 'Path attenuation (dB)';
  lay.yaxis.title.text = 'Spectral efficiency (bps/Hz)';
  lay.legend = {orientation:'h', y:-0.2, x:0.5, xanchor:'center'};
  Plotly.newPlot('cBFse', [
    {x:b.att, y:b.se_digital, name:'Fully Digital (8 streams)', mode:'lines+markers', line:{color:C.blue,  width:2.5}, marker:{size:6}},
    {x:b.att, y:b.se_hybrid,  name:'Hybrid OMP (4 RF chains)',  mode:'lines+markers', line:{color:C.green, width:2.5, dash:'dash'}, marker:{size:6}},
    {x:b.att, y:b.se_analog,  name:'Analog Only (1 stream)',    mode:'lines+markers', line:{color:C.red,   width:2.5, dash:'dot'},  marker:{size:6}}
  ], lay, CFG);
})();

// --- BF SNR vs attenuation
(function(){
  const b = DATA.bf_vs_att;
  const lay = baseLay();
  lay.xaxis.title.text = 'Path attenuation (dB)';
  lay.yaxis.title.text = 'Received SNR (dB)';
  lay.legend = {orientation:'h', y:-0.2, x:0.5, xanchor:'center'};
  Plotly.newPlot('cBFsnr', [
    {x:b.att, y:b.snr_digital, name:'Digital', mode:'lines+markers', line:{color:C.blue,  width:2.5}, marker:{size:6}},
    {x:b.att, y:b.snr_hybrid,  name:'Hybrid',  mode:'lines+markers', line:{color:C.green, width:2.5, dash:'dash'}, marker:{size:6}},
    {x:b.att, y:b.snr_analog,  name:'Analog',  mode:'lines+markers', line:{color:C.red,   width:2.5, dash:'dot'},  marker:{size:6}}
  ], lay, CFG);
})();

// --- Throughput vs distance
(function(){
  const t = DATA.throughput;
  const colors = {Clear:C.green, Light:C.cyan2, Moderate:C.amber, Heavy:C.orange, Extreme:C.red};
  const lay = baseLay();
  lay.xaxis.title.text = 'Link distance (km)';
  lay.yaxis.title.text = 'Throughput (Mbps)';
  lay.yaxis.type = 'log';
  lay.legend = {orientation:'h', y:-0.2, x:0.5, xanchor:'center'};
  Plotly.newPlot('cTput',
    ['Clear','Light','Moderate','Heavy','Extreme'].map(name => ({
      x:t.dist, y:t[name], name:name+' rain', mode:'lines',
      line:{color:colors[name], width:2.5},
      hovertemplate:name+'<br>%{x:.2f} km — %{y:.1f} Mbps<extra></extra>'
    })), lay, CFG);
})();

// --- Multi-band attenuation
(function(){
  const m = DATA.multiband;
  const bands = Object.keys(m).filter(k => k !== 'rain');
  const palette = {'3.5 GHz (Sub-6)':C.green, '24 GHz':C.cyan2, '28 GHz (5G NR)':C.purple,
                   '39 GHz':C.amber, '60 GHz (WiGig)':C.red, '77 GHz':C.pink};
  const lay = baseLay();
  lay.xaxis.title.text = 'Rain rate (mm/hr)';
  lay.yaxis.title.text = 'Specific attenuation (dB/km)';
  lay.yaxis.type = 'log';
  lay.legend = {orientation:'v', x:1.02, y:1, font:{size:11}, bgcolor:'rgba(14,22,40,.7)', bordercolor:C.line, borderwidth:1};
  Plotly.newPlot('cMB',
    bands.map(b => ({x:m.rain, y:m[b], name:b, mode:'lines', line:{color:palette[b], width:2.5}})),
    lay, CFG);
})();

// --- SNR CDF: BF vs no BF
(function(){
  const s = DATA.snr_cdf;
  const lay = baseLay();
  lay.xaxis.title.text = 'Received SNR (dB)';
  lay.yaxis.title.text = 'Cumulative probability (%)';
  lay.legend = {orientation:'h', y:-0.22, x:0.5, xanchor:'center'};
  Plotly.newPlot('cCDF', [
    {x:s.snr, y:s.cdf_nobf, name:'No Beamforming',        mode:'lines', line:{color:C.red, width:3}},
    {x:s.snr, y:s.cdf_bf,   name:'8×8 Hybrid Beamforming', mode:'lines', line:{color:C.green, width:3, dash:'dash'}}
  ], lay, CFG);
})();

/* ============================================================================
   SECTION 05 — 3D Visualizations
   ============================================================================ */
function linspace(a, b, n){ return Array.from({length:n}, (_,i) => a + (b-a)*i/(n-1)); }

// --- ① ITU-R attenuation surface: rain × frequency
(function(){
  const rain  = linspace(0, 60, 32);
  const freqs = linspace(10, 80, 32);
  const ITUK  = f => f<=28 ? 0.00041*Math.pow(f,2.4) : f<=60 ? 0.0001*Math.pow(f,2.85) : 0.0008*Math.pow(f,2.55);
  const ITUA  = f => Math.max(0.55, 1.4 - 0.012*f);
  const z = freqs.map(f => rain.map(r => +(ITUK(f) * Math.pow(Math.max(r,0), ITUA(f))).toFixed(3)));
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'Rain rate (mm/hr)', font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'Frequency (GHz)',    font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'γ_R (dB/km)',         font:{size:11,color:C.mut}}}};
  Plotly.newPlot('d1', [{type:'surface', x:rain, y:freqs, z:z, colorscale:'Plasma',
    showscale:true, colorbar:{title:{text:'γ_R (dB/km)',font:{size:10}},thickness:12,len:0.7}}],
    lay, CFG3);
})();

// --- ② Received SNR surface: rain × distance
(function(){
  const rain = linspace(0, 50, 30);
  const dist = linspace(0.2, 4.0, 30);
  const k28 = 0.2051, a28 = 0.9679;
  const z = dist.map(d => rain.map(r => {
    const red = 1.0 / (1.0 + d/35.0 * Math.pow(Math.max(r,0), 0.1));
    const att = k28 * Math.pow(Math.max(r,0), a28) * d * Math.max(0.4, Math.min(1.0,red)) + 0.12*d;
    const fspl = 20*Math.log10(d*1000) + 20*Math.log10(28e9) + 20*Math.log10(4*Math.PI/3e8);
    const noise = 10*Math.log10(1.38e-23*290*400e6*1e3) + 7;
    return +(30 + 24 + 24 - fspl - att - noise + 18.06).toFixed(2);
  }));
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'Rain rate (mm/hr)', font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'Distance (km)',     font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'SNR (dB)',           font:{size:11,color:C.mut}}}};
  Plotly.newPlot('d2', [{type:'surface', x:rain, y:dist, z:z, colorscale:'RdBu', reversescale:true,
    showscale:true, colorbar:{title:{text:'SNR (dB)',font:{size:10}},thickness:12,len:0.7}}],
    lay, CFG3);
})();

// --- ③ Weather cloud (REAL test samples)
(function(){
  const ps = DATA.predscatter;
  // need temp/hum at same indices → use predscatter, infer temp/hum is not stored, so use KP.scatter
  const sc = KP.scatter;
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'Rain rate (mm/hr)', font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'Temperature (°C)',  font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'Humidity (%)',       font:{size:11,color:C.mut}}}};
  Plotly.newPlot('d3', [{type:'scatter3d', mode:'markers',
    x:sc.rain, y:sc.temp, z:sc.hum,
    marker:{size:3.4, color:sc.att, colorscale:'Turbo', opacity:0.78,
            showscale:true, colorbar:{title:{text:'Measured<br>att (dB)',font:{size:10}},thickness:12,len:0.7}},
    hovertemplate:'Rain: %{x:.2f} mm/hr<br>Temp: %{y:.1f} °C<br>Humidity: %{z:.1f} %%<br>Att: %{marker.color:.3f} dB<extra></extra>'}],
    lay, CFG3);
})();

// --- ④ Spectral efficiency surface
(function(){
  const snr = linspace(-5, 35, 30);
  const nel = linspace(1, 64, 30);
  const z = nel.map(n => snr.map(s => {
    const eff_snr = s + 10*Math.log10(n);
    const lin = Math.pow(10, eff_snr/10);
    return +(Math.min(12, Math.log2(1 + lin))).toFixed(3);
  }));
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'Base SNR (dB)',     font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'BF elements N',     font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'SE (bps/Hz)',        font:{size:11,color:C.mut}}}};
  Plotly.newPlot('d4', [{type:'surface', x:snr, y:nel, z:z, colorscale:'Viridis',
    showscale:true, colorbar:{title:{text:'SE',font:{size:10}},thickness:12,len:0.7}}],
    lay, CFG3);
})();

// --- ⑥ BER surface  (log10)
(function(){
  const snr  = linspace(-5, 30, 30);
  const rain = linspace(0, 50, 30);
  const erfc = x => {
    const t = 1/(1 + 0.3275911*Math.abs(x));
    const y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*Math.exp(-x*x);
    return x>=0 ? 1-y : 1+y;
  };
  const z = rain.map(r => snr.map(s => {
    const att = 0.2051 * Math.pow(Math.max(r,0), 0.97) * 1.85;
    const eff = s - att*0.6;
    const lin = Math.pow(10, Math.min(eff, 15)/10);
    return +Math.log10(Math.max(0.5*erfc(Math.sqrt(Math.max(lin,0))), 1e-12)).toFixed(3);
  }));
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'SNR (dB)',         font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'Rain (mm/hr)',     font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'log₁₀ BER',         font:{size:11,color:C.mut}}}};
  Plotly.newPlot('d6', [{type:'surface', x:snr, y:rain, z:z, colorscale:'Hot', reversescale:true,
    showscale:true, colorbar:{title:{text:'log10 BER',font:{size:10}},thickness:12,len:0.7}}],
    lay, CFG3);
})();

// --- ⑦ Model comparison cube
(function(){
  const mk = DATA.models_order;
  const lay = baseLay();
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.scene = {...SCENE,
    xaxis:{...SCENE.xaxis, title:{text:'RMSE (dB)',        font:{size:11,color:C.mut}}},
    yaxis:{...SCENE.yaxis, title:{text:'R² Score',          font:{size:11,color:C.mut}}},
    zaxis:{...SCENE.zaxis, title:{text:'Latency (µs, log)', font:{size:11,color:C.mut}}, type:'log'}};
  Plotly.newPlot('d7', [{type:'scatter3d', mode:'markers+text',
    x:mk.map(k=>ML.models[k].rmse),
    y:mk.map(k=>ML.models[k].r2),
    z:mk.map(k=>ML.models[k].latency_us),
    text:mk.map(k=>MODEL_LBL[k]),
    textposition:'top center', textfont:{size:10, color:C.white},
    marker:{size:mk.map(k=>10 + (1-ML.models[k].rmse/Math.max(...mk.map(x=>ML.models[x].rmse)))*14),
            color:mk.map(k=>isTrad(k)?C.purple:C.cyan2),
            line:{color:C.white, width:1.5}, opacity:0.9},
    hovertemplate:'<b>%{text}</b><br>RMSE: %{x:.4f} dB<br>R²: %{y:.4f}<br>Latency: %{z:.2f} µs<extra></extra>'}],
    lay, CFG3);
})();

/* ============================================================================
   SECTION 06 — Figure Gallery (visual index of every chart)
   ============================================================================ */
(function(){
  // mini SVG thumbnails: each is a tiny representative of the real chart
  const grad = (id,c1,c2) => `<defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="${c1}"/><stop offset="100%" stop-color="${c2}"/></linearGradient></defs>`;
  const bg   = `<rect width="100%" height="100%" fill="#0a1020"/>`;
  const bars = (vals, color) => vals.map((v,i)=>{
    const w = 100/vals.length, x = i*w + 2;
    const h = v*72; return `<rect x="${x}" y="${82-h}" width="${w-4}" height="${h}" rx="2" fill="${color}"/>`;
  }).join('');
  const polyline = (pts, color, w=2) => `<polyline fill="none" stroke="${color}" stroke-width="${w}" points="${pts}"/>`;
  const dots = (pts, color, r=2) => pts.map(p=>`<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${color}"/>`).join('');

  const makeSvg = inner => `<svg class="galthumb" viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">${bg}${inner}</svg>`;

  const cats = [
    {title:'⭐ Grand Comparison', items:[
      {id:'mainChart', name:'Measured vs ITU-R vs Crane', desc:'Rain rate sweep · 38 GHz',
       svg: makeSvg(grad('g0',C.green,C.cyan)+
         polyline('5,90 25,80 45,68 65,58 85,55 105,48 125,42 145,38 165,33 195,28', C.green, 2.5)+
         polyline('5,95 25,88 45,80 65,73 85,67 105,61 125,56 145,52 165,49 195,46', C.cyan2, 1.8)+
         polyline('5,98 25,93 45,87 65,82 85,77 105,73 125,70 145,67 165,64 195,62', C.amber, 1.8))},
      {id:'cSorted', name:'All Models vs Ground Truth', desc:'Sorted predictions',
       svg: makeSvg(polyline('5,55 30,50 55,45 80,40 105,35 130,30 155,25 195,18', C.white, 2)+
         polyline('5,58 30,52 55,47 80,43 105,38 130,32 155,28 195,22', C.green, 1.5)+
         polyline('5,40 30,42 55,44 80,46 105,48 130,50 155,52 195,55', C.amber, 1.2)+
         polyline('5,35 30,38 55,42 80,46 105,50 130,55 155,58 195,62', C.orange, 1.2))}]},
    {title:'🧠 Machine Learning', items:[
      {id:'cAcc',  name:'RMSE & MAE', desc:'Grouped error bars',
       svg: makeSvg(bars([.32,.38,.45,.55,.62,.7,.95,1],C.cyan2)+
         '<g opacity=".7">'+bars([.13,.18,.22,.25,.27,.32,.50,.55],C.purple)+'</g>')},
      {id:'cR2',   name:'R² Score', desc:'Variance explained',
       svg: makeSvg(bars([.91,.87,.83,.79,.65,.57,.24,.09],C.green))},
      {id:'cFI',   name:'Feature Importance', desc:'RandomForest MDI',
       svg: makeSvg(`<rect x="2" y="14" width="160" height="12" rx="2" fill="${C.pink}"/>`+
         `<rect x="2" y="34" width="40" height="12" rx="2" fill="${C.purple}"/>`+
         `<rect x="2" y="54" width="58" height="12" rx="2" fill="${C.blue}"/>`+
         `<rect x="2" y="74" width="38" height="12" rx="2" fill="${C.blue}"/>`+
         `<rect x="2" y="94" width="14" height="12" rx="2" fill="${C.blue}"/>`)},
      {id:'cPred', name:'Predicted vs Actual', desc:'RandomForest scatter',
       svg: makeSvg(dots([[20,90],[35,82],[40,78],[55,68],[60,62],[75,55],[85,48],[100,40],[115,35],[130,28],[150,22],[170,18],[45,75],[80,55],[110,38],[140,30],[28,85],[70,58]], C.cyan2)+
         `<line x1="10" y1="100" x2="195" y2="10" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="4,3"/>`)},
      {id:'cResid', name:'Residual Analysis', desc:'Error vs true',
       svg: makeSvg(`<line x1="10" y1="55" x2="195" y2="55" stroke="${C.red}" stroke-width="1.5" stroke-dasharray="4,3"/>`+
         dots([[20,40],[30,38],[40,42],[55,55],[70,60],[85,68],[100,72],[120,75],[140,78],[160,82],[35,48],[65,52],[90,62]], C.green))},
      {id:'cHist', name:'Att. Distribution', desc:'Long-tailed histogram',
       svg: makeSvg(`<g>`+[88,42,28,18,12,8,6,4,3,2,2,1,1,1].map((h,i)=>
         `<rect x="${5+i*13}" y="${100-h}" width="11" height="${h}" fill="${C.cyan2}"/>`).join('')+`</g>`)}]},
    {title:'📡 Traditional', items:[
      {id:'cBER',  name:'BER vs Model', desc:'Operating-point spread',
       svg: makeSvg(bars([.05,.08,.12,.16,.28,.38,.65,.85], C.amber))},
      {id:'cLat',  name:'Inference Latency', desc:'µs/sample, log',
       svg: makeSvg(bars([.42,.55,.18,.25,.05,.92,.04,.04], C.purple))},
      {id:'cCond', name:'By Weather Condition', desc:'Sunny / Cloudy / Rainy',
       svg: makeSvg('<rect x="20" y="78" width="40" height="22" fill="'+C.amber+'"/>'+
         '<rect x="80" y="62" width="40" height="38" fill="'+C.cyan2+'"/>'+
         '<rect x="140" y="22" width="40" height="78" fill="'+C.blue+'"/>')}]},
    {title:'🧊 3D Gallery', items:[
      {id:'d1', name:'γ_R Surface', desc:'Rain × Frequency',
       svg: makeSvg(grad('s1',C.amber,C.red)+'<polygon fill="url(#s1)" points="10,90 60,55 110,30 160,15 190,30 150,55 100,70 50,80"/>')},
      {id:'d2', name:'SNR Surface', desc:'Rain × Distance',
       svg: makeSvg(grad('s2',C.cyan,C.red)+'<polygon fill="url(#s2)" points="15,85 60,72 105,55 150,35 185,18 170,40 125,55 85,65 40,75"/>')},
      {id:'d3', name:'Weather Cloud', desc:'3D scatter of samples',
       svg: makeSvg(dots(Array.from({length:60},()=>[Math.random()*180+10, Math.random()*90+10]), C.cyan2, 1.5))},
      {id:'d4', name:'SE Surface', desc:'SNR × BF elements',
       svg: makeSvg(grad('s4',C.purple,C.green)+'<polygon fill="url(#s4)" points="15,100 60,70 105,45 150,28 185,18 165,38 120,55 75,72 30,88"/>')},
      {id:'d6', name:'BER Surface', desc:'SNR × Rain (log)',
       svg: makeSvg(grad('s6',C.red,C.amber)+'<polygon fill="url(#s6)" points="15,28 60,40 105,58 150,72 185,82 165,75 120,62 75,48 30,38"/>')},
      {id:'d7', name:'Model Comparison 3D', desc:'RMSE × R² × Latency',
       svg: makeSvg(dots([[40,85],[55,60],[70,70],[90,75],[120,55],[160,40],[100,30],[150,90]], C.cyan2, 4))}]},
    {title:'📶 Beamforming & Link KPIs', items:[
      {id:'cSEsnr', name:'SE vs SNR', desc:'Shannon / Hybrid / Analog',
       svg: makeSvg(polyline('5,100 30,92 55,78 80,60 105,42 130,28 155,20 195,15', C.white, 2)+
         polyline('5,100 30,93 55,81 80,65 105,48 130,34 155,26 195,22', C.green, 2)+
         polyline('5,100 30,96 55,88 80,76 105,62 130,50 155,42 195,38', C.red, 2, 'dash'))},
      {id:'cBFse', name:'BF SE vs Att.', desc:'Digital / Hybrid / Analog',
       svg: makeSvg(polyline('5,18 30,22 55,28 80,35 105,42 130,52 155,62 195,75', C.blue, 2)+
         polyline('5,22 30,26 55,32 80,40 105,48 130,58 155,68 195,80', C.green, 2)+
         polyline('5,60 30,65 55,70 80,74 105,78 130,82 155,86 195,90', C.red, 2))},
      {id:'cBFsnr', name:'BF SNR vs Att.', desc:'Degradation per mode',
       svg: makeSvg(polyline('5,15 30,22 55,32 80,42 105,52 130,62 155,72 195,82', C.blue, 2)+
         polyline('5,18 30,25 55,35 80,45 105,55 130,65 155,75 195,85', C.green, 2)+
         polyline('5,28 30,35 55,45 80,55 105,65 130,72 155,78 195,86', C.red, 2))},
      {id:'cTput', name:'Throughput vs Distance', desc:'5 rain levels',
       svg: makeSvg(polyline('5,30 30,42 55,55 80,65 105,72 130,78 155,82 195,88', C.green, 2)+
         polyline('5,35 30,48 55,60 80,68 105,75 130,80 155,85 195,90', C.cyan2, 1.5)+
         polyline('5,45 30,58 55,70 80,78 105,84 130,88 155,92 195,95', C.amber, 1.5)+
         polyline('5,55 30,68 55,80 80,86 105,90 130,93 155,96 195,98', C.red, 1.5))},
      {id:'cMB', name:'Multi-Band Attenuation', desc:'3.5 – 77 GHz',
       svg: makeSvg(polyline('5,98 30,98 55,97 80,97 105,97 130,96 155,96 195,96', C.green, 1.5)+
         polyline('5,80 30,78 55,75 80,72 105,70 130,68 155,66 195,64', C.cyan2, 1.5)+
         polyline('5,72 30,68 55,62 80,57 105,52 130,48 155,44 195,40', C.purple, 1.5)+
         polyline('5,55 30,50 55,42 80,35 105,28 130,22 155,18 195,15', C.amber, 1.5)+
         polyline('5,25 30,22 55,18 80,16 105,14 130,12 155,10 195,8', C.red, 1.8))},
      {id:'cCDF', name:'SNR CDF', desc:'With vs Without BF',
       svg: makeSvg(polyline('5,98 35,92 60,82 80,65 100,42 120,22 140,12 195,8', C.red, 2)+
         polyline('5,95 50,90 80,75 110,55 130,30 150,18 195,10', C.green, 2, 'dash'))}]},
    {title:'🇱🇧 Coverage & System', items:[
      {id:'lebMap', name:'Lebanon 5G Map', desc:'10 cities · coverage projection',
       svg: makeSvg(`<polygon points="50,90 70,80 95,55 110,30 100,20 80,30 60,55 45,80" fill="rgba(25,213,255,.08)" stroke="${C.cyan2}" stroke-width="1.2"/>`+
         dots([[68,42],[78,30],[72,55],[75,68],[85,68],[88,42],[63,68],[60,80],[55,90],[58,85]], C.amber, 3.5))},
      {id:'cSE', name:'BF Spectral Efficiency', desc:'No-BF vs Hybrid',
       svg: makeSvg(`<rect x="40" y="60" width="50" height="40" fill="${C.mut}"/>`+
         `<rect x="110" y="36" width="50" height="64" fill="${C.green}"/>`)},
      {id:'cSNR', name:'BF Received SNR', desc:'No-BF vs Hybrid',
       svg: makeSvg(`<rect x="40" y="55" width="50" height="45" fill="${C.mut}"/>`+
         `<rect x="110" y="20" width="50" height="80" fill="${C.cyan2}"/>`)}]}
  ];

  let total = 0;
  document.getElementById('galleryRoot').innerHTML = cats.map(cat => {
    total += cat.items.length;
    return `<div class="galcat">${cat.title}</div>
      <div class="galgrid">${cat.items.map(it => `
        <button class="galtile" data-target="${it.id}">${it.svg}
          <div class="galmeta"><b>${it.name}</b><span>${it.desc}</span></div>
        </button>`).join('')}</div>`;
  }).join('');
  document.getElementById('galcount').textContent = total + ' figures';
  document.querySelectorAll('.galtile').forEach(t => {
    t.addEventListener('click', () => {
      const id = t.dataset.target;
      const el = document.getElementById(id);
      if (!el) return;
      el.scrollIntoView({behavior:'smooth', block:'center'});
      el.style.transition = 'box-shadow .4s';
      el.style.boxShadow = '0 0 0 3px var(--cyan2), 0 0 40px rgba(25,213,255,.4)';
      setTimeout(() => el.style.boxShadow = '', 1400);
    });
  });
})();

/* ============================================================================
   SECTION 07 — Lebanon coverage map (rebuilt from project's ITU-R physics)
   ============================================================================ */
(function(){
  const L = DATA.lebanon;
  const qcol = q => ({Excellent:C.green, Good:C.cyan2, Fair:C.amber, Poor:C.red}[q] || C.cyan2);

  // Approximate Lebanon border outline (lon,lat)
  const border = [[35.10,33.09],[35.45,33.06],[35.55,33.25],[35.62,33.45],[35.78,33.65],
    [36.06,33.83],[36.30,34.20],[36.42,34.62],[36.32,34.70],[36.08,34.65],[35.98,34.62],
    [35.88,34.63],[35.62,34.45],[35.45,34.30],[35.38,34.08],[35.22,33.78],[35.10,33.45],[35.10,33.09]];

  const lay = baseLay();
  lay.xaxis = {visible:false, range:[34.9, 36.6]};
  lay.yaxis = {visible:false, range:[33.0, 34.8], scaleanchor:'x', scaleratio:1.18};
  lay.margin = {l:0, r:0, t:0, b:0};
  lay.showlegend = false;

  Plotly.newPlot('lebMap', [
    {type:'scatter', mode:'lines', x:border.map(p=>p[0]), y:border.map(p=>p[1]),
     fill:'toself', fillcolor:'rgba(25,213,255,.05)',
     line:{color:C.cyan2, width:1.5}, hoverinfo:'skip', showlegend:false},
    {type:'scatter', mode:'markers+text',
     x:L.map(c=>c.lon), y:L.map(c=>c.lat),
     text:L.map(c=>c.city), textposition:'top center',
     textfont:{size:11, color:'#dce8ff'},
     marker:{size:L.map(c=>14 + c.pop/200000), color:L.map(c=>qcol(c.q)),
             line:{color:'#fff', width:1.5}, opacity:0.95},
     customdata:L.map((_,i)=>i),
     hovertemplate:'<b>%{text}</b><br>Att @28: '+L.map(c=>c.att28).join('|')+' dB<extra></extra>',
     showlegend:false}
  ], lay, CFG);

  function showCity(c){
    const qb = {Excellent:C.green, Good:C.cyan2, Fair:C.amber, Poor:C.red}[c.q];
    document.getElementById('cityBox').innerHTML = `
      <h3>📍 ${c.city}</h3>
      <div class="citymeta">${c.region} · ${fmt(c.pop)} residents</div>
      <div class="cstats">
        <div class="cstat"><div class="v">${c.att28}</div><div class="l">Att @28 GHz (dB)</div></div>
        <div class="cstat"><div class="v">${c.att60}</div><div class="l">Att @60 GHz (dB)</div></div>
        <div class="cstat"><div class="v">${c.snr}</div><div class="l">SNR (dB)</div></div>
        <div class="cstat"><div class="v">${fmt(c.tput)}</div><div class="l">Throughput (Mbps)</div></div>
        <div class="cstat"><div class="v">${c.avail}%</div><div class="l">Availability</div></div>
        <div class="cstat"><div class="v">${c.rain}</div><div class="l">R₀.₀₁ (mm/hr)</div></div>
      </div>
      <div class="qbadge" style="color:${qb};border-color:${qb}55;background:${qb}15">
        ${c.q} signal quality @ 28 GHz</div>`;
  }
  showCity(L.find(c => c.city === 'Beirut') || L[0]);
  document.getElementById('lebMap').on('plotly_click', e => {
    if (e.points && e.points[0] && e.points[0].customdata !== undefined)
      showCity(L[e.points[0].customdata]);
  });
})();

/* ============================================================================
   Nav active-link tracking, reveal-on-scroll, resize
   ============================================================================ */
(function(){
  const io = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && e.target.classList.add('in')), {threshold:.08});
  document.querySelectorAll('section,.kpi,.card,.leadcard').forEach(el => {el.classList.add('reveal'); io.observe(el);});

  const sections = ['home','leaderboard','compare','attenuation','link','viz3d','gallery','map','about','contact'];
  const links = Array.from(document.querySelectorAll('#nlinks a'));
  function highlight(){
    let cur = sections[0];
    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top <= 100) cur = id;
    });
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#'+cur));
  }
  window.addEventListener('scroll', highlight, {passive:true}); highlight();

  const ids = ['mainChart','cAcc','cR2','cBER','cLat','cSE','cSNR','cPred','cSorted','cResid','cHist','cFI','cCond',
               'cBERsnr','cSEsnr','cBFse','cBFsnr','cTput','cMB','cCDF',
               'd1','d2','d3','d4','d6','d7','lebMap'];
  window.addEventListener('resize', () => ids.forEach(id => {
    const el = document.getElementById(id); if (el && el.layout) Plotly.Plots.resize(el);
  }));
})();

})();
