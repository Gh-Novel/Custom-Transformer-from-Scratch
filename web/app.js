/* ─────────────────────────────────────────────────────────────────────────
 * Mini-GPT Live front-end
 *   - Connects to /api/generate (SSE)
 *   - Streams tokens to the output panel
 *   - Updates a growing attention heatmap (D3) and a top-k probability chart
 * ───────────────────────────────────────────────────────────────────────── */

// ── DOM ──────────────────────────────────────────────────────────────────
const $promptInput   = document.getElementById('prompt');
const $maxNew        = document.getElementById('maxNew');
const $temp          = document.getElementById('temp');
const $topK          = document.getElementById('topK');
const $delay         = document.getElementById('delay');

const $maxNewVal     = document.getElementById('maxNewVal');
const $tempVal       = document.getElementById('tempVal');
const $topKVal       = document.getElementById('topKVal');
const $delayVal      = document.getElementById('delayVal');

const $generateBtn   = document.getElementById('generateBtn');
const $stopBtn       = document.getElementById('stopBtn');
const $output        = document.getElementById('output');
const $modelInfo     = document.getElementById('modelInfo');
const $genStatus     = document.getElementById('genStatus');

const $heatmap       = document.getElementById('heatmap');
const $probs         = document.getElementById('probs');

// ── State ────────────────────────────────────────────────────────────────
let eventSource    = null;
let attentionRows  = [];      // each row = a flat array of attention weights
let ctxTokensLast  = [];      // last seen context-window tokens (for x-axis labels)
let promptLength   = 0;
let lastChar       = null;

// ── Slider live values ───────────────────────────────────────────────────
function bindSlider(slider, label, fmt = v => v) {
  const update = () => label.textContent = fmt(slider.value);
  slider.addEventListener('input', update);
  update();
}
bindSlider($maxNew, $maxNewVal);
bindSlider($temp,   $tempVal,   v => Number(v).toFixed(2));
bindSlider($topK,   $topKVal,   v => v == 0 ? 'off' : v);
bindSlider($delay,  $delayVal);

// ── Model info ────────────────────────────────────────────────────────────
fetch('/api/info')
  .then(r => r.json())
  .then(info => {
    $modelInfo.textContent =
      `${(info.params/1e6).toFixed(2)}M params · ` +
      `${info.num_layers}L × ${info.num_heads}H × ${info.embed_dim}d\n` +
      `vocab=${info.vocab_size} · ctx=${info.block_size} · ${info.device}\n` +
      `step=${info.step} · val_loss=${info.val_loss.toFixed(3)}`;
  })
  .catch(() => $modelInfo.textContent = 'model info unavailable');

// ── Buttons ──────────────────────────────────────────────────────────────
$generateBtn.addEventListener('click', startGeneration);
$stopBtn.addEventListener('click', stopGeneration);
$promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') startGeneration();
});

// ── Streaming logic ──────────────────────────────────────────────────────
function startGeneration() {
  stopGeneration();   // clean any prior stream
  resetUI();

  const params = new URLSearchParams({
    prompt:  $promptInput.value,
    max_new: $maxNew.value,
    temp:    $temp.value,
    top_k:   $topK.value,
    delay:   ($delay.value / 1000).toFixed(3),
  });

  eventSource = new EventSource(`/api/generate?${params.toString()}`);
  setStatus('running', 'streaming…');

  eventSource.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'prompt') handlePrompt(data);
    else if (data.type === 'token') handleToken(data);
    else if (data.type === 'done')  handleDone(data);
  };

  eventSource.onerror = () => {
    setStatus('error', 'connection closed');
    stopGeneration();
  };
}

function stopGeneration() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  $generateBtn.disabled = false;
  $stopBtn.disabled = true;
}

function setStatus(cls, text) {
  $genStatus.className = 'hint ' + cls;
  $genStatus.textContent = text;
  $generateBtn.disabled = (cls === 'running');
  $stopBtn.disabled     = (cls !== 'running');
}

function resetUI() {
  attentionRows = [];
  promptLength = 0;
  lastChar = null;
  $output.innerHTML = '<span class="cursor">▍</span>';
  drawHeatmap([], []);
  drawProbs([], null);
}

// ── Event handlers ───────────────────────────────────────────────────────
function handlePrompt(data) {
  promptLength = data.text.length;
  ctxTokensLast = data.ctx_tokens;
  $output.innerHTML =
    `<span class="prompt-text">${escapeHtml(data.text)}</span><span class="cursor">▍</span>`;
}

function handleToken(data) {
  // 1. Append the new char to the output (with a brief flash animation)
  appendChar(data.char);

  // 2. Update probability chart
  drawProbs(data.topk, data.char);

  // 3. Update attention heatmap
  ctxTokensLast = data.ctx_tokens;
  attentionRows.push(data.attention);
  drawHeatmap(attentionRows, ctxTokensLast);
}

function handleDone(data) {
  setStatus('', `done · ${data.total} tokens`);
  stopGeneration();
}

// ── Output helpers ───────────────────────────────────────────────────────
function appendChar(ch) {
  // Replace the cursor with new char + cursor
  const cursor = $output.querySelector('.cursor');
  if (cursor) cursor.remove();

  const span = document.createElement('span');
  span.className = 'new-char flash';
  span.textContent = ch;
  $output.appendChild(span);

  const newCursor = document.createElement('span');
  newCursor.className = 'cursor';
  newCursor.textContent = '▍';
  $output.appendChild(newCursor);

  $output.scrollTop = $output.scrollHeight;

  // Remove flash class after animation
  setTimeout(() => span.classList.remove('flash'), 320);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

// ── D3: Probability bar chart ────────────────────────────────────────────
function drawProbs(topk, chosen) {
  $probs.innerHTML = '';
  if (!topk.length) return;

  const maxProb = topk[0][1];
  topk.forEach(([ch, p]) => {
    const row    = document.createElement('div');
    row.className = 'bar-row' + (ch === chosen ? ' chosen' : '');

    const charEl = document.createElement('div');
    charEl.className = 'bar-char';
    charEl.textContent = displayChar(ch);

    const bgEl   = document.createElement('div');
    bgEl.className = 'bar-bg';

    const fillEl = document.createElement('div');
    fillEl.className = 'bar-fill';
    // Animate from 0 width
    requestAnimationFrame(() => {
      fillEl.style.width = `${(p / maxProb * 100).toFixed(1)}%`;
    });

    const pctEl  = document.createElement('div');
    pctEl.className = 'bar-pct';
    pctEl.textContent = `${(p * 100).toFixed(1)}%`;

    bgEl.appendChild(fillEl);
    row.appendChild(charEl);
    row.appendChild(bgEl);
    row.appendChild(pctEl);
    $probs.appendChild(row);
  });
}

function displayChar(ch) {
  if (ch === '\n') return '↵';
  if (ch === ' ')  return '␣';
  if (ch === '\t') return '⇥';
  return ch;
}

// ── D3: Attention heatmap ────────────────────────────────────────────────
function drawHeatmap(rows, ctxTokens) {
  d3.select($heatmap).selectAll('*').remove();
  if (!rows.length) {
    d3.select($heatmap)
      .append('div')
      .style('color', 'var(--text-dim)')
      .style('font-size', '12px')
      .style('padding', '20px')
      .text('Attention pattern will render here as tokens stream in.');
    return;
  }

  const containerW = $heatmap.clientWidth;
  const containerH = $heatmap.clientHeight;
  const margin     = { top: 18, right: 12, bottom: 28, left: 12 };
  const W          = Math.max(200, containerW  - margin.left - margin.right);
  const H          = Math.max(140, containerH  - margin.top  - margin.bottom);

  const numRows    = rows.length;
  const numCols    = Math.max(...rows.map(r => r.length));

  const svg = d3.select($heatmap)
    .append('svg')
    .attr('viewBox', `0 0 ${containerW} ${containerH}`)
    .attr('preserveAspectRatio', 'none');

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`);

  const cellW = W / numCols;
  const cellH = H / Math.max(numRows, 8);

  // Find global max for normalization
  let globalMax = 0;
  for (const r of rows) for (const v of r) if (v > globalMax) globalMax = v;
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, globalMax || 1]);

  // Draw cells
  rows.forEach((row, ri) => {
    g.selectAll(`.cell-${ri}`)
      .data(row)
      .enter()
      .append('rect')
      .attr('x', (_, ci) => ci * cellW)
      .attr('y', ri * cellH)
      .attr('width',  Math.max(1, cellW))
      .attr('height', Math.max(1, cellH))
      .attr('fill', d => colorScale(d))
      .on('mousemove', function (ev, d) {
        showTooltip(ev, d, ri);
      })
      .on('mouseleave', hideTooltip);
  });

  // Bottom axis: token labels (only every Nth char for readability)
  const labelStep = Math.max(1, Math.ceil(numCols / 50));
  for (let i = 0; i < ctxTokens.length; i += labelStep) {
    g.append('text')
      .attr('x', i * cellW + cellW / 2)
      .attr('y', numRows * cellH + 14)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-dim)')
      .attr('font-family', 'var(--mono)')
      .attr('font-size', '9px')
      .text(displayChar(ctxTokens[i]));
  }

  // Title (small)
  g.append('text')
    .attr('x', 0)
    .attr('y', -6)
    .attr('fill', 'var(--text-dim)')
    .attr('font-size', '10px')
    .text(`${numRows} step${numRows!==1?'s':''}  ·  ${numCols} keys  ·  max=${globalMax.toFixed(3)}`);
}

// ── Tooltip ──────────────────────────────────────────────────────────────
let tooltip = null;
function showTooltip(ev, value, rowIdx) {
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'heatmap-tooltip';
    document.body.appendChild(tooltip);
  }
  tooltip.textContent = `step ${rowIdx}\nweight ${value.toFixed(4)}`;
  tooltip.style.left   = (ev.pageX + 12) + 'px';
  tooltip.style.top    = (ev.pageY + 12) + 'px';
  tooltip.style.opacity = '1';
}
function hideTooltip() { if (tooltip) tooltip.style.opacity = '0'; }

// Initial empty render
drawHeatmap([], []);
drawProbs([], null);

// ── Help modal ───────────────────────────────────────────────────────────
const $helpBtn      = document.getElementById('helpBtn');
const $helpOverlay  = document.getElementById('helpOverlay');
const $helpCloseBtn = document.getElementById('helpCloseBtn');

function openHelp()  { $helpOverlay.hidden = false; document.body.style.overflow = 'hidden'; }
function closeHelp() { $helpOverlay.hidden = true;  document.body.style.overflow = ''; }

$helpBtn.addEventListener('click', openHelp);
$helpCloseBtn.addEventListener('click', closeHelp);

// Click outside the modal box closes it
$helpOverlay.addEventListener('click', (e) => {
  if (e.target === $helpOverlay) closeHelp();
});

// Esc closes it
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$helpOverlay.hidden) closeHelp();
});

// Auto-open on first visit (uses localStorage)
if (!localStorage.getItem('miniGptSeenHelp')) {
  setTimeout(openHelp, 400);
  localStorage.setItem('miniGptSeenHelp', '1');
}
