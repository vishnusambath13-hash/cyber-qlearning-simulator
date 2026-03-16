// ============================================================
// APP.JS
// Frontend application logic for the Cyber Battle Q-Learning HUD.
//
// Responsibilities:
//   — Fetch simulation state from FastAPI backend on load
//   — POST /simulation/step on each round
//   — POST /simulation/reset on reset
//   — Render all HUD components from API responses
//   — Run auto-mode interval loop
//   — Trigger battlefield animations
//   — Wire all button/slider event listeners
//
// Depends on:
//   — styles.css  (all class names used here)
//   — Chart.js    (loaded in index.html before this file)
//   — FastAPI backend running on API_BASE
// ============================================================

'use strict';

// ----------------------------------------------------------
// CONFIGURATION
// ----------------------------------------------------------

const API_BASE    = 'http://localhost:8000';
const ENDPOINTS   = {
  state:   `${API_BASE}/simulation/state`,
  step:    `${API_BASE}/simulation/step`,
  reset:   `${API_BASE}/simulation/reset`,
  actions: `${API_BASE}/simulation/actions`,
};

const TIMELINE_MAX  = 24;
const LOG_MAX       = 60;
const TREND_MAX     = 40;

// ----------------------------------------------------------
// APPLICATION STATE
// Pure frontend state — not simulation logic.
// ----------------------------------------------------------

const AppState = {
  isAutoRunning:  false,
  autoIntervalId: null,
  autoSpeed:      1000,       // ms per round in auto mode
  trendChartObj:  null,       // Chart.js instance
  actions:        null,       // Cached from GET /simulation/actions
  lastResult:     null,       // Most recent RoundResult from backend
};

// ----------------------------------------------------------
// ELEMENT CACHE
// All getElementById calls happen once here.
// ----------------------------------------------------------

const El = {
  // Top bar
  statusDot:      document.getElementById('statusDot'),
  roundNum:       document.getElementById('roundNum'),
  healthReadout:  document.getElementById('healthReadout'),
  modeLabel:      document.getElementById('modeLabel'),
  modeDot:        document.getElementById('modeDot'),
  sessionId:      document.getElementById('sessionId'),
  epsilonSlider: document.getElementById('epsilonSlider'),

  // Health bar
  healthFill:     document.getElementById('healthFill'),
  healthText:     document.getElementById('healthText'),

  // Attacker
  atkActionDisplay: document.getElementById('atkActionDisplay'),
  atkActionVal:     document.getElementById('atkActionVal'),
  atkBreaches:      document.getElementById('atkBreaches'),
  atkRate:          document.getElementById('atkRate'),
  atkProbBars:      document.getElementById('atkProbBars'),
  atkQtable:        document.getElementById('atkQtable'),
  atkChart:         document.getElementById('atkChart'),

  // Defender
  defActionDisplay: document.getElementById('defActionDisplay'),
  defActionVal:     document.getElementById('defActionVal'),
  defBlocks:        document.getElementById('defBlocks'),
  defRate:          document.getElementById('defRate'),
  defProbBars:      document.getElementById('defProbBars'),
  defQtable:        document.getElementById('defQtable'),
  defChart:         document.getElementById('defChart'),

  // Engagement
  lastAtkName:    document.getElementById('lastAtkName'),
  lastDefName:    document.getElementById('lastDefName'),
  outcomeChip:    document.getElementById('outcomeChip'),
  atkReward:      document.getElementById('atkReward'),
  defReward:      document.getElementById('defReward'),

  // Timeline + log
  timeline:       document.getElementById('timeline'),
  combatLog:      document.getElementById('combatLog'),

  // Epsilon
  epsilonValue:   document.getElementById('epsilonValue'),
  epsilonFill:    document.getElementById('epsilonFill'),

  // Trend chart
  trendChart:     document.getElementById('trendChart'),

  // Buttons
  btnRun:         document.getElementById('btnRun'),
  btnAuto:        document.getElementById('btnAuto'),
  btnPause:       document.getElementById('btnPause'),
  btnReset:       document.getElementById('btnReset'),
  speedSlider:    document.getElementById('speedSlider'),
  speedLabel:     document.getElementById('speedLabel'),

  // Battlefield
  battlefield:    document.getElementById('battlefield'),
  bfPanel:        document.getElementById('battlefieldPanel'),
  bfAnimLayer:    document.getElementById('bfAnimLayer'),
  bfStatusBadge:  document.getElementById('bfStatusBadge'),
  targetCore:     document.getElementById('targetCore'),
};

// ----------------------------------------------------------
// API HELPERS
// ----------------------------------------------------------

/**
 * Wrapper around fetch() with JSON parsing and error handling.
 * Shows an error banner on network failure.
 * @param {string} url
 * @param {object} options - fetch options
 * @returns {Promise<object|null>}
 */
async function apiFetch(url, options = {}) {
  const defaults = {
    headers: { 'Content-Type': 'application/json' },
  };
  try {
    const res  = await fetch(url, { ...defaults, ...options });
    const data = await res.json();
    if (!res.ok) {
      showErrorBanner(data.error || `HTTP ${res.status}`);
      return null;
    }
    hideErrorBanner();
    return data;
  } catch (err) {
    showErrorBanner(`Cannot reach backend — is uvicorn running on port 8000?`);
    return null;
  }
}

/**
 * POST /simulation/step — run N rounds on the backend.
 * @param {number} steps - number of rounds to run (default 1)
 * @returns {Promise<object|null>} RoundResult or null
 */
async function fetchStep(steps = 1) {

  const epsilon =
    parseFloat(El.epsilonSlider?.value ?? 0.9);

  return apiFetch(ENDPOINTS.step, {
    method: 'POST',
    body: JSON.stringify({
      steps,
      epsilon
    }),
  });
}

/**
 * POST /simulation/reset — reset the simulation.
 * @returns {Promise<object|null>} ResetResponse or null
 */
async function fetchReset() {
  return apiFetch(ENDPOINTS.reset, {
    method: 'POST',
    body:   JSON.stringify({ confirm: true }),
  });
}

/**
 * GET /simulation/state — full state snapshot.
 * @returns {Promise<object|null>} SimulationState or null
 */
async function fetchState() {
  return apiFetch(ENDPOINTS.state);
}

/**
 * GET /simulation/actions — label metadata.
 * @returns {Promise<object|null>} ActionsMetadata or null
 */
async function fetchActions() {
  return apiFetch(ENDPOINTS.actions);
}

// ----------------------------------------------------------
// ERROR BANNER
// ----------------------------------------------------------

let _errorBannerEl = null;

function showErrorBanner(message) {
  if (_errorBannerEl) _errorBannerEl.remove();
  _errorBannerEl = document.createElement('div');
  _errorBannerEl.className = 'api-error-banner';
  _errorBannerEl.textContent = `⚠ ${message}`;
  document.body.appendChild(_errorBannerEl);
}

function hideErrorBanner() {
  if (_errorBannerEl) {
    _errorBannerEl.remove();
    _errorBannerEl = null;
  }
}

// ----------------------------------------------------------
// RENDER HELPERS
// ----------------------------------------------------------

/**
 * Pad round number to 3 digits: 7 → "007"
 */
function padRound(n) {
  return String(n).padStart(3, '0');
}

/**
 * Return health bar fill CSS class based on HP value.
 */
function healthFillClass(hp) {
  if (hp > 60) return 'health-bar__fill--high';
  if (hp > 25) return 'health-bar__fill--mid';
  return 'health-bar__fill--low';
}

/**
 * Return CSS color string for the top-bar health readout.
 */
function healthColor(hp) {
  if (hp > 60) return 'var(--health-high)';
  if (hp > 25) return 'var(--health-mid)';
  return 'var(--health-crit)';
}

/**
 * Remove all CSS classes from an element that start with a prefix.
 */
function clearModifiers(el, prefix) {
  if (!el) return;
  [...el.classList]
    .filter(c => c.startsWith(prefix))
    .forEach(c => el.classList.remove(c));
}

/**
 * Add a class, then remove it after durationMs.
 */
function flashClass(el, cls, durationMs = 400) {
  if (!el) return;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), durationMs);
}

// ----------------------------------------------------------
// RENDER: TOP BAR
// ----------------------------------------------------------

function renderTopBar(data) {
  // Round
  if (El.roundNum) {
    El.roundNum.textContent = padRound(data.total_rounds ?? data.round ?? 0);
  }

  // Health readout
  const hp  = data.system_health;
  const col = healthColor(hp);
  if (El.healthReadout) {
    El.healthReadout.textContent  = hp + '%';
    El.healthReadout.style.color  = col;
    El.healthReadout.style.textShadow = `0 0 15px ${col}`;
  }

  // Session ID
  if (El.sessionId) {
    El.sessionId.textContent = data.session_id || '—';
  }
}

// ----------------------------------------------------------
// RENDER: HEALTH BAR
// ----------------------------------------------------------

function renderHealthBar(hp) {
  const pct = Math.max(0, Math.min(100, hp));
  if (El.healthFill) {
    El.healthFill.style.width = pct + '%';
    clearModifiers(El.healthFill, 'health-bar__fill--');
    El.healthFill.classList.add(healthFillClass(pct));
  }
  if (El.healthText) {
    El.healthText.textContent = pct + '%';
  }
}

// ----------------------------------------------------------
// RENDER: ACTION DISPLAYS
// ----------------------------------------------------------

function renderAtkAction(short) {
  if (El.atkActionVal) El.atkActionVal.textContent = short;
  flashClass(El.atkActionDisplay, 'is-flashing', 400);
}

function renderDefAction(short) {
  if (El.defActionVal) El.defActionVal.textContent = short;
  flashClass(El.defActionDisplay, 'is-flashing', 400);
}

// ----------------------------------------------------------
// RENDER: STATS
// ----------------------------------------------------------

function renderStats(data) {
  if (El.atkBreaches) El.atkBreaches.textContent = data.total_breaches;
  if (El.atkRate)     El.atkRate.textContent     = Math.round(data.breach_rate) + '%';
  if (El.defBlocks)   El.defBlocks.textContent   = data.total_blocks;
  if (El.defRate)     El.defRate.textContent     = Math.round(data.block_rate)  + '%';
}

// ----------------------------------------------------------
// RENDER: PROBABILITY BARS
// ----------------------------------------------------------

function renderProbBars(container, probs, labels, axis) {
  if (!container || !probs) return;
  const fillCls  = `prob-row__fill--${axis}`;
  const valueCls = `prob-row__value--${axis}`;
  container.innerHTML = probs.map((p, i) => {
    const pct = Math.round(p * 100);
    return `
      <div class="prob-row">
        <span class="prob-row__name">${labels[i]}</span>
        <div class="prob-row__track">
          <div class="prob-row__fill ${fillCls}" style="width:${pct}%"></div>
        </div>
        <span class="prob-row__value ${valueCls}">${pct}%</span>
      </div>`;
  }).join('');
}

// ----------------------------------------------------------
// RENDER: Q-TABLE HEATMAP
// ----------------------------------------------------------

/**
 * Map a normalized 0–1 Q-value to an attacker cell color (red axis).
 */
function atkCellColor(norm) {
  const r = Math.round(40  + norm * 215);
  const g = Math.round(norm * 30);
  const b = Math.round(norm * 20);
  const a = (0.25 + norm * 0.75).toFixed(2);
  return `rgba(${r},${g},${b},${a})`;
}

/**
 * Map a normalized 0–1 Q-value to a defender cell color (green axis).
 */
function defCellColor(norm) {
  const r = Math.round(norm * 20);
  const g = Math.round(60  + norm * 195);
  const b = Math.round(50  + norm * 80);
  const a = (0.25 + norm * 0.75).toFixed(2);
  return `rgba(${r},${g},${b},${a})`;
}

function renderQtable(container, qtable, labels, axis) {
  if (!container || !qtable) return;

  const flat    = qtable.flat();
  const minV    = Math.min(...flat);
  const maxV    = Math.max(...flat);
  const range   = maxV - minV || 1;
  const colorFn = axis === 'atk' ? atkCellColor : defCellColor;

  // Column header row
  const header = `
    <div class="qtable__header">
      ${labels.map(l =>
        `<span class="qtable__col-label">${l.substring(0,3)}</span>`
      ).join('')}
    </div>`;

  // Data rows
  const rows = qtable.map((row, s) => {
    const cells = row.map(val => {
      const norm  = (val - minV) / range;
      const color = colorFn(norm);
      return `<div class="qtable__cell"
                   style="background:${color}"
                   title="Q=${val.toFixed(3)}"></div>`;
    }).join('');
    return `
      <div class="qtable__row">
        <span class="qtable__label">${labels[s]}</span>
        ${cells}
      </div>`;
  }).join('');

  container.innerHTML = header + rows;
}

// ----------------------------------------------------------
// RENDER: BAR CHARTS (action frequency)
// ----------------------------------------------------------

function renderBarChart(container, counts, labels, axis) {
  if (!container || !counts) return;

  const max = Math.max(...counts, 1);
  const barCls = `bar-chart__bar--${axis}`;

  container.innerHTML = counts.map((c, i) => {

    const pct = Math.round((c / max) * 100);
    const lbl = labels[i].substring(0, 4);

    return `
      <div class="bar-chart__item">
        <span class="bar-chart__bar-label">${lbl}</span>
        <div class="bar-chart__bar ${barCls}"
             style="width:${pct}%"
             title="${labels[i]}: ${c}">
        </div>
      </div>`;
  }).join('');
}

// ----------------------------------------------------------
// RENDER: ENGAGEMENT PANEL
// ----------------------------------------------------------

function renderEngagement(result) {
  // Action names
  if (El.lastAtkName) El.lastAtkName.textContent = result.atk_action_name;
  if (El.lastDefName) El.lastDefName.textContent = result.def_action_name;

  // Outcome chip
  if (El.outcomeChip) {
    clearModifiers(El.outcomeChip, 'outcome-chip--');
    if (result.blocked) {
      El.outcomeChip.classList.add('outcome-chip--block');
      El.outcomeChip.textContent = '⬡ BLOCKED';
    } else {
      El.outcomeChip.classList.add('outcome-chip--breach');
      El.outcomeChip.textContent = '⚠ BREACH';
    }
  }

  // Reward badges
  if (El.atkReward) {
    clearModifiers(El.atkReward, 'reward-badge--');
    El.atkReward.classList.add(
      result.atk_reward > 0 ? 'reward-badge--pos' : 'reward-badge--neg'
    );
    El.atkReward.textContent =
      (result.atk_reward > 0 ? '+' : '') + result.atk_reward;
  }

  if (El.defReward) {
    clearModifiers(El.defReward, 'reward-badge--');
    El.defReward.classList.add(
      result.def_reward > 0 ? 'reward-badge--pos' : 'reward-badge--neg'
    );
    El.defReward.textContent =
      (result.def_reward > 0 ? '+' : '') + result.def_reward;
  }
}

// ----------------------------------------------------------
// RENDER: TIMELINE
// ----------------------------------------------------------

function appendTimelineChip(result) {
  if (!El.timeline) return;

  // Remove current highlight from previous chip
  const prev = El.timeline.querySelector('.timeline__chip--current');
  if (prev) prev.classList.remove('timeline__chip--current');

  const chip      = document.createElement('div');
  const stateCls  = result.blocked
    ? 'timeline__chip--block'
    : 'timeline__chip--breach';

  chip.className = `timeline__chip ${stateCls} timeline__chip--current`;
  chip.title     = `R${result.round}: ${result.atk_action_short} vs `
                 + `${result.def_action_short} → `
                 + `${result.blocked ? 'BLOCK' : 'BREACH'}`;

  chip.innerHTML = `
    <span class="timeline__chip-atk">
      ${result.atk_action_short.substring(0, 3)}
    </span>
    <span class="timeline__chip-def">
      ${result.def_action_short.substring(0, 3)}
    </span>`;

  El.timeline.appendChild(chip);

  // Cap visible chips
  while (El.timeline.children.length > TIMELINE_MAX) {
    El.timeline.removeChild(El.timeline.firstChild);
  }

  El.timeline.scrollLeft = El.timeline.scrollWidth;
}

function clearTimeline() {
  if (El.timeline) El.timeline.innerHTML = '';
}

// ----------------------------------------------------------
// RENDER: COMBAT LOG
// ----------------------------------------------------------

function appendLogEntry(result) {
  if (!El.combatLog) return;

  const entry    = document.createElement('div');
  const isBreach = !result.blocked;
  entry.className = 'log-entry' + (isBreach ? ' log-entry--breach' : '');

  const resCls  = isBreach ? 'log-entry__result--breach' : 'log-entry__result--block';
  const resTxt  = isBreach ? 'BREACH' : 'BLOCKED';

  entry.innerHTML = `
    <span class="log-entry__round">R${padRound(result.round)}</span>
    <span class="log-entry__atk">${result.atk_action_short}</span>
    <span class="log-entry__sep">|</span>
    <span class="log-entry__def">${result.def_action_short}</span>
    <span class="log-entry__sep">|</span>
    <span class="${resCls}">${resTxt}</span>
    <span class="log-entry__health">HP:${result.system_health}</span>`;

  El.combatLog.insertBefore(entry, El.combatLog.firstChild);

  while (El.combatLog.children.length > LOG_MAX) {
    El.combatLog.removeChild(El.combatLog.lastChild);
  }
}

function clearLog() {
  if (El.combatLog) El.combatLog.innerHTML = '';
}

// ----------------------------------------------------------
// RENDER: EPSILON BAR
// ----------------------------------------------------------

function renderEpsilon(epsilon) {

  const pct = Math.round(epsilon * 100);

  if (El.epsilonValue)
    El.epsilonValue.textContent = pct + '%';

  if (El.epsilonSlider)
    El.epsilonSlider.value = epsilon;

}
// ----------------------------------------------------------
// RENDER: TREND CHART (Chart.js)
// ----------------------------------------------------------

function renderTrendChart(trendData) {
  if (!El.trendChart || !window.Chart) return;
  if (trendData.length < 2) return;

  if (AppState.trendChartObj) {
    AppState.trendChartObj.destroy();
    AppState.trendChartObj = null;
  }

  AppState.trendChartObj = new Chart(El.trendChart, {
    type: 'line',
    data: {
      labels:   trendData.map(() => ''),
      datasets: [
        {
          data:            trendData,
          borderColor:     '#ff2d2d',
          borderWidth:     1.5,
          pointRadius:     0,
          fill:            true,
          backgroundColor: 'rgba(255,45,45,0.10)',
          tension:         0.4,
        },
        {
          data:            trendData.map(v => 100 - v),
          borderColor:     '#00ff9d',
          borderWidth:     1.5,
          pointRadius:     0,
          fill:            false,
          tension:         0.4,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           false,
      plugins: {
        legend:  { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: { display: false },
        y: { display: false, min: 0, max: 100 },
      },
    },
  });
}

// ----------------------------------------------------------
// RENDER: STATUS INDICATORS
// ----------------------------------------------------------

function setStatusLive(live) {
  const cls = live ? 'status-indicator--live' : 'status-indicator--standby';
  [El.statusDot, El.modeDot].forEach(dot => {
    if (!dot) return;
    clearModifiers(dot, 'status-indicator--');
    dot.classList.add(cls);
  });
}

function setModeLabel(text) {
  if (El.modeLabel) El.modeLabel.textContent = text;
}

// ----------------------------------------------------------
// RENDER: BUTTON STATES
// ----------------------------------------------------------

function renderButtonStates({ isRunning, isGameOver }) {
  if (El.btnAuto) {
    El.btnAuto.classList.toggle('is-active', isRunning);
  }

  // Disable run + auto when game over
  [El.btnRun, El.btnAuto].forEach(btn => {
    if (!btn) return;
    btn.disabled = isGameOver;
    btn.classList.toggle('disabled', isGameOver);
  });

  // Pause only active when auto is running
  if (El.btnPause) {
    El.btnPause.disabled = !isRunning;
    El.btnPause.classList.toggle('disabled', !isRunning);
  }
}

// ----------------------------------------------------------
// RENDER: BATTLEFIELD STATE
// ----------------------------------------------------------

function setBattlefieldScanning(scanning) {
  if (El.battlefield) {
    El.battlefield.classList.toggle('is-scanning', scanning);
  }
}

function setBattlefieldStatus(text, variant) {
  if (!El.bfStatusBadge) return;
  clearModifiers(El.bfStatusBadge, 'bf-status-badge--');
  El.bfStatusBadge.classList.add(`bf-status-badge--${variant}`);
  El.bfStatusBadge.textContent = text;
}

function setBattlefieldCritical(critical) {
  if (El.battlefield) {
    El.battlefield.classList.toggle('is-critical', critical);
  }
}

function setBattlefieldGameOver() {
  if (El.battlefield) El.battlefield.classList.add('is-game-over');
  if (El.bfPanel)     flashClass(El.bfPanel, 'is-damaged', 700);
}

function flashTargetCore(breach) {
  if (!El.targetCore) return;
  const cls = breach ? 'is-breached' : 'is-defended';
  flashClass(El.targetCore, cls, 800);
}

function flashBattlefieldPanel(breach) {
  if (!El.bfPanel) return;
  flashClass(El.bfPanel, breach ? 'is-damaged' : 'is-defended', 700);
}

// ----------------------------------------------------------
// BATTLEFIELD ANIMATIONS
// JS-spawned elements appended to .bf-anim-layer
// then removed after their CSS animation ends.
// ----------------------------------------------------------

/**
 * Spawn N expanding scan rings (attack = red, defend = green).
 */
function spawnScanRings(axis, count = 3) {
  if (!El.bfAnimLayer) return;
  for (let i = 0; i < count; i++) {
    const ring = document.createElement('div');
    ring.className = `scan-ring scan-ring--${axis}`
      + (i > 0 ? ` scan-ring--delay-${i}` : '');
    El.bfAnimLayer.appendChild(ring);
    setTimeout(() => ring.remove(), 1400 + i * 200);
  }
}

/**
 * Spawn vertical breach strike lines at random X positions.
 */
function spawnBreachLines(count = 3) {
  if (!El.bfAnimLayer) return;
  const bf = El.battlefield;
  if (!bf) return;
  const w = bf.offsetWidth;

  for (let i = 0; i < count; i++) {
    const line  = document.createElement('div');
    const xPos  = 20 + Math.random() * (w - 40);
    const h     = 30 + Math.random() * 50;
    const yPos  = 5  + Math.random() * 30;

    line.className = 'breach-line';
    line.style.cssText = `
      left:   ${xPos}px;
      top:    ${yPos}%;
      height: ${h}px;
    `;
    El.bfAnimLayer.appendChild(line);
    setTimeout(() => line.remove(), 600);
  }
}

/**
 * Spawn a full-battlefield shield flash overlay.
 */
function spawnShieldFlash(axis) {
  if (!El.bfAnimLayer) return;
  const flash = document.createElement('div');
  flash.className = `shield-flash shield-flash--${axis}`;
  El.bfAnimLayer.appendChild(flash);
  setTimeout(() => flash.remove(), 700);
}

/**
 * Spawn firewall assembly lines (horizontal, staggered).
 */
function spawnFirewallLines(count = 5) {
  if (!El.bfAnimLayer) return;
  const bf = El.battlefield;
  if (!bf) return;
  const h = bf.offsetHeight;

  for (let i = 0; i < count; i++) {
    const line = document.createElement('div');
    line.className = 'firewall-line';
    line.style.cssText = `
      top:               ${(i + 1) * (h / (count + 1))}px;
      animation-delay:   ${i * 0.08}s;
    `;
    El.bfAnimLayer.appendChild(line);
    setTimeout(() => line.remove(), 900 + i * 100);
  }
}

/**
 * Spawn packet flood dots (DDoS attack visualization).
 */
function spawnPacketDots(count = 12) {
  if (!El.bfAnimLayer) return;
  const bf = El.battlefield;
  if (!bf) return;
  const w  = bf.offsetWidth;
  const h  = bf.offsetHeight;
  const cx = w / 2;
  const cy = h / 2;

  for (let i = 0; i < count; i++) {
    setTimeout(() => {
      const dot = document.createElement('div');
      // Spawn from random edge
      const edge = Math.floor(Math.random() * 4);
      let sx, sy;
      switch (edge) {
        case 0: sx = Math.random() * w; sy = 0;  break;
        case 1: sx = w;                 sy = Math.random() * h; break;
        case 2: sx = Math.random() * w; sy = h;  break;
        default: sx = 0;               sy = Math.random() * h;
      }
      const tx = cx - sx + (Math.random() - 0.5) * 30;
      const ty = cy - sy + (Math.random() - 0.5) * 30;

      dot.className = 'packet-dot';
      dot.style.cssText = `
        left:  ${sx}px;
        top:   ${sy}px;
        --tx:  ${tx}px;
        --ty:  ${ty}px;
      `;
      El.bfAnimLayer.appendChild(dot);
      setTimeout(() => dot.remove(), 700);
    }, i * 40);
  }
}

/**
 * Flash a brief round number in the battlefield.
 */
function spawnRoundFlash(round) {
  if (!El.bfAnimLayer) return;
  const flash = document.createElement('div');
  flash.className   = 'round-flash';
  flash.textContent = `ROUND ${padRound(round)}`;
  El.bfAnimLayer.appendChild(flash);
  setTimeout(() => flash.remove(), 1000);
}

/**
 * Route the correct animation set based on attack type and outcome.
 * @param {object} result - RoundResult from backend
 */
function triggerBattlefieldAnimations(result) {
  const atkId   = result.atk_action_id;
  const blocked = result.blocked;

  // Round flash
  spawnRoundFlash(result.round);

  if (blocked) {
    // Defense won
    spawnShieldFlash('def');
    spawnScanRings('def', 2);
    spawnFirewallLines(4);
    setBattlefieldStatus('BLOCKED', 'scanning');
  } else {
    // Attacker breached
    spawnShieldFlash('atk');
    spawnScanRings('atk', 3);

    // Attack-specific effects
    if (atkId === 1) {
      // DDoS → packet flood
      spawnPacketDots(14);
    } else {
      // All other attacks → breach lines
      spawnBreachLines(3);
    }

    setBattlefieldStatus('BREACH', 'breach');
  }

  // Reset status badge after animation
  setTimeout(() => setBattlefieldStatus('IDLE', 'idle'), 1200);
}

// ----------------------------------------------------------
// PORT NODE HIGHLIGHTING
// Highlights the relevant port node for the current attack.
// ----------------------------------------------------------

const PORT_NODES = [
  document.getElementById('port-http'),
  document.getElementById('port-https'),
  document.getElementById('port-ssh'),
  document.getElementById('port-ftp'),
  document.getElementById('port-smtp'),
  document.getElementById('port-dns'),
];

// Map attack IDs to port node indices
const ATK_TO_PORT = { 0:0, 1:1, 2:4, 3:2, 4:5, 5:3, 6:1 };

function highlightPort(atkId, blocked) {
  PORT_NODES.forEach(n => {
    if (!n) return;
    n.classList.remove('is-targeted', 'is-shielded');
  });

  const idx  = ATK_TO_PORT[atkId] ?? 0;
  const node = PORT_NODES[idx];
  if (!node) return;

  node.classList.add(blocked ? 'is-shielded' : 'is-targeted');
  setTimeout(() => {
    node.classList.remove('is-targeted', 'is-shielded');
  }, 1000);
}

// ----------------------------------------------------------
// GAME OVER
// ----------------------------------------------------------

function showGameOver(data) {
  const existing = document.querySelector('.game-over-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.className = 'game-over-overlay';
  overlay.innerHTML = `
    <div class="game-over-box">
      <span class="game-over-box__title">SYSTEM COMPROMISED</span>
      <div class="game-over-box__stats">
        <div>Rounds fought &nbsp;&nbsp;
          <span class="game-over-box__stat-val">${data.total_rounds}</span>
        </div>
        <div>Total breaches &nbsp;
          <span class="game-over-box__stat-val text-atk">${data.total_breaches}</span>
        </div>
        <div>Blocks landed &nbsp;&nbsp;
          <span class="game-over-box__stat-val text-def">${data.total_blocks}</span>
        </div>
        <div>Breach rate &nbsp;&nbsp;&nbsp;&nbsp;
          <span class="game-over-box__stat-val">${Math.round(data.breach_rate)}%</span>
        </div>
      </div>
      <button class="btn btn--reset" id="gameOverResetBtn">
        ↺ RESET SIMULATION
      </button>
    </div>`;

  document.body.appendChild(overlay);

  document.getElementById('gameOverResetBtn')
    .addEventListener('click', () => {
      overlay.remove();
      handleReset();
    });
}

function hideGameOver() {
  const overlay = document.querySelector('.game-over-overlay');
  if (overlay) overlay.remove();
}

// ----------------------------------------------------------
// FULL RENDER PASS
// Called after every step with the RoundResult from backend.
// ----------------------------------------------------------

function renderRound(result) {
  const labels = AppState.actions;

  // Top bar + health
  renderTopBar(result);
  renderHealthBar(result.system_health);

  // Actions
  renderAtkAction(result.atk_action_short);
  renderDefAction(result.def_action_short);

  // Stats
  renderStats(result);

  // Engagement
  renderEngagement(result);

  // Q-tables
  if (labels) {
    const atkShorts = labels.attacks.map(a => a.short);
    const defShorts = labels.defenses.map(d => d.short);

    renderQtable(El.atkQtable, result.atk_qtable.qtable, atkShorts, 'atk');
    renderQtable(El.defQtable, result.def_qtable.qtable, defShorts, 'def');
    renderProbBars(El.atkProbBars, result.atk_qtable.probabilities, atkShorts, 'atk');
    renderProbBars(El.defProbBars, result.def_qtable.probabilities, defShorts, 'def');
    renderBarChart(El.atkChart, result.atk_counts, atkShorts, 'atk');
    renderBarChart(El.defChart, result.def_counts, defShorts, 'def');
  }

  // Epsilon
  renderEpsilon(result.epsilon);

  // Trend chart
  renderTrendChart(result.trend_data);

  // Timeline + log
  appendTimelineChip(result);
  appendLogEntry(result);

  // Battlefield
  flashTargetCore(!result.blocked);
  flashBattlefieldPanel(!result.blocked);
  setBattlefieldCritical(result.system_health <= 25);
  triggerBattlefieldAnimations(result);
  highlightPort(result.atk_action_id, result.blocked);

  // Game over
  if (result.is_game_over) {
    setBattlefieldGameOver();
    stopAuto();
    showGameOver(result);
    setModeLabel('GAME OVER');
    setStatusLive(false);
    renderButtonStates({ isRunning: false, isGameOver: true });
  }

  AppState.lastResult = result;
}

// ----------------------------------------------------------
// RENDER: INITIAL STATE (page load / after reset)
// ----------------------------------------------------------

function renderInitialState(state) {
  const labels = AppState.actions;

  renderTopBar(state);
  renderHealthBar(state.system_health);
  renderEpsilon(state.epsilon);
  renderStats(state);

  if (labels) {
    const atkShorts = labels.attacks.map(a  => a.short);
    const defShorts = labels.defenses.map(d => d.short);

    renderQtable(El.atkQtable, state.atk_qtable.qtable, atkShorts, 'atk');
    renderQtable(El.defQtable, state.def_qtable.qtable, defShorts, 'def');
    renderProbBars(El.atkProbBars, state.atk_qtable.probabilities, atkShorts, 'atk');
    renderProbBars(El.defProbBars, state.def_qtable.probabilities, defShorts, 'def');
    renderBarChart(El.atkChart, state.atk_counts, atkShorts, 'atk');
    renderBarChart(El.defChart, state.def_counts, defShorts, 'def');
  }

  if (state.trend_data?.length >= 2) {
    renderTrendChart(state.trend_data);
  }

  setBattlefieldCritical(state.system_health <= 25);
  setBattlefieldStatus('IDLE', 'idle');
  setModeLabel('STANDBY');
  setStatusLive(false);
  renderButtonStates({ isRunning: false, isGameOver: state.is_game_over });

  // If resuming a game-over session
  if (state.is_game_over) {
    setBattlefieldGameOver();
    showGameOver(state);
    setModeLabel('GAME OVER');
  }

  // Replay last result into engagement panel if available
  if (state.last_result) {
    renderEngagement(state.last_result);
    appendTimelineChip(state.last_result);
  }
}

// ----------------------------------------------------------
// RESET UI (wipe all panels to blank state)
// ----------------------------------------------------------

function resetUI() {
  if (El.atkActionVal)  El.atkActionVal.textContent  = 'STANDBY';
  if (El.defActionVal)  El.defActionVal.textContent  = 'STANDBY';
  if (El.lastAtkName)   El.lastAtkName.textContent   = '—';
  if (El.lastDefName)   El.lastDefName.textContent   = '—';
  if (El.outcomeChip) {
    clearModifiers(El.outcomeChip, 'outcome-chip--');
    El.outcomeChip.textContent = '—';
  }
  if (El.atkReward) {
    clearModifiers(El.atkReward, 'reward-badge--');
    El.atkReward.textContent = '—';
  }
  if (El.defReward) {
    clearModifiers(El.defReward, 'reward-badge--');
    El.defReward.textContent = '—';
  }

  clearTimeline();
  clearLog();

  if (El.battlefield) {
    El.battlefield.classList.remove(
      'is-scanning', 'is-critical', 'is-game-over'
    );
  }
  if (El.bfPanel) {
    El.bfPanel.classList.remove('is-damaged', 'is-defended');
  }
  if (El.targetCore) {
    El.targetCore.classList.remove('is-breached', 'is-defended');
  }
  if (El.bfAnimLayer) {
    El.bfAnimLayer.innerHTML = '';
  }

  if (AppState.trendChartObj) {
    AppState.trendChartObj.destroy();
    AppState.trendChartObj = null;
  }

  setBattlefieldStatus('IDLE', 'idle');
  setStatusLive(false);
  setModeLabel('STANDBY');
  renderButtonStates({ isRunning: false, isGameOver: false });
  hideGameOver();
}

// ----------------------------------------------------------
// AUTO-RUN LOOP
// ----------------------------------------------------------

function startAuto() {
  if (AppState.isAutoRunning) return;
  AppState.isAutoRunning  = true;
  AppState.autoIntervalId = setInterval(async () => {
    await handleStep();
  }, AppState.autoSpeed);

  setBattlefieldScanning(true);
  setStatusLive(true);
  setModeLabel('AUTO');
  renderButtonStates({ isRunning: true, isGameOver: false });
}

function stopAuto() {
  if (AppState.autoIntervalId) {
    clearInterval(AppState.autoIntervalId);
    AppState.autoIntervalId = null;
  }
  AppState.isAutoRunning = false;
  setBattlefieldScanning(false);
}

// ----------------------------------------------------------
// ACTION HANDLERS
// ----------------------------------------------------------

/**
 * Run one simulation step via POST /simulation/step.
 */
async function handleStep() {
  const result = await fetchStep(1);
  if (!result) return;

  renderRound(result);

  // Stop auto if game over
  if (result.is_game_over) {
    stopAuto();
  }
}

/**
 * Toggle auto-run mode on/off.
 */
function handleToggleAuto() {
  if (AppState.isAutoRunning) {
    stopAuto();
    setModeLabel('PAUSED');
    setStatusLive(false);
    renderButtonStates({ isRunning: false, isGameOver: false });
  } else {
    startAuto();
  }
}

/**
 * Pause auto-run.
 */
function handlePause() {
  if (!AppState.isAutoRunning) return;
  stopAuto();
  setModeLabel('PAUSED');
  setStatusLive(false);
  renderButtonStates({ isRunning: false, isGameOver: false });
}

/**
 * Reset simulation via POST /simulation/reset, then reload state.
 */
async function handleReset() {
  stopAuto();
  resetUI();
  setModeLabel('RESETTING...');

  const resetRes = await fetchReset();
  if (!resetRes) {
    setModeLabel('RESET FAILED');
    return;
  }

  // Fetch fresh state after reset
  const state = await fetchState();
  if (state) {
    renderInitialState(state);
  }

  setModeLabel('STANDBY');
}

/**
 * Update auto speed from slider.
 */
function handleSpeedChange(val) {
  AppState.autoSpeed = parseInt(val, 10);
  if (El.speedLabel) {
    El.speedLabel.textContent = (AppState.autoSpeed / 1000).toFixed(1) + 's';
  }
  // Restart auto with new speed if currently running
  if (AppState.isAutoRunning) {
    stopAuto();
    startAuto();
  }
}

// ----------------------------------------------------------
// EVENT LISTENERS
// ----------------------------------------------------------

function bindEvents() {
  El.btnRun?.addEventListener('click', handleStep);

  El.btnAuto?.addEventListener('click', handleToggleAuto);

  El.btnPause?.addEventListener('click', handlePause);

  El.btnReset?.addEventListener('click', handleReset);

  El.speedSlider?.addEventListener('input', e => {
    handleSpeedChange(e.target.value);
  });
  El.epsilonSlider?.addEventListener('input', e => {

  const val = parseFloat(e.target.value);

  if (El.epsilonValue) {
    El.epsilonValue.textContent = Math.round(val * 100) + '%';
  }

});

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    switch (e.key) {
      case ' ':
        e.preventDefault();
        if (AppState.isAutoRunning) handlePause();
        else handleStep();
        break;
      case 'a':
      case 'A':
        handleToggleAuto();
        break;
      case 'r':
      case 'R':
        if (e.ctrlKey || e.metaKey) break; // don't intercept Ctrl+R
        handleReset();
        break;
    }
  });
}

// ----------------------------------------------------------
// STARTUP
// ----------------------------------------------------------

async function init() {
  // 1. Bind all event listeners
  bindEvents();

  // 2. Fetch action label metadata (cached for session)
  AppState.actions = await fetchActions();

  // 3. Fetch current simulation state from backend
  const state = await fetchState();
  if (!state) {
    setModeLabel('BACKEND OFFLINE');
    return;
  }

  // 4. Render initial HUD state
  renderInitialState(state);

  console.log(
    `%c[CYBER BATTLE] HUD online. Session: ${state.session_id}`,
    'color:#00ff9d;font-family:monospace;'
  );
}

// ----------------------------------------------------------
// BOOT
// ----------------------------------------------------------
document.addEventListener('DOMContentLoaded', init);