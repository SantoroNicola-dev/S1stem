/**
 * CYBER-ROVER TACTICAL EDGE CORE // S1STEM-CDR
 * Two-Tier Hybrid Cascade Decision Routing (Reflex <0.20ms + WebGPU Neural Escalation)
 * 100% Original Intellectual Property
 */

import { S1stemCDREngine, extractPolarHypotheses } from '../../core/s1stem.js';
import { RoverGame, DIRS, TILE } from './rover-game.js';
import { NerdTelemetry } from '../../core/nerd-telemetry.js';

// Initialize S1stem-CDR Engine
const s1 = new S1stemCDREngine({
  mode: 'cascade',
  affinityThreshold: 0.38,
  confidenceThreshold: 0.65,
  marginThreshold: 0.15,
  temperature: 0.08,
});

// Initialize Game & Telemetry
const game = new RoverGame();
const telemetry = new NerdTelemetry({
  domain: 'rover-cdr',
  temperature: 0.08,
  stateDim: 384,
  projDim: 128,
  streamId: 'log-stream',
  statsContainerId: 'nerd-stats-grid',
});

// State & Flags
let isRunning = false;
let ticksPerSec = 10;
let loopTimer = null;
let currentMode = 'visual'; // 'visual' | 'text'
let isNeuralLoaded = false;
let isNeuralLoading = false;
let soundEnabled = false;

let stats = {
  totalDecisions: 0,
  tier1Exits: 0,
  tier2Escalations: 0,
  lastLatencyMs: 0.12,
  avgReflexLatency: 0.12,
  avgNeuralLatency: 14.5,
  reflexSum: 0,
  neuralSum: 0,
};

// DOM References
const canvas = document.getElementById('rover-canvas');
const scoreEl = document.getElementById('game-score');
const shieldsEl = document.getElementById('game-shields');
const cellsEl = document.getElementById('game-cells');
const capacitorEl = document.getElementById('game-capacitor');

// Telemetry & Badges
const tierBadge = document.getElementById('tier-badge');
const latencyEl = document.getElementById('decision-latency');
const latencySub = document.getElementById('decision-latency-sub');
const confidenceEl = document.getElementById('decision-confidence');
const marginEl = document.getElementById('decision-margin');
const escalationReasonEl = document.getElementById('escalation-reason');
const chosenActionEl = document.getElementById('chosen-action');
const promptDisplay = document.getElementById('prompt-display');
const barsContainer = document.getElementById('candidate-bars');

// Routing Stats
const statTotalDecisions = document.getElementById('stat-total-decisions');
const statTier1Pct = document.getElementById('stat-tier1-pct');
const statTier2Pct = document.getElementById('stat-tier2-pct');
const statTier1Bar = document.getElementById('stat-tier1-bar');
const statTier2Bar = document.getElementById('stat-tier2-bar');

// Noul Card
const noulQuestionEl = document.getElementById('noul-question');
const noulBarYes = document.getElementById('noul-bar-yes');
const noulBarNo = document.getElementById('noul-bar-no');
const noulHypoYes = document.getElementById('noul-hypo-yes');
const noulHypoNo = document.getElementById('noul-hypo-no');

// Safety Score
const safetyScoreVal = document.getElementById('safety-score-val');
const safetyDots = document.querySelectorAll('.safety-dot');

// Sentries Cards
const sentryStrip = document.getElementById('sentry-strip');

// Controls
const btnToggle = document.getElementById('btn-toggle-play');
const btnStep = document.getElementById('btn-step-frame');
const btnReset = document.getElementById('btn-reset-game');
const btnSound = document.getElementById('btn-sound-toggle');
const btnLoadNeural = document.getElementById('btn-load-neural');
const neuralStatusPill = document.getElementById('neural-status-pill');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-val');

// Mode Switchers
const modeBtns = document.querySelectorAll('.mode-btn');

// --- Web Audio Synthesizer (Zero External Audio Files) ---
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
      audioCtx = new AudioContext();
    }
  }
}

function playTone(freq, type = 'sine', duration = 0.08, vol = 0.08) {
  if (!soundEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch {}
}

function playEscalationChime() {
  if (!soundEnabled || !audioCtx) return;
  // Futuristic dual ping for Tier 2 neural escalation
  playTone(659.25, 'triangle', 0.12, 0.12); // E5
  setTimeout(() => playTone(987.77, 'triangle', 0.20, 0.12), 60); // B5
}

function playHarvestTone() {
  if (!soundEnabled || !audioCtx) return;
  const f = (game.tickCount % 2 === 0) ? 440 : 554.37;
  playTone(f, 'sine', 0.05, 0.04);
}

// --- Background Neural Backbone Loader (WebGPU / WASM) ---
async function loadNeuralPipeline() {
  if (isNeuralLoading || isNeuralLoaded) return;
  isNeuralLoading = true;
  if (btnLoadNeural) {
    btnLoadNeural.disabled = true;
    btnLoadNeural.textContent = 'Initializing WebGPU...';
  }
  if (neuralStatusPill) {
    neuralStatusPill.textContent = 'LOADING (WEBGPU)...';
    neuralStatusPill.className = 'badge badge-amber';
  }

  const success = await s1.loadNeuralBackbone((p) => {
    if (btnLoadNeural && p.message) {
      btnLoadNeural.textContent = p.message.slice(0, 24);
    }
  });

  isNeuralLoading = false;
  isNeuralLoaded = success;

  if (success) {
    if (btnLoadNeural) {
      btnLoadNeural.textContent = '🧠 Neural Online (WebGPU)';
      btnLoadNeural.classList.add('btn-emerald');
    }
    if (neuralStatusPill) {
      neuralStatusPill.textContent = 'ONLINE (WEBGPU/WASM)';
      neuralStatusPill.className = 'badge badge-green';
    }
    telemetry.logEvent('Neural backbone MiniLM loaded onto WebGPU/WASM pipeline.', 'success');
  } else {
    if (btnLoadNeural) {
      btnLoadNeural.disabled = false;
      btnLoadNeural.textContent = 'Retry WebGPU Load';
    }
    if (neuralStatusPill) {
      neuralStatusPill.textContent = 'REFLEX FALLBACK';
      neuralStatusPill.className = 'badge badge-cyan';
    }
  }
}

// --- Canvas Graphics Rendering ---
function renderCanvas() {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const cellW = w / game.cols;
  const cellH = h / game.rows;

  // 1. Dark Cybernetic Grid Backdrop
  ctx.fillStyle = '#05070c';
  ctx.fillRect(0, 0, w, h);

  // Subtle conduit lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= game.cols; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cellW, 0);
    ctx.lineTo(x * cellW, h);
    ctx.stroke();
  }
  for (let y = 0; y <= game.rows; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * cellH);
    ctx.lineTo(w, y * cellH);
    ctx.stroke();
  }

  // 2. Outpost Matrix Walls & Collectibles
  for (let y = 0; y < game.rows; y++) {
    for (let x = 0; x < game.cols; x++) {
      const t = game.grid[y][x];
      const cx = x * cellW + cellW / 2;
      const cy = y * cellH + cellH / 2;

      if (t === TILE.WALL) {
        // Reinforced High-Tech Bulkhead Wall
        ctx.fillStyle = '#111827';
        ctx.strokeStyle = 'rgba(14, 165, 233, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x * cellW + 1, y * cellH + 1, cellW - 2, cellH - 2, 4);
        ctx.fill();
        ctx.stroke();
      } else if (t === TILE.DOOR) {
        // Sentry Containment Gate
        ctx.fillStyle = 'rgba(236, 72, 153, 0.65)';
        ctx.fillRect(x * cellW + 1, y * cellH + cellH * 0.35, cellW - 2, cellH * 0.3);
      } else if (t === TILE.PLASMA) {
        // Glowing Green Plasma Energy Cell
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (t === TILE.CAPACITOR) {
        // Pulsating Supercharge Capacitor
        const pulse = 1.0 + Math.sin(game.tickCount * 0.15) * 0.25;
        ctx.save();
        ctx.fillStyle = '#00f0ff';
        ctx.shadowColor = '#00f0ff';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(cx, cy, 5.5 * pulse, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      } else if (t === TILE.DATA) {
        // Classified Data Terminal (Diamond)
        ctx.fillStyle = '#ec4899';
        ctx.beginPath();
        ctx.moveTo(cx, cy - 6);
        ctx.lineTo(cx + 6, cy);
        ctx.lineTo(cx, cy + 6);
        ctx.lineTo(cx - 6, cy);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  // 3. Draw Rover Unit TR-01
  const rx = game.rover.subX * cellW + cellW / 2;
  const ry = game.rover.subY * cellH + cellH / 2;
  const radius = Math.min(cellW, cellH) * 0.44;

  let angle = 0;
  if (game.rover.dir === 'RIGHT') angle = 0;
  else if (game.rover.dir === 'DOWN') angle = 0.5 * Math.PI;
  else if (game.rover.dir === 'LEFT') angle = Math.PI;
  else if (game.rover.dir === 'UP') angle = 1.5 * Math.PI;

  ctx.save();
  ctx.translate(rx, ry);
  ctx.rotate(angle);

  // Headlight scanning cone
  const grad = ctx.createRadialGradient(radius * 0.8, 0, 2, radius * 2.8, 0, radius * 2.5);
  grad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
  grad.addColorStop(1, 'rgba(0, 240, 255, 0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(radius * 0.6, -radius * 0.4);
  ctx.lineTo(radius * 3.0, -radius * 1.8);
  ctx.lineTo(radius * 3.0, radius * 1.8);
  ctx.lineTo(radius * 0.6, radius * 0.4);
  ctx.closePath();
  ctx.fill();

  // Rover Chassis (Octagonal Tactical Hull)
  ctx.fillStyle = '#059669';
  ctx.strokeStyle = '#34d399';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#10b981';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.roundRect(-radius, -radius * 0.8, radius * 2, radius * 1.6, 5);
  ctx.fill();
  ctx.stroke();

  // Core Reactor (Pulsing Diamond)
  ctx.fillStyle = '#00f0ff';
  ctx.beginPath();
  ctx.arc(0, 0, radius * 0.35, 0, Math.PI * 2);
  ctx.fill();

  // Directional Arrow Indicator
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(radius * 0.6, 0);
  ctx.lineTo(radius * 0.1, -radius * 0.35);
  ctx.lineTo(radius * 0.1, radius * 0.35);
  ctx.closePath();
  ctx.fill();

  ctx.restore();

  // 4. Draw Rogue Sentries
  game.sentries.forEach((s) => {
    const sx = s.subX * cellW + cellW / 2;
    const sy = s.subY * cellH + cellH / 2;
    const sRadius = Math.min(cellW, cellH) * 0.42;

    ctx.save();

    if (s.state === 'RECHARGING') {
      // Low-power floating sensor beacon
      ctx.fillStyle = '#a855f7';
      ctx.beginPath();
      ctx.arc(sx, sy, sRadius * 0.4, 0, Math.PI * 2);
      ctx.fill();
    } else {
      let sColor = s.color;
      if (s.state === 'STUNNED') {
        const isFlashing = game.overclockTimeLeft < 90 && (Math.floor(game.tickCount / 8) % 2 === 0);
        sColor = isFlashing ? '#ffffff' : '#38bdf8';
      }

      ctx.fillStyle = sColor;
      ctx.shadowColor = sColor;
      ctx.shadowBlur = 10;

      // Hexagonal Drone Chassis
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (i * Math.PI) / 3;
        const hx = sx + Math.cos(a) * sRadius;
        const hy = sy + Math.sin(a) * sRadius;
        if (i === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
      }
      ctx.closePath();
      ctx.fill();

      // Sentry Central Eye / Sensor Lens
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(sx, sy, sRadius * 0.45, 0, Math.PI * 2);
      ctx.fill();

      // Directional Pupil Lens
      const d = DIRS[s.dir] || DIRS.RIGHT;
      ctx.fillStyle = s.state === 'STUNNED' ? '#38bdf8' : '#ef4444';
      ctx.beginPath();
      ctx.arc(sx + d.dx * 3, sy + d.dy * 3, sRadius * 0.22, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  });

  // 5. Draw Floating Alerts & Particles
  game.particles.forEach((p) => {
    ctx.save();
    ctx.fillStyle = p.color;
    ctx.globalAlpha = p.life;
    ctx.beginPath();
    ctx.arc(p.x * cellW + cellW / 2, p.y * cellH + cellH / 2, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });

  game.floatingAlerts.forEach((f) => {
    ctx.save();
    ctx.font = 'bold 11px JetBrains Mono, monospace';
    ctx.fillStyle = f.color || '#38bdf8';
    ctx.globalAlpha = f.opacity;
    ctx.fillText(f.text, f.x * cellW, f.y * cellH);
    ctx.restore();
  });
}

// --- Live Telemetry Update ---
function updateTelemetry(decision, chosenDir, statePrompt, noulRes, scoreRes) {
  if (scoreEl) scoreEl.textContent = `${game.score}`;
  if (shieldsEl) shieldsEl.textContent = `${game.shields}`;
  if (cellsEl) cellsEl.textContent = `${game.totalCells - game.cellsHarvested}`;
  if (capacitorEl) {
    if (game.overclockTimeLeft > 0) {
      capacitorEl.textContent = `${(game.overclockTimeLeft / 60).toFixed(1)}s (OVERCLOCK)`;
      capacitorEl.style.color = 'var(--accent-cyan)';
    } else {
      capacitorEl.textContent = 'STANDARD';
      capacitorEl.style.color = 'var(--text-dim)';
    }
  }

  if (promptDisplay && statePrompt) {
    promptDisplay.textContent = statePrompt;
  }

  // Decision & Tier Badge
  if (decision) {
    const isEscalated = decision.tier === 'tier2_neural' || decision.escalated;
    if (tierBadge) {
      if (isEscalated) {
        tierBadge.className = 'tier-badge neural';
        tierBadge.innerHTML = '<span>🧠</span><span>TIER 2 NEURAL (ESCALATED)</span>';
      } else {
        tierBadge.className = 'tier-badge reflex';
        tierBadge.innerHTML = '<span>⚡</span><span>TIER 1 FAST REFLEX (&lt;0.2ms)</span>';
      }
    }

    if (latencyEl) {
      latencyEl.textContent = `${decision.latencyMs.toFixed(3)} ms`;
      if (isEscalated) {
        latencyEl.className = 'metric-value highlight-orange';
        if (latencySub) latencySub.textContent = 'Escalated to WebGPU MiniLM backbone';
      } else {
        latencyEl.className = 'metric-value highlight-emerald';
        if (latencySub) latencySub.textContent = 'Sub-millisecond unit hypersphere S¹²⁷ exit';
      }
    }

    if (confidenceEl) {
      confidenceEl.textContent = `${(decision.confidence * 100).toFixed(1)}%`;
    }

    if (marginEl) {
      const marginVal = decision.margin !== undefined ? decision.margin : 0;
      marginEl.textContent = `Δ ${marginVal.toFixed(2)}`;
    }

    if (escalationReasonEl) {
      if (isEscalated) {
        const marg = decision.tier1Margin !== undefined ? decision.tier1Margin.toFixed(2) : '-';
        const aff = decision.tier1DotProduct !== undefined ? decision.tier1DotProduct.toFixed(2) : '-';
        escalationReasonEl.textContent = `ESCALATED: Low Margin Δ=${marg} or Affinity Dot=${aff} (Multi-Sentry Encirclement Dilemma)`;
        escalationReasonEl.style.color = '#f472b6';
      } else {
        escalationReasonEl.textContent = `FAST PATH EXIT: High Affinity & Confidence Margin Clear`;
        escalationReasonEl.style.color = '#10b981';
      }
    }

    if (chosenActionEl) {
      chosenActionEl.textContent = `${chosenDir} (${(decision.confidence * 100).toFixed(0)}%)`;
    }

    // Action Distribution Ranking Bars
    if (barsContainer && decision.ranking) {
      barsContainer.innerHTML = '';
      decision.ranking.forEach((r) => {
        const isWinner = r.action.id.includes(chosenDir);
        const row = document.createElement('div');
        row.className = `action-row ${isWinner ? 'winner' : ''}`;
        const pct = (r.probability * 100).toFixed(1);

        row.innerHTML = `
          <span class="action-label" title="${r.action.description}">${r.action.label}</span>
          <div class="action-track">
            <div class="action-bar ${isEscalated ? 'neural-bar' : ''}" style="width: ${Math.min(100, Math.max(3, pct))}%"></div>
          </div>
          <span class="action-pct font-mono">${pct}%</span>
        `;
        barsContainer.appendChild(row);
      });
    }

    // Update Cumulative Routing Metrics
    stats.totalDecisions++;
    if (isEscalated) {
      stats.tier2Escalations++;
      stats.neuralSum += decision.latencyMs;
      stats.avgNeuralLatency = stats.neuralSum / stats.tier2Escalations;
    } else {
      stats.tier1Exits++;
      stats.reflexSum += decision.latencyMs;
      stats.avgReflexLatency = stats.reflexSum / stats.tier1Exits;
    }

    const t1Pct = ((stats.tier1Exits / stats.totalDecisions) * 100).toFixed(1);
    const t2Pct = ((stats.tier2Escalations / stats.totalDecisions) * 100).toFixed(1);

    if (statTotalDecisions) statTotalDecisions.textContent = `${stats.totalDecisions}`;
    if (statTier1Pct) statTier1Pct.textContent = `${t1Pct}% (${stats.tier1Exits})`;
    if (statTier2Pct) statTier2Pct.textContent = `${t2Pct}% (${stats.tier2Escalations})`;
    if (statTier1Bar) statTier1Bar.style.width = `${t1Pct}%`;
    if (statTier2Bar) statTier2Bar.style.width = `${t2Pct}%`;
  }

  // Polar Noul Widget Update
  if (noulRes) {
    const polar = game.generatePolarHypotheses()[0];
    if (noulQuestionEl) noulQuestionEl.textContent = `"${polar.question}"`;

    const pYes = (noulRes.probability?.yes ?? 0) * 100;
    const pNo = (noulRes.probability?.no ?? 0) * 100;

    if (noulBarYes) {
      noulBarYes.style.width = `${Math.max(10, pYes)}%`;
      noulBarYes.textContent = `YES ${pYes.toFixed(0)}%`;
    }
    if (noulBarNo) {
      noulBarNo.style.width = `${Math.max(10, pNo)}%`;
      noulBarNo.textContent = `NO ${pNo.toFixed(0)}%`;
    }

    const decomp = extractPolarHypotheses(polar.question);
    if (noulHypoYes && decomp) noulHypoYes.textContent = decomp.affirmative;
    if (noulHypoNo && decomp) noulHypoNo.textContent = decomp.negative;
  }

  // Safety Score Update (from s1.score())
  if (scoreRes && safetyScoreVal) {
    const val = typeof scoreRes.topScore === 'number' ? scoreRes.topScore : Number(scoreRes.winner) || 3;
    safetyScoreVal.textContent = `${val}/5`;
    safetyDots.forEach((dot, idx) => {
      dot.className = `safety-dot ${idx < val ? 'active' : ''}`;
    });
  }

  // Update Sentries Strip
  if (sentryStrip) {
    sentryStrip.innerHTML = '';
    const rx = Math.round(game.rover.subX);
    const ry = Math.round(game.rover.subY);

    game.sentries.forEach((s) => {
      const d = game.dist(rx, ry, s.subX, s.subY).toFixed(1);
      const card = document.createElement('div');
      card.className = 'sentry-card';
      card.style.borderLeftColor = s.color;
      card.innerHTML = `
        <span class="sentry-name" style="color: ${s.color}">${s.name}</span>
        <span class="sentry-dist">${d}m</span>
        <span class="sentry-state-tag" style="color: ${s.state === 'STUNNED' ? '#38bdf8' : (s.state === 'RECHARGING' ? '#a855f7' : '#ef4444')}">${s.state}</span>
      `;
      sentryStrip.appendChild(card);
    });
  }
}

// --- Single Simulation Step ---
async function stepSimulation() {
  if (game.isGameOver || game.isWon) {
    pauseGame();
    return;
  }

  // 1. Evaluate Candidate Corridors
  const candidates = game.evaluateDirectionCandidates();
  const statePrompt = game.generateStatePrompt();

  const choices = candidates.map((c) => ({
    id: c.id,
    label: c.label,
    description: c.description,
  }));

  const hasSafe = candidates.some((c) => c.isWalkable && c.isSafe);
  const mask = candidates.map((c) => (hasSafe ? (c.isWalkable && c.isSafe) : c.isWalkable));
  const biases = candidates.map((c) => c.bias);

  // 2. Query S1stem v1.1.0 (Qwen3.5 VLM)
  let decision;
  try {
    if (currentMode === 'visual') {
      decision = await s1.choice(canvas, {
        text: statePrompt,
        choices,
        biases,
        mask,
      });
    } else {
      decision = await s1.choice(statePrompt, {
        choices,
        biases,
        mask,
      });
    }
  } catch (err) {
    console.error('S1stem Decision Error:', err);
    return;
  }

  let chosenDir = 'LEFT';
  if (decision.selectedAction?.id.includes('UP')) chosenDir = 'UP';
  else if (decision.selectedAction?.id.includes('RIGHT')) chosenDir = 'RIGHT';
  else if (decision.selectedAction?.id.includes('DOWN')) chosenDir = 'DOWN';
  else if (decision.selectedAction?.id.includes('LEFT')) chosenDir = 'LEFT';

  // 3. Query Polar Noul Arbiter
  let noulRes = null;
  try {
    const polarList = game.generatePolarHypotheses();
    noulRes = await s1.noul(statePrompt, polarList[0].question);
  } catch (err) {
    console.warn('Noul evaluation warning:', err);
  }

  // 4. Query Ordinal Safety Score Rubric
  let scoreRes = null;
  try {
    scoreRes = await s1.score(statePrompt, 'Cyber-Rover Sector Clearance and Hazard Safety', { min: 1, max: 5 });
  } catch (err) {
    console.warn('Score evaluation warning:', err);
  }

  // Sound triggers
  if (decision.tier === 'tier2_neural' || decision.escalated) {
    playEscalationChime();
  } else {
    playHarvestTone();
  }

  // 5. Advance Game State
  game.step(chosenDir);

  // 6. Update Graphics & Live Telemetry
  renderCanvas();
  updateTelemetry(decision, chosenDir, statePrompt, noulRes, scoreRes);
}

// --- Simulation Loop Controls ---
function playGame() {
  if (isRunning) return;
  initAudio();
  isRunning = true;
  if (btnToggle) {
    btnToggle.innerHTML = '<span>⏸</span><span>Pause AI</span>';
    btnToggle.classList.remove('btn-emerald');
    btnToggle.classList.add('btn-primary');
  }

  function loop() {
    if (!isRunning) return;
    stepSimulation().then(() => {
      const delay = Math.max(15, Math.floor(1000 / ticksPerSec));
      loopTimer = setTimeout(loop, delay);
    });
  }
  loop();
}

function pauseGame() {
  isRunning = false;
  if (loopTimer) clearTimeout(loopTimer);
  if (btnToggle) {
    btnToggle.innerHTML = '<span>▶</span><span>Start AI</span>';
    btnToggle.classList.remove('btn-primary');
    btnToggle.classList.add('btn-emerald');
  }
}

function resetGame() {
  pauseGame();
  game.reset();
  stats.totalDecisions = 0;
  stats.tier1Exits = 0;
  stats.tier2Escalations = 0;
  stats.reflexSum = 0;
  stats.neuralSum = 0;
  renderCanvas();
  updateTelemetry(null, 'IDLE', game.generateStatePrompt(), null, null);
}

// --- Scenario Presets (Stress-Testing CDR) ---
const SCENARIO_PRESETS = {
  pincer: {
    title: 'Double Sentry Pincer Ambush (Tier 2 Escalation)',
    setup: () => {
      game.rover.subX = 9;
      game.rover.subY = 16;
      game.rover.dir = 'LEFT';
      game.sentries[0].subX = 7;
      game.sentries[0].subY = 16;
      game.sentries[0].dir = 'RIGHT';
      game.sentries[1].subX = 11;
      game.sentries[1].subY = 16;
      game.sentries[1].dir = 'LEFT';
      game.overclockTimeLeft = 0;
    },
  },
  overclock: {
    title: 'Overclock Blitz (Reflex Salvage)',
    setup: () => {
      game.rover.subX = 2;
      game.rover.subY = 1;
      game.overclockTimeLeft = 360;
      game.sentries.forEach((s) => {
        s.state = 'STUNNED';
      });
      game.sentries[0].subX = 4;
      game.sentries[0].subY = 1;
    },
  },
  highway: {
    title: 'Plasma Conduit Highway (Tier 1 Fast Reflex Exit)',
    setup: () => {
      game.rover.subX = 2;
      game.rover.subY = 4;
      game.rover.dir = 'RIGHT';
      game.sentries.forEach((s) => {
        s.subX = 9;
        s.subY = 9;
      });
    },
  },
  paraphrase: {
    title: 'Semantic Paraphrase (Zero Lexical Overlap)',
    setup: () => {
      game.rover.subX = 14;
      game.rover.subY = 14;
      game.sentries[0].subX = 14;
      game.sentries[0].subY = 12;
      game.sentries[1].subX = 16;
      game.sentries[1].subY = 14;
    },
  },
};

// Wire Scenario Preset Buttons
document.querySelectorAll('.scenario-chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.scenario;
    if (SCENARIO_PRESETS[key]) {
      SCENARIO_PRESETS[key].setup();
      renderCanvas();
      stepSimulation();
    }
  });
});

// Wire Mode Switcher Buttons
modeBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    modeBtns.forEach((b) => b.classList.remove('active', 'visual', 'text'));
    currentMode = btn.dataset.mode;
    btn.classList.add('active', currentMode);

    if (currentMode === 'visual' && !isNeuralLoaded) {
      loadNeuralPipeline();
    }
    telemetry.logEvent(`S1stem routing mode switched to: ${currentMode.toUpperCase()}`);
  });
});

// Wire Control Buttons
if (btnToggle) {
  btnToggle.addEventListener('click', () => {
    if (isRunning) pauseGame();
    else playGame();
  });
}

if (btnStep) {
  btnStep.addEventListener('click', () => {
    pauseGame();
    stepSimulation();
  });
}

if (btnReset) {
  btnReset.addEventListener('click', resetGame);
}

if (btnSound) {
  btnSound.addEventListener('click', () => {
    soundEnabled = !soundEnabled;
    initAudio();
    btnSound.textContent = soundEnabled ? '🔊 Sound: ON' : '🔇 Sound: OFF';
    btnSound.classList.toggle('btn-accent', soundEnabled);
  });
}

if (btnLoadNeural) {
  btnLoadNeural.addEventListener('click', loadNeuralPipeline);
}

if (speedSlider) {
  speedSlider.addEventListener('input', (e) => {
    ticksPerSec = Number(e.target.value);
    if (speedLabel) speedLabel.textContent = `${ticksPerSec} Hz`;
  });
}

// Keyboard Navigation
window.addEventListener('keydown', (e) => {
  let keyDir = null;
  if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') keyDir = 'UP';
  else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyDir = 'RIGHT';
  else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') keyDir = 'DOWN';
  else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyDir = 'LEFT';
  else if (e.key === ' ') {
    e.preventDefault();
    if (isRunning) pauseGame();
    else playGame();
    return;
  }

  if (keyDir) {
    e.preventDefault();
    game.rover.nextDir = keyDir;
    if (!isRunning) {
      stepSimulation();
    }
  }
});

// Initial Setup
renderCanvas();
updateTelemetry(null, 'READY', game.generateStatePrompt(), null, null);

// Start autonomous simulation loop
setTimeout(() => {
  playGame();
}, 400);
