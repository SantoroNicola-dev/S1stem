import { S1stemEngine } from '../../core/s1stem.js';
import { SnakeGame, SNAKE_ACTIONS, HIERARCHICAL_GOALS } from './snake-game.js';
import { SNAKE_WEIGHTS } from './snake-weights.js';
import { NerdTelemetry } from '../../core/nerd-telemetry.js';

// Dedicated Snake S1stem Instance with specialized weights
const s1 = new S1stemEngine({
  weights: SNAKE_WEIGHTS,
  temperature: 0.07,
});
const game = new SnakeGame(16, 25);
const telemetry = new NerdTelemetry({
  domain: 'snake',
  temperature: 0.07,
  stateDim: 384,
  projDim: 128,
  streamId: 'log-stream',
  statsContainerId: 'nerd-stats-grid',
});

let isRunning = false;
let ticksPerSec = 7;
let loopTimer = null;

// DOM references
const canvas = document.getElementById('snake-canvas');
const scoreEl = document.getElementById('game-score');
const movesEl = document.getElementById('game-moves');
const applesEl = document.getElementById('game-apples');
const latencyEl = document.getElementById('decision-latency');
const chosenDirEl = document.getElementById('chosen-direction');
const promptDisplay = document.getElementById('prompt-display');
const barsContainer = document.getElementById('candidate-bars');
const goalsContainer = document.getElementById('goals-list');

let currentEvaluationMode = 'visual'; // 'visual' | 'text'

const btnToggle = document.getElementById('btn-toggle-play');
const btnStep = document.getElementById('btn-step-frame');
const btnReset = document.getElementById('btn-reset-game');
const btnModeToggle = document.getElementById('btn-mode-toggle');
const speedSlider = document.getElementById('speed-slider');
const speedLabel = document.getElementById('speed-val');

if (btnModeToggle) {
  btnModeToggle.addEventListener('click', () => {
    currentEvaluationMode = currentEvaluationMode === 'visual' ? 'text' : 'visual';
    btnModeToggle.textContent = currentEvaluationMode === 'visual' ? '👁️ Visual Frame' : '📝 Text State';
    const badge = document.getElementById('ai-status-badge');
    if (badge) {
      badge.textContent = currentEvaluationMode === 'visual' ? 'QWEN3.5 VISION' : 'QWEN3.5 TEXT';
    }
  });
}

// Render Goals List
function renderGoals() {
  if (!goalsContainer) return;
  goalsContainer.innerHTML = '';
  HIERARCHICAL_GOALS.forEach((g) => {
    const card = document.createElement('div');
    card.className = `goal-card p-${g.priority}`;
    card.innerHTML = `
      <div class="goal-header">
        <span class="goal-title">${g.label}</span>
        <span class="goal-badge">Priority ${g.priority}</span>
      </div>
      <p class="goal-desc">${g.description}</p>
    `;
    goalsContainer.appendChild(card);
  });
}

// Canvas Rendering
function renderCanvas() {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const cellSize = w / game.gridSize;

  // Background
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, w, h);

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= game.gridSize; i++) {
    const pos = i * cellSize;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(w, pos);
    ctx.stroke();
  }

  // Draw Food
  const fx = game.food.x * cellSize;
  const fy = game.food.y * cellSize;
  ctx.save();
  if (game.food.isBonus) {
    ctx.fillStyle = '#ef4444';
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(fx + cellSize / 2, fy + cellSize / 2, cellSize / 2.3, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.fillStyle = '#10b981';
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(fx + cellSize / 2, fy + cellSize / 2, cellSize / 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // Draw Snake Body
  game.snake.forEach((seg, idx) => {
    const x = seg.x * cellSize;
    const y = seg.y * cellSize;
    const isHead = idx === 0;

    ctx.save();
    if (isHead) {
      ctx.fillStyle = '#10b981';
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.roundRect(x + 1, y + 1, cellSize - 2, cellSize - 2, 6);
      ctx.fill();

      // Eyes
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(x + cellSize * 0.35, y + cellSize * 0.35, 2, 0, Math.PI * 2);
      ctx.arc(x + cellSize * 0.65, y + cellSize * 0.35, 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const alpha = Math.max(0.35, 1 - (idx / game.snake.length) * 0.6);
      ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x + 2, y + 2, cellSize - 4, cellSize - 4, 4);
      ctx.fill();
    }
    ctx.restore();
  });

  // Floating score popups
  game.floatingPoints.forEach((f) => {
    ctx.save();
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillStyle = `rgba(52, 211, 153, ${f.opacity})`;
    ctx.fillText(f.text, f.x * cellSize + 4, f.y * cellSize - 4);
    ctx.restore();
  });
}

function updateTelemetry(decision, chosenDir, statePrompt, evals) {
  if (scoreEl) scoreEl.textContent = `${game.score}`;
  if (movesEl) movesEl.textContent = `${game.movesCount}`;
  if (applesEl) applesEl.textContent = `${game.applesEaten || Math.floor(game.score / 25)}`;
  if (promptDisplay && statePrompt) promptDisplay.textContent = statePrompt;

  if (latencyEl && decision) {
    latencyEl.textContent = `${decision.latencyMs.toFixed(3)} ms`;
  }

  if (chosenDirEl && decision) {
    chosenDirEl.textContent = `${chosenDir} (${(decision.confidence * 100).toFixed(0)}%)`;
  }

  // Update Action Distribution Bars with Unified Styling
  if (barsContainer && decision && decision.ranking) {
    barsContainer.innerHTML = '';
    const dirMap = {
      MOVE_UP: 'UP (North)',
      MOVE_RIGHT: 'RIGHT (East)',
      MOVE_DOWN: 'DOWN (South)',
      MOVE_LEFT: 'LEFT (West)',
    };

    decision.ranking.forEach((r) => {
      const isWinner = r.action.id.includes(chosenDir);
      const row = document.createElement('div');
      row.className = `action-row ${isWinner ? 'winner' : ''}`;

      const pct = (r.probability * 100).toFixed(1);
      row.innerHTML = `
        <span class="action-label">${dirMap[r.action.id] || r.action.label}</span>
        <div class="action-track">
          <div class="action-bar" style="width: ${Math.min(100, Math.max(3, pct))}%"></div>
        </div>
        <span class="action-pct font-mono">${pct}%</span>
      `;
      barsContainer.appendChild(row);
    });
  }

  // Record into Nerd Telemetry & Live Diagnostic Stream with prominent %
  if (decision) {
    const optimalDir = evals?.find((e) => e.isOptimalShortest)?.dir || '-';
    telemetry.record({
      action: `MOVE_${chosenDir}`,
      confidence: decision.confidence,
      latencyMs: decision.latencyMs,
      ranking: decision.ranking,
      details: `Score:${game.score} | BFS:${optimalDir}`,
    });
  }
}

// Single step execution
async function stepSimulation() {
  if (game.isGameOver || game.isGameWon) {
    pauseGame();
    return;
  }

  // 1. Evaluate spatial candidate corridors
  const evals = game.evaluateDirections(); // UP, RIGHT, DOWN, LEFT

  // Priority 1: Survive - strictly mask out fatal collisions and 180° turnaround
  // Priority 2: Prevent Traps - if there's any viable open escape, mask out dead-end pockets
  const hasOpenEscape = evals.some((e) => e.isSafe && !e.isDeadEnd);
  const mask = evals.map((e) => {
    if (!e.isSafe) return false;
    if (hasOpenEscape && e.isDeadEnd) return false;
    return true;
  });

  const safeMask = mask.some(Boolean) ? mask : evals.map((e) => e.isSafe);

  // Policy Biases (beta_i):
  // Priority 4: Shortest BFS geodesic corridor receives strong positive bonus
  // Target proximity and reachable open space bonuses
  const biases = evals.map((e) => {
    if (!e.isSafe) return -999;
    let b = 0;
    if (e.isOptimalShortest) b += 5.0;
    else if (e.distanceToFood < 999) b += Math.max(0, (20 - e.distanceToFood) * 0.3);
    b += (e.freeSpace / 256) * 1.5;
    return b;
  });

  // 2. Generate descriptive state prompt
  const statePrompt = game.generateStatePrompt();

  // 3. Query S1stem v1.1.0 engine (Qwen3.5 VLM)
  let decision;
  if (currentEvaluationMode === 'visual') {
    decision = await s1.choice(canvas, { text: statePrompt, mask: safeMask, biases });
  } else {
    decision = await s1.choice(statePrompt, { mask: safeMask, biases });
  }

  // Extract chosen direction
  let chosenDir = 'RIGHT';
  if (decision.selectedAction.id.includes('UP')) chosenDir = 'UP';
  else if (decision.selectedAction.id.includes('DOWN')) chosenDir = 'DOWN';
  else if (decision.selectedAction.id.includes('LEFT')) chosenDir = 'LEFT';
  else if (decision.selectedAction.id.includes('RIGHT')) chosenDir = 'RIGHT';

  // 4. Advance simulation
  game.step(chosenDir);

  // 5. Update graphics and telemetry
  renderCanvas();
  updateTelemetry(decision, chosenDir, statePrompt, evals);
}

function playGame() {
  if (isRunning) return;
  isRunning = true;
  if (btnToggle) {
    btnToggle.innerHTML = `<span>⏸ Pause</span>`;
    btnToggle.classList.remove('btn-emerald');
    btnToggle.classList.add('btn-secondary');
  }

  const intervalMs = Math.round(1000 / ticksPerSec);
  loopTimer = setInterval(stepSimulation, intervalMs);
}

function pauseGame() {
  if (!isRunning) return;
  isRunning = false;
  if (btnToggle) {
    btnToggle.innerHTML = `<span>▶ Play</span>`;
    btnToggle.classList.remove('btn-secondary');
    btnToggle.classList.add('btn-emerald');
  }
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
}

function resetGame() {
  game.reset();
  renderCanvas();
  telemetry.logSystem('Snake environment reset.');
  updateTelemetry(null, '-', 'Standing by for first tick...', null);
}

// Event Listeners
btnToggle?.addEventListener('click', () => {
  if (isRunning) pauseGame();
  else playGame();
});

btnStep?.addEventListener('click', () => {
  pauseGame();
  stepSimulation();
});

btnReset?.addEventListener('click', () => {
  pauseGame();
  resetGame();
});

speedSlider?.addEventListener('input', (e) => {
  ticksPerSec = parseInt(e.target.value, 10);
  if (speedLabel) speedLabel.textContent = `${ticksPerSec} tps`;
  if (isRunning) {
    pauseGame();
    playGame();
  }
});

// Boot initialization
async function boot() {
  // Pre-cache snake actions
  await s1.registerActions(SNAKE_ACTIONS);
  renderGoals();
  renderCanvas();
  telemetry.logSystem('Snake S1stem initialized (384 -> 128).');
}

boot();
