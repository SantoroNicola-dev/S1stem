import { S1stemEngine } from '../../core/s1stem.js';
import { TetrisGame, TETROMINOES, BOARD_WIDTH, BOARD_HEIGHT } from './tetris-game.js';
import { TETRIS_WEIGHTS } from './tetris-weights.js';
import { NerdTelemetry } from '../../core/nerd-telemetry.js';

// Dedicated Tetris S1stem Engine Instance with specialized weights
const s1 = new S1stemEngine({
  weights: TETRIS_WEIGHTS,
  temperature: 0.05,
});
const game = new TetrisGame();
const telemetry = new NerdTelemetry({
  domain: 'tetris',
  temperature: 0.05,
  stateDim: 384,
  projDim: 128,
  streamId: 'log-stream',
  statsContainerId: 'nerd-stats-grid',
});

let isRunning = false;
let stepsPerSec = 25;
let loopTimer = null;
let activeTargetPlacement = null;
let activeDecision = null;
let activePrompt = '';

let currentEvaluationMode = 'visual'; // 'visual' | 'text'

// DOM Elements
const canvas = document.getElementById('tetris-canvas');
const nextCanvas = document.getElementById('next-piece-canvas');
const scoreEl = document.getElementById('game-score');
const linesEl = document.getElementById('game-lines');
const levelEl = document.getElementById('game-level');
const tetrisCountEl = document.getElementById('game-tetris-count');
const latencyEl = document.getElementById('decision-latency');
const candidateCountEl = document.getElementById('candidate-count');
const chosenPlacementEl = document.getElementById('chosen-placement');
const rankingBarsEl = document.getElementById('ranking-bars');
const promptDisplay = document.getElementById('prompt-display');

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

// Next Piece Rendering
function renderNextPiece() {
  if (!nextCanvas || !game.nextPieceType) return;
  const ctx = nextCanvas.getContext('2d');
  if (!ctx) return;

  const w = nextCanvas.width;
  const h = nextCanvas.height;
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, w, h);

  const pieceData = TETROMINOES[game.nextPieceType];
  const shape = pieceData.shapes[0];
  const cellSize = 18;
  const shapeW = shape[0].length * cellSize;
  const shapeH = shape.length * cellSize;
  const startX = Math.floor((w - shapeW) / 2);
  const startY = Math.floor((h - shapeH) / 2);

  ctx.fillStyle = pieceData.color;
  ctx.shadowColor = pieceData.color;
  ctx.shadowBlur = 8;

  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      if (shape[r][c] !== 0) {
        ctx.fillRect(startX + c * cellSize + 1, startY + r * cellSize + 1, cellSize - 2, cellSize - 2);
      }
    }
  }
}

// Board Canvas Rendering
function renderBoard(chosenPlacement = null) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const cellSize = w / BOARD_WIDTH;

  // Background
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, w, h);

  // Subtle grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  ctx.lineWidth = 1;
  for (let c = 0; c <= BOARD_WIDTH; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cellSize, 0);
    ctx.lineTo(c * cellSize, h);
    ctx.stroke();
  }
  for (let r = 0; r <= BOARD_HEIGHT; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * cellSize);
    ctx.lineTo(w, r * cellSize);
    ctx.stroke();
  }

  // Draw Ghost Target Placement (if planned by AI)
  if (chosenPlacement && !game.isGameOver) {
    const shape = chosenPlacement.shape;
    const px = chosenPlacement.x;
    const py = chosenPlacement.dropY;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);

    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] !== 0) {
          const gx = (px + c) * cellSize;
          const gy = (py + r) * cellSize;
          ctx.strokeRect(gx + 2, gy + 2, cellSize - 4, cellSize - 4);
        }
      }
    }
    ctx.setLineDash([]);
  }

  // Draw Locked Grid Blocks
  for (let r = 0; r < BOARD_HEIGHT; r++) {
    for (let c = 0; c < BOARD_WIDTH; c++) {
      const type = game.grid[r][c];
      if (type !== 0) {
        const color = TETROMINOES[type]?.color || '#818cf8';
        const x = c * cellSize;
        const y = r * cellSize;

        ctx.save();
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, cellSize - 2, cellSize - 2, 3);
        ctx.fill();

        // Top-left block highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fillRect(x + 2, y + 2, cellSize - 4, 3);
        ctx.restore();
      }
    }
  }

  // Draw Current Falling Piece (with rotation and descent)
  if (game.currentPiece && !game.isGameOver) {
    const shape = game.currentPiece.shape;
    const px = game.currentPiece.x;
    const py = game.currentPiece.y;
    const color = game.currentPiece.color;

    ctx.save();
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c] !== 0) {
          const x = (px + c) * cellSize;
          const y = (py + r) * cellSize;
          ctx.beginPath();
          ctx.roundRect(x + 1, y + 1, cellSize - 2, cellSize - 2, 4);
          ctx.fill();
        }
      }
    }
    ctx.restore();
  }

  renderNextPiece();
}

function updateTelemetry(decision, candidateCount, statePrompt = null) {
  if (scoreEl) scoreEl.textContent = `${game.score}`;
  if (linesEl) linesEl.textContent = `${game.linesCleared}`;
  if (levelEl) levelEl.textContent = `${game.level}`;
  if (tetrisCountEl) tetrisCountEl.textContent = `${game.tetrisCount}`;
  if (candidateCountEl && candidateCount > 0) candidateCountEl.textContent = `${candidateCount}`;
  if (promptDisplay && statePrompt) promptDisplay.textContent = statePrompt;

  if (latencyEl && decision) {
    latencyEl.textContent = `${decision.latencyMs.toFixed(3)} ms`;
  }

  if (chosenPlacementEl && decision) {
    const winner = decision.selectedAction;
    chosenPlacementEl.textContent = `${winner.label} (${(decision.confidence * 100).toFixed(0)}%)`;
  }

  // Render Top Candidate Ranking Bars with Unified Styling
  if (rankingBarsEl && decision && decision.ranking) {
    rankingBarsEl.innerHTML = '';
    const topCandidates = decision.ranking.slice(0, 6);

    topCandidates.forEach((r, idx) => {
      const isWinner = idx === 0;
      const row = document.createElement('div');
      row.className = `action-row ${isWinner ? 'winner' : ''}`;

      const pct = (r.probability * 100).toFixed(1);
      row.innerHTML = `
        <span class="action-label">${r.action.label}</span>
        <div class="action-track">
          <div class="action-bar" style="width: ${Math.min(100, Math.max(3, pct))}%"></div>
        </div>
        <span class="action-pct font-mono">${pct}%</span>
      `;
      rankingBarsEl.appendChild(row);
    });
  }
}

// Single step execution (supports natural frame-by-frame descent & rotation)
async function stepSimulation() {
  if (game.isGameOver) {
    pauseGame();
    return;
  }

  // 1. If we don't have an active target placement, plan one with S1stem
  if (!activeTargetPlacement) {
    const candidates = game.generateCandidatePlacements();
    if (candidates.length === 0) {
      game.isGameOver = true;
      renderBoard();
      updateTelemetry(null, 0);
      pauseGame();
      return;
    }

    // Pre-cache candidate placements in Action Head Wa
    await s1.registerActions(candidates);

    // Formulate strategic state observation prompt
    activePrompt = game.generateStatePrompt();

    // Inject policy prior biases (beta_i):
    // Strong reward for lines, severe penalty for holes & surface bumpiness
    const biases = candidates.map((c) => {
      const m = c.metrics;
      return (
        m.completedLines * 4.5 -
        m.holes * 6.0 -
        m.bumpiness * 0.40 -
        m.maxHeight * 0.20
      );
    });

    // Evaluate placement with S1stem v1.1.0 (Qwen3.5 VLM)
    if (currentEvaluationMode === 'visual') {
      activeDecision = await s1.choice(canvas, { text: activePrompt, biases });
    } else {
      activeDecision = await s1.choice(activePrompt, { biases });
    }
    activeTargetPlacement = activeDecision.selectedAction;

    updateTelemetry(activeDecision, candidates.length, activePrompt);

    // Record into Nerd Telemetry & Live Diagnostic Stream with prominent %
    const m = activeTargetPlacement.metrics || {};
    telemetry.record({
      action: `${game.currentPiece?.type || 'P'}: ${activeTargetPlacement.label}`,
      confidence: activeDecision.confidence,
      latencyMs: activeDecision.latencyMs,
      ranking: activeDecision.ranking,
      details: `Lines:${m.completedLines || 0} Holes:${m.holes || 0} Bump:${m.bumpiness || 0}`,
    });
  }

  // 2. Advance current falling piece one step towards target placement
  const result = game.stepPieceTowards(activeTargetPlacement);

  // 3. When piece reaches destination row and locks
  if (result.locked) {
    activeTargetPlacement = null;
    updateTelemetry(activeDecision, 0);
  }

  // 4. Render updated board with falling piece and ghost outline
  renderBoard(activeTargetPlacement);
}

function playGame() {
  if (isRunning) return;
  isRunning = true;
  if (btnToggle) {
    btnToggle.innerHTML = `<span>⏸ Pause</span>`;
    btnToggle.classList.remove('btn-primary');
    btnToggle.classList.add('btn-secondary');
  }

  const intervalMs = Math.round(1000 / stepsPerSec);
  loopTimer = setInterval(stepSimulation, intervalMs);
}

function pauseGame() {
  if (!isRunning) return;
  isRunning = false;
  if (btnToggle) {
    btnToggle.innerHTML = `<span>▶ Play</span>`;
    btnToggle.classList.remove('btn-secondary');
    btnToggle.classList.add('btn-primary');
  }
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
}

function resetGame() {
  game.reset();
  activeTargetPlacement = null;
  activeDecision = null;
  renderBoard();
  telemetry.logSystem('Tetris matrix reset.');
  updateTelemetry(null, 0, 'Standing by for next tetromino...');
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
  stepsPerSec = parseInt(e.target.value, 10);
  if (speedLabel) speedLabel.textContent = `${stepsPerSec} steps/s`;
  if (isRunning) {
    pauseGame();
    playGame();
  }
});

// Boot-time setup
function init() {
  renderBoard();
  telemetry.logSystem('Tetris S1stem initialized (384 -> 128).');
  updateTelemetry(null, 0, 'Standing by to rank macro-placements...');
}

init();
