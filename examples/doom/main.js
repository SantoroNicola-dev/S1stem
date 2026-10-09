import { S1stemEngine } from '../../core/s1stem.js';
import { DOOM_WEIGHTS, DOOM_ACTIONS } from './doom-weights.js';
import { DoomEngine } from './doom-engine.js';
import { NerdTelemetry } from '../../core/nerd-telemetry.js';

// DOM Element references
const canvas = document.getElementById('doom-canvas');
const autoStartBtn = document.getElementById('auto-start-btn');
const toggleAiBtn = document.getElementById('toggle-ai-btn');
const resetBtn = document.getElementById('reset-btn');
const crtToggleBtn = document.getElementById('crt-toggle-btn');
const statusBadge = document.getElementById('status-badge');
const aiStatusBadge = document.getElementById('ai-status-badge');
const fpsCounter = document.getElementById('fps-counter');
const latencyCounter = document.getElementById('latency-counter');
const decisionsCounter = document.getElementById('decisions-counter');
const promptDisplay = document.getElementById('prompt-display');
const lastDecision = document.getElementById('last-decision');
const threatLeftMeter = document.getElementById('threat-left');
const threatCenterMeter = document.getElementById('threat-center');
const threatRightMeter = document.getElementById('threat-right');
const radarCenterScore = document.getElementById('radar-center-score');
const actionDistribution = document.getElementById('action-distribution');
const crtOverlay = document.getElementById('crt-overlay');

// Instantiate Doom WASM Engine and S1stem Reflex Core
const doom = new DoomEngine(canvas);
const s1 = new S1stemEngine({
  weights: DOOM_WEIGHTS,
  temperature: 0.08,
});
const telemetry = new NerdTelemetry({
  domain: 'doom',
  temperature: 0.08,
  stateDim: 384,
  projDim: 128,
  streamId: 'log-stream',
  statsContainerId: 'nerd-stats-grid',
});

let isAiActive = true;
let isProcessingDecision = false;
let decisionCount = 0;
let decisionTimer = null;
let lastActionTime = 0;

doom.onInfo = (msg) => {
  if (msg.includes('player') || msg.includes('startskill') || msg.includes('[ENGINE]')) {
    telemetry.logSystem(msg, 'engine');
  }
};

doom.onError = (msg) => telemetry.logSystem(msg, 'error');

// Build UI Action Bars
function initActionBars() {
  actionDistribution.innerHTML = '';
  for (const act of DOOM_WEIGHTS.actions) {
    const row = document.createElement('div');
    row.className = 'action-row';
    row.id = `act-row-${act}`;

    const label = document.createElement('span');
    label.className = 'action-label';
    label.textContent = act;

    const track = document.createElement('div');
    track.className = 'action-track';

    const bar = document.createElement('div');
    bar.className = 'action-bar';
    bar.id = `act-bar-${act}`;

    const pct = document.createElement('span');
    pct.className = 'action-pct font-mono';
    pct.id = `act-pct-${act}`;
    pct.textContent = '0%';

    track.appendChild(bar);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pct);
    actionDistribution.appendChild(row);
  }
}

// Map chosen action to Doom key operations
function executeReflexAction(action) {
  const actId = typeof action === 'string' ? action : (action?.id || action?.label || '');
  const pulseMs = 140;

  switch (actId) {
    case 'FORWARD':
      doom.pulseKey('UP', pulseMs);
      break;
    case 'TURN_LEFT':
      doom.pulseKey('LEFT', 90);
      break;
    case 'TURN_RIGHT':
      doom.pulseKey('RIGHT', 90);
      break;
    case 'STRAFE_LEFT':
      doom.pulseKey('STRAFE_L', pulseMs);
      break;
    case 'STRAFE_RIGHT':
      doom.pulseKey('STRAFE_R', pulseMs);
      break;
    case 'FIRE':
      doom.pulseKey('FIRE', 180);
      break;
    case 'STRAFE_FIRE_L':
      doom.pulseKey('STRAFE_L', pulseMs);
      doom.pulseKey('FIRE', pulseMs);
      break;
    case 'STRAFE_FIRE_R':
      doom.pulseKey('STRAFE_R', pulseMs);
      doom.pulseKey('FIRE', pulseMs);
      break;
    case 'ADVANCE_FIRE':
      doom.pulseKey('UP', pulseMs);
      doom.pulseKey('FIRE', pulseMs);
      break;
    case 'USE':
      doom.pulseKey('USE', 100);
      break;
  }
}

// Tactical Perceptual State Formatter
function buildTacticalPrompt(perception) {
  const threatC = perception.sectorThreat.center.toFixed(2);
  const threatL = perception.sectorThreat.left.toFixed(2);
  const threatR = perception.sectorThreat.right.toFixed(2);
  const target = perception.centerTargetScore > 0.25 ? 'ACQUIRED' : 'SCANNING';
  const corridor = perception.averageLuminance > 0.35 ? 'OPEN' : 'SHADOW';

  return `[DOOM_TACTICAL] TARGET: ${target} | THREAT_C: ${threatC} | THREAT_L: ${threatL} | THREAT_R: ${threatR} | CORRIDOR: ${corridor} | FOCUS: COMBAT_ADVANCE`;
}

// System 1 Reflex Cycle
async function triggerDecisionCycle() {
  if (!isAiActive || !doom.isRunning || isProcessingDecision) return;

  const now = performance.now();
  if (now - lastActionTime < 110) return; // ~9 decisions per second (optimal reaction tempo)
  lastActionTime = now;
  isProcessingDecision = true;

  try {
    const p = doom.perception;
    const statePrompt = buildTacticalPrompt(p);

    // Dynamic Tactical Prior Biases based on real-time visual perception
    const biases = {};
    const actions = DOOM_WEIGHTS.actions;

    // Base exploration bias: keep moving forward unless engaged
    biases['FORWARD'] = 1.2;

    // Target in crosshairs -> heavy bias on aggressive fire
    if (p.centerTargetScore > 0.25) {
      biases['FIRE'] = 3.5 + p.centerTargetScore * 3.0;
      biases['ADVANCE_FIRE'] = 3.0 + p.centerTargetScore * 2.5;
    }

    // Lateral threats detected -> tactical strafe and flank
    if (p.sectorThreat.left > 0.3) {
      biases['STRAFE_FIRE_R'] = 2.0;
      biases['TURN_LEFT'] = 2.5;
    }
    if (p.sectorThreat.right > 0.3) {
      biases['STRAFE_FIRE_L'] = 2.0;
      biases['TURN_RIGHT'] = 2.5;
    }

    // Periodic room scanning if in dark corridor
    if (p.averageLuminance < 0.25) {
      biases['TURN_LEFT'] = 1.5;
      biases['TURN_RIGHT'] = 1.5;
    }

    // Occasional interaction pulse to open doors or flip switches
    if (Math.random() < 0.08) {
      biases['USE'] = 1.8;
    }

    // Query sub-millisecond edge projection model
    const choice = await s1.choice(statePrompt, { biases });
    decisionCount++;

    const chosenActionId = choice.winner || choice.selectedAction?.id || choice.selectedAction;

    // Execute tactical reflex
    executeReflexAction(chosenActionId);

    // Update Telemetry Display
    promptDisplay.textContent = statePrompt;
    lastDecision.textContent = `${chosenActionId} (${(choice.confidence * 100).toFixed(0)}%)`;
    latencyCounter.textContent = `${choice.latencyMs.toFixed(3)} ms`;
    decisionsCounter.textContent = decisionCount.toLocaleString();

    // Highlight winning action bar and probabilities
    for (const act of actions) {
      const prob = choice.distribution ? (choice.distribution[act] ?? 0) : 0;
      const bar = document.getElementById(`act-bar-${act}`);
      const pct = document.getElementById(`act-pct-${act}`);
      const row = document.getElementById(`act-row-${act}`);

      if (bar && pct && row) {
        bar.style.width = `${(prob * 100).toFixed(1)}%`;
        pct.textContent = `${(prob * 100).toFixed(0)}%`;
        if (act === chosenActionId) {
          row.classList.add('winner');
        } else {
          row.classList.remove('winner');
        }
      }
    }

    // Update Perception Radar UI
    threatLeftMeter.style.height = `${Math.min(100, p.sectorThreat.left * 100).toFixed(0)}%`;
    threatCenterMeter.style.height = `${Math.min(100, p.sectorThreat.center * 100).toFixed(0)}%`;
    threatRightMeter.style.height = `${Math.min(100, p.sectorThreat.right * 100).toFixed(0)}%`;
    radarCenterScore.textContent = `${(p.centerTargetScore * 100).toFixed(0)}%`;

    // Record into Nerd Telemetry & Live Diagnostic Stream with prominent %
    const targetStatus = p.centerTargetScore > 0.25 ? 'TARGET_LOCK' : 'SCANNING';
    telemetry.record({
      action: chosenActionId,
      confidence: choice.confidence,
      latencyMs: choice.latencyMs,
      ranking: choice.ranking,
      details: `${targetStatus} | C:${(p.centerTargetScore * 100).toFixed(0)}% L:${(p.sectorThreat.left * 100).toFixed(0)}% R:${(p.sectorThreat.right * 100).toFixed(0)}%`,
    });
  } catch (err) {
    console.error('Decision cycle error:', err);
  } finally {
    isProcessingDecision = false;
  }
}

// UI and Keyboard Event Handlers
function setupControls() {
  canvas.addEventListener('click', () => {
    canvas.focus();
  });

  autoStartBtn.addEventListener('click', async () => {
    autoStartBtn.disabled = true;
    autoStartBtn.textContent = 'LAUNCHING...';
    await doom.autoStartGame();
    autoStartBtn.disabled = false;
    autoStartBtn.textContent = 'E1M1 ACTIVE';
    autoStartBtn.classList.add('active');
  });

  toggleAiBtn.addEventListener('click', () => {
    isAiActive = !isAiActive;
    if (isAiActive) {
      toggleAiBtn.textContent = 'AI REFLEX: ON';
      toggleAiBtn.classList.remove('off');
      aiStatusBadge.textContent = 'ONLINE (SYSTEM 1)';
      aiStatusBadge.className = 'badge badge-green';
      telemetry.logSystem('AI Autopilot resumed.');
    } else {
      toggleAiBtn.textContent = 'AI REFLEX: OFF';
      toggleAiBtn.classList.add('off');
      aiStatusBadge.textContent = 'MANUAL OVERRIDE';
      aiStatusBadge.className = 'badge badge-amber';
      telemetry.logSystem('Manual keyboard override active (WASD / Arrows / Space / Ctrl).');
    }
  });

  crtToggleBtn.addEventListener('click', () => {
    crtOverlay.classList.toggle('hidden');
    crtToggleBtn.classList.toggle('active');
  });

  resetBtn.addEventListener('click', async () => {
    resetBtn.disabled = true;
    resetBtn.textContent = 'RESTARTING...';
    telemetry.logSystem('Session restart initiated.');
    await doom.restartGame();
    resetBtn.disabled = false;
    resetBtn.textContent = 'RESTART GAME';
    telemetry.logSystem('Game restarted in E1M1 Hangar.');
  });

  // Manual Keyboard Control Bridge (forward keys to WASM if user plays)
  const keyMap = {
    'ArrowUp': 'UP',
    'KeyW': 'UP',
    'ArrowDown': 'DOWN',
    'KeyS': 'DOWN',
    'ArrowLeft': 'LEFT',
    'KeyA': 'STRAFE_L',
    'ArrowRight': 'RIGHT',
    'KeyD': 'STRAFE_R',
    'ControlLeft': 'FIRE',
    'ControlRight': 'FIRE',
    'KeyF': 'FIRE',
    'Space': 'USE',
    'Enter': 'ENTER',
    'Escape': 'ESCAPE',
    'Tab': 'TAB',
  };

  window.addEventListener('keydown', (e) => {
    const act = keyMap[e.code];
    if (act) {
      doom.pressKey(act);
      if (!isAiActive) e.preventDefault();
    }
  });

  window.addEventListener('keyup', (e) => {
    const act = keyMap[e.code];
    if (act) {
      doom.releaseKey(act);
      if (!isAiActive) e.preventDefault();
    }
  });
}

// Master Loop
function startLoops() {
  decisionTimer = setInterval(triggerDecisionCycle, 100);

  setInterval(() => {
    if (fpsCounter) {
      fpsCounter.textContent = `${doom.fps} FPS`;
    }
  }, 500);
}

// Bootstrap
async function boot() {
  initActionBars();
  setupControls();
  telemetry.logSystem('Initializing Doom WebAssembly Core...');

  try {
    // 1. Pre-register candidate actions into hypersphere S^127
    telemetry.logSystem('Pre-computing S1stem tactical combat action representations...');
    await s1.registerActions(DOOM_ACTIONS);
    telemetry.logSystem(`Pre-computed and cached ${DOOM_ACTIONS.length} tactical combat action embeddings.`);

    // 2. Initialize WASM core
    await doom.init();
    doom.start();
    statusBadge.textContent = 'READY (35Hz)';
    statusBadge.className = 'badge badge-green';
    telemetry.logSystem('Doom 1.9 Shareware Core online and ticking at 35Hz!');
    startLoops();

    // Auto-boot knee deep in the dead after 1.2 seconds
    setTimeout(() => {
      doom.autoStartGame().then(() => {
        autoStartBtn.textContent = 'E1M1 ACTIVE';
        autoStartBtn.classList.add('active');
      });
    }, 1200);
  } catch (err) {
    statusBadge.textContent = 'FAILED';
    statusBadge.className = 'badge badge-red';
    telemetry.logSystem(`Initialization error: ${err.message}`, 'error');
  }
}

boot();
