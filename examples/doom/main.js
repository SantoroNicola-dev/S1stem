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
const missionObjectiveText = document.getElementById('mission-objective-text');
const missionWaypointBadge = document.getElementById('mission-waypoint-badge');
const missionCompassBadge = document.getElementById('mission-compass-badge');
const missionDepthBadge = document.getElementById('mission-depth-badge');

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
let isGameSpawned = false;
let isProcessingDecision = false;
let decisionCount = 0;
let decisionTimer = null;
let lastActionTime = 0;
const WEAPON_COOLDOWN_MS = 750; // Shotgun/pistol mechanical refire cadence
let lastShotTime = 0;
let forwardStreak = 0;
let lastTurnDir = 'TURN_LEFT';

// Tactical Mission Director for E1M1 Progression
class TacticalMissionDirector {
  constructor() {
    this.startTime = 0;
    this.phaseIndex = 0;
    this.heading = 0; // 0 = East (Spawn), 90 = North (Zigzag entrance), 180 = West, 270 = South

    // E1M1 Progression Waypoints
    this.phases = [
      {
        id: 'CROSS_HANGAR',
        name: 'PHASE 1: CROSS HANGAR TO PILLARS',
        durationSec: 2.8,
        targetHeading: 0,   // Sprint East straight toward pillars/armor
        description: 'Advancing across hangar floor to retrieve green armor & health'
      },
      {
        id: 'PIVOT_ZIGZAG',
        name: 'PHASE 2: PIVOT & BREACH ZIGZAG HALL',
        durationSec: 4.5,
        targetHeading: 90,  // Turn North (left) to enter zigzag doorway
        description: 'Entering northern corridor overlooking outdoor courtyard'
      },
      {
        id: 'ZIGZAG_CORRIDOR',
        name: 'PHASE 3: NAVIGATE ZIGZAG WALKWAY',
        durationSec: 8.0,
        targetHeading: 45,  // Follow corridor North-East towards computer stairs
        description: 'Traversing zigzag bends with optical depth alignment'
      },
      {
        id: 'COMPUTER_STAIRS',
        name: 'PHASE 4: DESCEND TO COMPUTER ROOM',
        durationSec: 10.0,
        targetHeading: 90,  // Descend stairs into main computer facility
        description: 'Breaching computer room and neutralizing sector hostiles'
      },
      {
        id: 'SECURE_AND_EXIT',
        name: 'PHASE 5: TACTICAL SWEEP & EXIT PATH',
        durationSec: 35.0,
        targetHeading: 0,   // Forward across bridge towards exit chamber
        description: 'Securing walkway across toxic pool and reaching exit switch'
      }
    ];
  }

  reset() {
    this.startTime = performance.now();
    this.phaseIndex = 0;
    this.heading = 0;
  }

  getCurrentPhase(perception) {
    if (!this.startTime) this.startTime = performance.now();
    const elapsedSec = (performance.now() - this.startTime) / 1000;

    // Smart early phase transition: if Phase 1 has run for > 2.0s and wall/pillar is near, pivot immediately!
    if (this.phaseIndex === 0 && elapsedSec > 2.0 && perception?.wallProximity > 0.55) {
      this.phaseIndex = 1;
      return this.phases[1];
    }

    let accum = 0;
    for (let i = 0; i < this.phases.length; i++) {
      accum += this.phases[i].durationSec;
      if (elapsedSec < accum || i === this.phases.length - 1) {
        this.phaseIndex = i;
        return this.phases[i];
      }
    }
    return this.phases[this.phases.length - 1];
  }

  registerAction(actionId) {
    if (actionId === 'TURN_LEFT') {
      this.heading = (this.heading + 35) % 360;
    } else if (actionId === 'TURN_RIGHT') {
      this.heading = (this.heading - 35 + 360) % 360;
    }
  }

  getHeadingDiff(targetDeg) {
    let diff = (targetDeg - this.heading) % 360;
    if (diff > 180) diff -= 360;
    if (diff < -180) diff += 360;
    return diff; // > 0 means target is to the left; < 0 means to the right
  }

  formatCompass() {
    const h = Math.round((this.heading % 360 + 360) % 360);
    const cardinal = (h >= 315 || h < 45) ? 'E' : (h >= 45 && h < 135) ? 'N' : (h >= 135 && h < 225) ? 'W' : 'S';
    return `${String(h).padStart(3, '0')}° ${cardinal}`;
  }
}

const missionDirector = new TacticalMissionDirector();

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

// Map chosen action to Doom key operations with tuned pulse durations
function executeReflexAction(action) {
  const actId = typeof action === 'string' ? action : (action?.id || action?.label || '');
  const movePulseMs = 180;    // Assertive running stride
  const strafePulseMs = 160;  // Agile lateral dodge
  const turnPulseMs = 125;    // Smooth ~30-35° rotation for fluid cornering
  const firePulseMs = 80;     // Crisp trigger pull
  const usePulseMs = 120;     // Switch / door activation

  missionDirector.registerAction(actId);

  switch (actId) {
    case 'FORWARD':
      doom.pulseKey('UP', movePulseMs);
      forwardStreak++;
      break;
    case 'TURN_LEFT':
      doom.pulseKey('LEFT', turnPulseMs);
      lastTurnDir = 'TURN_LEFT';
      forwardStreak = 0;
      break;
    case 'TURN_RIGHT':
      doom.pulseKey('RIGHT', turnPulseMs);
      lastTurnDir = 'TURN_RIGHT';
      forwardStreak = 0;
      break;
    case 'STRAFE_LEFT':
      doom.pulseKey('STRAFE_L', strafePulseMs);
      break;
    case 'STRAFE_RIGHT':
      doom.pulseKey('STRAFE_R', strafePulseMs);
      break;
    case 'FIRE':
      doom.pulseKey('FIRE', firePulseMs);
      lastShotTime = performance.now();
      break;
    case 'STRAFE_FIRE_L':
      doom.pulseKey('STRAFE_L', strafePulseMs);
      doom.pulseKey('FIRE', firePulseMs);
      lastShotTime = performance.now();
      break;
    case 'STRAFE_FIRE_R':
      doom.pulseKey('STRAFE_R', strafePulseMs);
      doom.pulseKey('FIRE', firePulseMs);
      lastShotTime = performance.now();
      break;
    case 'ADVANCE_FIRE':
      doom.pulseKey('UP', movePulseMs);
      doom.pulseKey('FIRE', firePulseMs);
      lastShotTime = performance.now();
      forwardStreak++;
      break;
    case 'USE':
      doom.pulseKey('USE', usePulseMs);
      break;
  }
}

// Tactical Perceptual State Formatter
function buildTacticalPrompt(perception, phase) {
  const threatC = perception.sectorThreat.center.toFixed(2);
  const threatL = perception.sectorThreat.left.toFixed(2);
  const threatR = perception.sectorThreat.right.toFixed(2);
  const target = perception.centerTargetScore > 0.20 ? 'ACQUIRED' : 'SCANNING';
  const corridor = (perception.corridorDepth?.center ?? 0.5) > 0.35 ? 'OPEN' : 'SHADOW';
  const phaseId = phase?.id || 'COMBAT_ADVANCE';

  return `[DOOM_TACTICAL] TARGET: ${target} | THREAT_C: ${threatC} | THREAT_L: ${threatL} | THREAT_R: ${threatR} | CORRIDOR: ${corridor} | FOCUS: ${phaseId}`;
}

// System 1 Reflex Cycle
async function triggerDecisionCycle() {
  if (!isGameSpawned || !isAiActive || !doom.isRunning || isProcessingDecision) return;

  const now = performance.now();
  if (now - lastActionTime < 110) return; // ~9 decisions per second (optimal reaction tempo)
  lastActionTime = now;
  isProcessingDecision = true;

  try {
    const p = doom.perception;
    const phase = missionDirector.getCurrentPhase(p);
    const statePrompt = buildTacticalPrompt(p, phase);
    const isGunReady = (now - lastShotTime >= WEAPON_COOLDOWN_MS);
    const headingDiff = missionDirector.getHeadingDiff(phase.targetHeading);
    const depthC = p.corridorDepth?.center ?? 0.5;
    const depthL = p.corridorDepth?.left ?? 0.5;
    const depthR = p.corridorDepth?.right ?? 0.5;

    let tacticalStance = 'ADVANCE';
    const biases = {};
    const actions = DOOM_WEIGHTS.actions;

    if (p.centerTargetScore > 0.20) {
      // --- PRIORITY 1: COMBAT LOCK (Hostile in crosshairs) ---
      tacticalStance = 'COMBAT_LOCK';
      if (isGunReady) {
        biases['ADVANCE_FIRE'] = 3.0 + p.centerTargetScore * 1.5;
        biases['FIRE'] = 2.6 + p.centerTargetScore * 1.2;
        biases['FORWARD'] = 1.0;
      } else {
        // Weapon cycling/reloading: Circle-strafe & maneuver evasively
        biases['FIRE'] = -6.0;
        biases['ADVANCE_FIRE'] = -5.0;
        biases['STRAFE_FIRE_L'] = -5.0;
        biases['STRAFE_FIRE_R'] = -5.0;
        biases['STRAFE_LEFT'] = 2.4;
        biases['STRAFE_RIGHT'] = 2.4;
        biases['FORWARD'] = 1.2;
      }
    } else if (p.sectorThreat.left > 0.28 || p.sectorThreat.right > 0.28) {
      // --- PRIORITY 2: FLANK THREAT (Lateral danger detected) ---
      tacticalStance = 'FLANK_EVADE';
      biases['FIRE'] = -6.0;
      biases['ADVANCE_FIRE'] = -4.0;
      biases['FORWARD'] = -3.0; // Stop running forward into crossfire
      if (p.sectorThreat.left > 0.28) {
        biases['STRAFE_RIGHT'] = 2.4;
        biases['TURN_LEFT'] = 3.4;
      }
      if (p.sectorThreat.right > 0.28) {
        biases['STRAFE_LEFT'] = 2.4;
        biases['TURN_RIGHT'] = 3.4;
      }
    } else if (p.isStuck || (p.wallProximity > 0.65 && depthC < 0.25 && forwardStreak > 3)) {
      // --- PRIORITY 3: OBSTACLE / DOOR UNSTICK ---
      tacticalStance = 'UNSTICK_BREACH';
      // Suppress forward into wall!
      biases['FORWARD'] = -5.0;
      biases['FIRE'] = -6.0;
      biases['ADVANCE_FIRE'] = -5.0;
      // Pulse USE to open any door/switch
      biases['USE'] = 2.5;

      // Decisively steer toward the open hallway
      if (depthL > depthR + 0.05) {
        biases['TURN_LEFT'] = 3.6;
        biases['STRAFE_LEFT'] = 2.0;
      } else if (depthR > depthL + 0.05) {
        biases['TURN_RIGHT'] = 3.6;
        biases['STRAFE_RIGHT'] = 2.0;
      } else {
        biases['TURN_LEFT'] = 3.6;
      }
    } else {
      // --- PRIORITY 4: GOAL-DIRECTED MISSION NAVIGATION ---
      tacticalStance = 'PATROL_ADVANCE';
      biases['FIRE'] = -6.0;
      biases['ADVANCE_FIRE'] = -4.0;
      biases['STRAFE_FIRE_L'] = -5.0;
      biases['STRAFE_FIRE_R'] = -5.0;

      // Check alignment with target heading
      if (Math.abs(headingDiff) > 25) {
        // Must rotate to align! Suppress forward so we actually turn!
        biases['FORWARD'] = -4.0;
        if (headingDiff > 0) {
          biases['TURN_LEFT'] = 3.6;
        } else {
          biases['TURN_RIGHT'] = 3.6;
        }
      } else {
        // Aligned: sprint forward!
        biases['FORWARD'] = 3.0;
        biases['TURN_LEFT'] = -2.0;
        biases['TURN_RIGHT'] = -2.0;

        // Fine corridor centering using visual depth
        const corridorDelta = depthR - depthL;
        if (corridorDelta > 0.15) {
          biases['TURN_RIGHT'] = 1.8;
          biases['FORWARD'] = 1.5;
        } else if (corridorDelta < -0.15) {
          biases['TURN_LEFT'] = 1.8;
          biases['FORWARD'] = 1.5;
        }

        // Periodic door / switch interaction pulse
        if (Math.random() < 0.04) {
          biases['USE'] = 1.8;
        }
      }
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

    // Update Mission Directive HUD
    if (missionObjectiveText) missionObjectiveText.textContent = phase.name;
    if (missionWaypointBadge) missionWaypointBadge.textContent = `WP ${missionDirector.phaseIndex + 1}/${missionDirector.phases.length}`;
    if (missionCompassBadge) missionCompassBadge.textContent = missionDirector.formatCompass();
    if (missionDepthBadge) {
      missionDepthBadge.textContent = `OPEN ${(depthC * 100).toFixed(0)}%`;
      missionDepthBadge.className = `badge font-mono ${depthC > 0.35 ? 'badge-green' : 'badge-amber'}`;
    }

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
    const targetStatus = p.centerTargetScore > 0.20 ? 'TARGET_LOCK' : 'SCANNING';
    telemetry.record({
      action: chosenActionId,
      confidence: choice.confidence,
      latencyMs: choice.latencyMs,
      ranking: choice.ranking,
      details: `${tacticalStance} | ${missionDirector.formatCompass()} | C:${(p.centerTargetScore * 100).toFixed(0)}%`,
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
    isGameSpawned = false;
    autoStartBtn.disabled = true;
    autoStartBtn.textContent = 'LAUNCHING...';
    await doom.autoStartGame();
    missionDirector.reset();
    isGameSpawned = true;
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
    isGameSpawned = false;
    resetBtn.disabled = true;
    resetBtn.textContent = 'RESTARTING...';
    telemetry.logSystem('Session restart initiated.');
    await doom.restartGame();
    missionDirector.reset();
    isGameSpawned = true;
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

    // Auto-boot knee deep in the dead after engine settles
    autoStartBtn.textContent = 'LAUNCHING...';
    setTimeout(() => {
      doom.autoStartGame().then(() => {
        missionDirector.reset();
        isGameSpawned = true;
        autoStartBtn.textContent = 'E1M1 ACTIVE';
        autoStartBtn.classList.add('active');
      }).catch((err) => {
        console.error('Doom auto start error:', err);
        autoStartBtn.textContent = 'LAUNCH E1M1';
      });
    }, 1200);
  } catch (err) {
    statusBadge.textContent = 'FAILED';
    statusBadge.className = 'badge badge-red';
    telemetry.logSystem(`Initialization error: ${err.message}`, 'error');
  }
}

boot();
