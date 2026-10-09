import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateDeterministicEmbedding } from '../core/s1stem.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// AdamW projection head trainer
class PureAdamWTrainer {
  constructor(inDim = 384, outDim = 128, seed = 42) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.W = new Float32Array(inDim * outDim);
    this.M = new Float32Array(inDim * outDim);
    this.V = new Float32Array(inDim * outDim);
    this.gradAcc = new Float32Array(inDim * outDim);

    let s = seed;
    const scale = Math.sqrt(2.0 / (inDim + outDim));
    for (let i = 0; i < this.W.length; i++) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      const u = s / 4294967296;
      this.W[i] = (u * 2 - 1) * scale;
    }
  }

  forward(input) {
    const u = new Float32Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) {
      let sum = 0;
      for (let i = 0; i < this.inDim; i++) {
        sum += input[i] * this.W[i * this.outDim + j];
      }
      u[j] = sum;
    }

    let sumSq = 0;
    for (let j = 0; j < this.outDim; j++) sumSq += u[j] * u[j];
    const norm = Math.sqrt(sumSq) || 1e-9;
    const v = new Float32Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) v[j] = u[j] / norm;

    return { u, v, norm };
  }

  accumulateGrad(input, u, v, norm, gradV) {
    let dotGradV = 0;
    for (let j = 0; j < this.outDim; j++) dotGradV += gradV[j] * v[j];

    const gradU = new Float32Array(this.outDim);
    for (let j = 0; j < this.outDim; j++) {
      gradU[j] = (gradV[j] - dotGradV * v[j]) / norm;
    }

    for (let i = 0; i < this.inDim; i++) {
      const inVal = input[i];
      const offset = i * this.outDim;
      for (let j = 0; j < this.outDim; j++) {
        this.gradAcc[offset + j] += inVal * gradU[j];
      }
    }
  }

  step(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, weightDecay = 1e-4, step = 1, gradScale = 1.0) {
    const beta1Correction = 1.0 - Math.pow(beta1, step);
    const beta2Correction = 1.0 - Math.pow(beta2, step);

    for (let i = 0; i < this.W.length; i++) {
      const g = this.gradAcc[i] * gradScale;
      this.W[i] -= lr * weightDecay * this.W[i];
      this.M[i] = beta1 * this.M[i] + (1.0 - beta1) * g;
      this.V[i] = beta2 * this.V[i] + (1.0 - beta2) * (g * g);

      const mHat = this.M[i] / beta1Correction;
      const vHat = this.V[i] / beta2Correction;

      this.W[i] -= (lr * mHat) / (Math.sqrt(vHat) + 1e-8);
      this.gradAcc[i] = 0;
    }
  }
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Train InfoNCE Two-Tower pair
function trainTwoTower(dataset, inDim = 384, outDim = 128, epochs = 25, seedS = 101, seedA = 202, lr = 0.005) {
  const headS = new PureAdamWTrainer(inDim, outDim, seedS);
  const headA = new PureAdamWTrainer(inDim, outDim, seedA);
  const tau = 0.07;
  let step = 0;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (const sample of dataset) {
      step++;
      const sRaw = generateDeterministicEmbedding(sample.context, inDim);
      const posRaw = generateDeterministicEmbedding(sample.positive, inDim);
      const negRaws = sample.negatives.map(t => generateDeterministicEmbedding(t, inDim));

      const fS = headS.forward(sRaw);
      const fPos = headA.forward(posRaw);
      const fNegs = negRaws.map(r => headA.forward(r));

      const candidates = [fPos, ...fNegs];
      const dots = candidates.map(c => dot(fS.v, c.v));
      const logits = dots.map(d => d / tau);

      const maxLogit = Math.max(...logits);
      const exps = logits.map(l => Math.exp(l - maxLogit));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / sumExp);

      // Loss: -log(probs[0])
      totalLoss += -Math.log(Math.max(1e-9, probs[0]));

      // Gradient backprop
      // For state head:
      const gradVS = new Float32Array(outDim);
      for (let k = 0; k < candidates.length; k++) {
        const yk = k === 0 ? 1.0 : 0.0;
        const gk = (probs[k] - yk) / tau;
        for (let j = 0; j < outDim; j++) {
          gradVS[j] += gk * candidates[k].v[j];
        }
      }
      headS.accumulateGrad(sRaw, fS.u, fS.v, fS.norm, gradVS);

      // For positive action:
      const gPos = (probs[0] - 1.0) / tau;
      const gradVPos = new Float32Array(outDim);
      for (let j = 0; j < outDim; j++) gradVPos[j] = gPos * fS.v[j];
      headA.accumulateGrad(posRaw, fPos.u, fPos.v, fPos.norm, gradVPos);

      // For negative actions:
      for (let k = 1; k < candidates.length; k++) {
        const gk = probs[k] / tau;
        const gradVNeg = new Float32Array(outDim);
        for (let j = 0; j < outDim; j++) gradVNeg[j] = gk * fS.v[j];
        headA.accumulateGrad(negRaws[k - 1], fNegs[k - 1].u, fNegs[k - 1].v, fNegs[k - 1].norm, gradVNeg);
      }

      headS.step(lr, 0.9, 0.999, 1e-4, step, 1.0);
      headA.step(lr, 0.9, 0.999, 1e-4, step, 1.0);
    }
  }

  return { headS, headA };
}

// -------------------------------------------------------------
// 1. SNAKE DATASET
// -------------------------------------------------------------
console.log('Generating Snake dataset and training dedicated weights...');

const snakeDirections = [
  { id: 'MOVE_UP', label: 'Move UP (North). Advance north towards target food. Best when food is north and corridor has open space.' },
  { id: 'MOVE_RIGHT', label: 'Move RIGHT (East). Advance east towards target food. Best when food is east and corridor has open space.' },
  { id: 'MOVE_DOWN', label: 'Move DOWN (South). Advance south towards target food. Best when food is south and corridor has open space.' },
  { id: 'MOVE_LEFT', label: 'Move LEFT (West). Advance west towards target food. Best when food is west and corridor has open space.' },
];

const snakeDataset = [];

// Directional cues
const dirs = [
  { dir: 'UP', label: 'North', action: snakeDirections[0] },
  { dir: 'RIGHT', label: 'East', action: snakeDirections[1] },
  { dir: 'DOWN', label: 'South', action: snakeDirections[2] },
  { dir: 'LEFT', label: 'West', action: snakeDirections[3] },
];

dirs.forEach((d, idx) => {
  const negs = snakeDirections.filter((_, i) => i !== idx).map(a => a.label);

  for (let dist = 1; dist <= 12; dist++) {
    snakeDataset.push({
      context: `HIERARCHICAL GOALS: 1. Survive 2. Prevent Traps 3. Collect Target 4. Minimal Steps. Target offset: ${d.label} (+${dist}). OPTIMAL GEODESIC MOVE: '${d.dir}' provides shortest path of ${dist} steps to food without detour.`,
      positive: d.action.label,
      negatives: negs,
    });
    snakeDataset.push({
      context: `STATE: Target food is located directly to the ${d.label} at distance ${dist}. Safe open corridor heading ${d.label}. DECISION: Choose direction ${d.dir}.`,
      positive: d.action.label,
      negatives: negs,
    });
  }

  // Safety cues: collision in one direction, safe in target
  const otherDirs = dirs.filter((_, i) => i !== idx);
  otherDirs.forEach(od => {
    snakeDataset.push({
      context: `WARNING: Dangerous collision imminent to the ${od.label} (${od.dir} is blocked). Path is clear and safe to the ${d.label}. Best move is ${d.dir}.`,
      positive: d.action.label,
      negatives: negs,
    });
  });
});

const snakeModels = trainTwoTower(snakeDataset, 384, 128, 30, 4201, 4202, 0.008);
console.log('✓ Snake specialized model trained.');

// -------------------------------------------------------------
// 2. TETRIS DATASET
// -------------------------------------------------------------
console.log('Generating Tetris dataset and training dedicated weights...');

const tetrisDataset = [];
const pieceTypes = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];

pieceTypes.forEach(p => {
  for (let col = 0; col < 10; col++) {
    const posAction = `Col ${col}, Rot 0°. Clears 2 lines! Creates ZERO holes (perfect). Max height 6/20, bumpiness 2.`;
    const neg1 = `Col ${col}, Rot 90°. No lines cleared. Produces 3 buried holes. Max height 12/20, bumpiness 8.`;
    const neg2 = `Col ${col}, Rot 180°. No lines cleared. Produces 2 buried holes. Max height 14/20, bumpiness 6.`;
    const neg3 = `Col ${col}, Rot 270°. No lines cleared. Produces 4 buried holes. Max height 16/20, bumpiness 10.`;

    tetrisDataset.push({
      context: `TETRIS GOAL: 1. Maximize Completed Lines. 2. Strictly Minimize Holes. Piece=${p}. MISSION: Select candidate with maximum line clears and zero buried holes. Optimal column: ${col}.`,
      positive: posAction,
      negatives: [neg1, neg2, neg3],
    });

    const tetrisClear = `Col ${col}, Rot 90°. Clears 4 lines! Creates ZERO holes (perfect). Max height 4/20, bumpiness 1.`;
    tetrisDataset.push({
      context: `TETRIS OPPORTUNITY: Tetris 4-row line clear available at column ${col}! Clean board drop.`,
      positive: tetrisClear,
      negatives: [neg1, neg2, neg3],
    });
  }
});

const tetrisModels = trainTwoTower(tetrisDataset, 384, 128, 30, 8401, 8402, 0.008);
console.log('✓ Tetris specialized model trained.');

// -------------------------------------------------------------
// 3. DOOM TACTICAL COMBAT DATASET
// -------------------------------------------------------------
console.log('Generating Doom tactical combat dataset and training dedicated weights...');

const doomActions = [
  { id: 'FORWARD', label: 'Advance Forward', desc: 'Advance forward through open corridor towards the exit or objective. Best when corridor ahead is clear.' },
  { id: 'TURN_LEFT', label: 'Turn Left', desc: 'Rotate camera left counter-clockwise. Best when threats detected on left flank or scanning dark corridors.' },
  { id: 'TURN_RIGHT', label: 'Turn Right', desc: 'Rotate camera right clockwise. Best when threats detected on right flank or scanning dark corridors.' },
  { id: 'STRAFE_LEFT', label: 'Strafe Left', desc: 'Evasive lateral sidestep left to dodge incoming imp fireballs while maintaining center view.' },
  { id: 'STRAFE_RIGHT', label: 'Strafe Right', desc: 'Evasive lateral sidestep right to dodge incoming imp fireballs while maintaining center view.' },
  { id: 'FIRE', label: 'Fire Weapon', desc: 'Discharge shotgun or pistol directly at hostile demon target locked in center crosshairs.' },
  { id: 'STRAFE_FIRE_L', label: 'Strafe Left & Fire', desc: 'Tactical strafe maneuver to the left while delivering continuous suppressing fire at hostiles.' },
  { id: 'STRAFE_FIRE_R', label: 'Strafe Right & Fire', desc: 'Tactical strafe maneuver to the right while delivering continuous suppressing fire at hostiles.' },
  { id: 'ADVANCE_FIRE', label: 'Advance & Fire', desc: 'Aggressive forward assault firing weapon directly into target demons ahead.' },
  { id: 'USE', label: 'Use / Interact', desc: 'Press use key to open locked doors, activate switches, or ride elevators in E1M1.' },
];

const doomDataset = [];

function fullActionText(a) {
  return `${a.label}. ${a.desc}`;
}

const allDoomTexts = doomActions.map(fullActionText);

function doomSample(context, posId, negIds) {
  const posAction = doomActions.find(a => a.id === posId);
  const posText = fullActionText(posAction);
  const negTexts = negIds 
    ? negIds.map(id => fullActionText(doomActions.find(a => a.id === id)))
    : allDoomTexts.filter(t => t !== posText);
  doomDataset.push({ context, positive: posText, negatives: negTexts });
}

// 1. Target locked in crosshair -> FIRE / ADVANCE_FIRE
for (let tc = 0.35; tc <= 1.0; tc += 0.05) {
  const tcStr = tc.toFixed(2);
  doomSample(
    `[DOOM_TACTICAL] TARGET: ACQUIRED | THREAT_C: ${tcStr} | THREAT_L: 0.05 | THREAT_R: 0.05 | CORRIDOR: OPEN | FOCUS: COMBAT_ADVANCE`,
    'FIRE',
    ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'USE', 'STRAFE_LEFT', 'STRAFE_RIGHT']
  );
  doomSample(
    `[DOOM_TACTICAL] TARGET: ACQUIRED | THREAT_C: ${tcStr} | THREAT_L: 0.10 | THREAT_R: 0.10 | CORRIDOR: OPEN | FOCUS: CLOSE_RANGE_ASSAULT`,
    'ADVANCE_FIRE',
    ['TURN_LEFT', 'TURN_RIGHT', 'USE', 'STRAFE_LEFT', 'STRAFE_RIGHT']
  );
  doomSample(
    `DOOM REFLEX: Hostile demon locked dead center in crosshairs with threat ${tcStr}. High priority target engagement.`,
    'FIRE',
    ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'USE']
  );
}

// 2. Lateral Threat: Left Flank -> STRAFE_FIRE_R or TURN_LEFT
for (let tl = 0.35; tl <= 1.0; tl += 0.08) {
  const tlStr = tl.toFixed(2);
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.05 | THREAT_L: ${tlStr} | THREAT_R: 0.00 | CORRIDOR: OPEN | FOCUS: COMBAT_ADVANCE`,
    'STRAFE_FIRE_R',
    ['STRAFE_FIRE_L', 'FORWARD', 'USE', 'STRAFE_LEFT']
  );
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.10 | THREAT_L: ${tlStr} | THREAT_R: 0.00 | CORRIDOR: SHADOW | FOCUS: FLANK_HOSTILE`,
    'TURN_LEFT',
    ['TURN_RIGHT', 'FORWARD', 'USE', 'STRAFE_RIGHT']
  );
  doomSample(
    `COMBAT RADAR: Red fireball incoming from the left flank (${tlStr}). Evasive sidestep right and return fire.`,
    'STRAFE_FIRE_R',
    ['STRAFE_FIRE_L', 'FORWARD', 'USE']
  );
}

// 3. Lateral Threat: Right Flank -> STRAFE_FIRE_L or TURN_RIGHT
for (let tr = 0.35; tr <= 1.0; tr += 0.08) {
  const trStr = tr.toFixed(2);
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.05 | THREAT_L: 0.00 | THREAT_R: ${trStr} | CORRIDOR: OPEN | FOCUS: COMBAT_ADVANCE`,
    'STRAFE_FIRE_L',
    ['STRAFE_FIRE_R', 'FORWARD', 'USE', 'STRAFE_RIGHT']
  );
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.10 | THREAT_L: 0.00 | THREAT_R: ${trStr} | CORRIDOR: SHADOW | FOCUS: FLANK_HOSTILE`,
    'TURN_RIGHT',
    ['TURN_LEFT', 'FORWARD', 'USE', 'STRAFE_LEFT']
  );
  doomSample(
    `COMBAT RADAR: Enemy demon flanking from the right (${trStr}). Evasive sidestep left with suppressing fire.`,
    'STRAFE_FIRE_L',
    ['STRAFE_FIRE_R', 'FORWARD', 'USE']
  );
}

// 4. Open corridor navigation -> FORWARD
for (let i = 0; i < 15; i++) {
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.00 | THREAT_L: 0.00 | THREAT_R: 0.00 | CORRIDOR: OPEN | FOCUS: COMBAT_ADVANCE`,
    'FORWARD',
    ['FIRE', 'ADVANCE_FIRE', 'USE', 'STRAFE_FIRE_L', 'STRAFE_FIRE_R']
  );
  doomSample(
    `TACTICAL TELEMETRY: Clear pathway ahead in E1M1 Hangar. No hostiles detected. Advance forward towards objective.`,
    'FORWARD',
    ['FIRE', 'ADVANCE_FIRE', 'USE', 'STRAFE_FIRE_L', 'STRAFE_FIRE_R']
  );
}

// 5. Dark corridor / Room scanning -> TURN_LEFT / TURN_RIGHT
for (let i = 0; i < 10; i++) {
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.00 | THREAT_L: 0.00 | THREAT_R: 0.00 | CORRIDOR: SHADOW | FOCUS: SCAN_ROOM`,
    i % 2 === 0 ? 'TURN_LEFT' : 'TURN_RIGHT',
    ['FIRE', 'ADVANCE_FIRE', 'USE']
  );
  doomSample(
    `TACTICAL SCAN: Corner navigation in shadow sector. Rotate view to search for demon ambushes.`,
    i % 2 === 0 ? 'TURN_LEFT' : 'TURN_RIGHT',
    ['FIRE', 'ADVANCE_FIRE', 'USE']
  );
}

// 6. Interactive Door / Switch -> USE
for (let i = 0; i < 12; i++) {
  doomSample(
    `[DOOM_TACTICAL] TARGET: SCANNING | THREAT_C: 0.00 | THREAT_L: 0.00 | THREAT_R: 0.00 | CORRIDOR: DOOR_BLOCKED | FOCUS: INTERACT_USE`,
    'USE',
    ['FIRE', 'TURN_LEFT', 'TURN_RIGHT', 'STRAFE_LEFT', 'STRAFE_RIGHT']
  );
  doomSample(
    `MISSION INTERACTION: Airload hangar door closed ahead. Flip switch or press use key to operate mechanism.`,
    'USE',
    ['FIRE', 'TURN_LEFT', 'TURN_RIGHT', 'STRAFE_LEFT', 'STRAFE_RIGHT']
  );
}

const doomModels = trainTwoTower(doomDataset, 384, 128, 30, 9991, 9992, 0.008);
console.log('✓ Doom specialized model trained.');

// Export weights to JS files
function exportWeights(filename, headS, headA, name, exportVarName, extraMeta = '', suffix = '') {
  const content = `/**
 * ${name} Specialized Two-Tower Projection Weights
 * Trained with InfoNCE & AdamW for dedicated domain alignment.
 */
export const ${exportVarName} = {
  name: "${name}",
  inDim: ${headS.inDim},
  outDim: ${headS.outDim},${extraMeta}
  stateWeights: [${Array.from(headS.W).join(',')}],
  actionWeights: [${Array.from(headA.W).join(',')}]
};
${suffix}export default ${exportVarName};
`;
  fs.writeFileSync(filename, content);
  console.log(`Saved: ${filename}`);
}

exportWeights(
  path.join(__dirname, '../examples/snake/snake-weights.js'),
  snakeModels.headS,
  snakeModels.headA,
  'SnakeDomain',
  'SNAKE_WEIGHTS'
);

exportWeights(
  path.join(__dirname, '../examples/tetris/tetris-weights.js'),
  tetrisModels.headS,
  tetrisModels.headA,
  'TetrisDomain',
  'TETRIS_WEIGHTS'
);

const doomMeta = `
  modelName: "s1stem-doom-tactical-v1",
  embeddingDim: 384,
  latentDim: 128,
  actions: [
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STRAFE_LEFT",
    "STRAFE_RIGHT",
    "FIRE",
    "STRAFE_FIRE_L",
    "STRAFE_FIRE_R",
    "ADVANCE_FIRE",
    "USE"
  ],`;

const doomActionsExport = `
export const DOOM_ACTIONS = [
  { id: 'FORWARD', label: 'Advance Forward', description: 'Advance forward through open corridor towards the exit or objective. Best when corridor ahead is clear.' },
  { id: 'TURN_LEFT', label: 'Turn Left', description: 'Rotate camera left counter-clockwise. Best when threats detected on left flank or scanning dark corridors.' },
  { id: 'TURN_RIGHT', label: 'Turn Right', description: 'Rotate camera right clockwise. Best when threats detected on right flank or scanning dark corridors.' },
  { id: 'STRAFE_LEFT', label: 'Strafe Left', description: 'Evasive lateral sidestep left to dodge incoming imp fireballs while maintaining center view.' },
  { id: 'STRAFE_RIGHT', label: 'Strafe Right', description: 'Evasive lateral sidestep right to dodge incoming imp fireballs while maintaining center view.' },
  { id: 'FIRE', label: 'Fire Weapon', description: 'Discharge shotgun or pistol directly at hostile demon target locked in center crosshairs.' },
  { id: 'STRAFE_FIRE_L', label: 'Strafe Left & Fire', description: 'Tactical strafe maneuver to the left while delivering continuous suppressing fire at hostiles.' },
  { id: 'STRAFE_FIRE_R', label: 'Strafe Right & Fire', description: 'Tactical strafe maneuver to the right while delivering continuous suppressing fire at hostiles.' },
  { id: 'ADVANCE_FIRE', label: 'Advance & Fire', description: 'Aggressive forward assault firing weapon directly into target demons ahead.' },
  { id: 'USE', label: 'Use / Interact', description: 'Press use key to open locked doors, activate switches, or ride elevators in E1M1.' }
];

`;

exportWeights(
  path.join(__dirname, '../examples/doom/doom-weights.js'),
  doomModels.headS,
  doomModels.headA,
  'DoomTacticalDomain',
  'DOOM_WEIGHTS',
  doomMeta,
  doomActionsExport
);

console.log('✓ All weights (Snake, Tetris, Doom) successfully trained and exported!');


