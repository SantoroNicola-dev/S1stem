/**
 * S1stem vs Competitors Head-to-Head Extended Benchmark Suite
 * ============================================================
 * Benchmark comparing:
 *   1. S1stem v1.2.0 (BGE-Base 109M Dual-Encoder CLM)
 *   2. Microsoft-Decision-1 (official Azure Foundry SystemOne API)
 *   3. gpt-6.1-sol (Azure Foundry OpenAI chat completions API)
 * 
 * Features:
 *   • Strict separation between Held-Out Test Set and Contaminated Dev Set
 *   • Stratified sampling with fixed seed across 6 public datasets + 1 stress set
 *   • Wilson Score 95% Confidence Interval for Accuracy
 *   • Paired McNemar Test against Ground Truth
 *   • Expected Calibration Error (ECE) & Brier score on real Softmax probabilities
 *   • Measured Flip Rate across 8 invariant perturbations (all 3 models)
 *   • Scalability Limit Probing (K = 32 to 3,000)
 * 
 * Usage:
 *   node Benchmark/run_benchmark.js [--samples 50] [--seed 42]
 *   node Benchmark/run_benchmark.js --samples 3 (quick smoke run)
 * 
 * License: Apache-2.0
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import {
  loadBenchmarkConfig,
  S1stemAdapter,
  MicrosoftDecision1Adapter,
  Gpt6LunaDecisionsAdapter,
  Gpt61SolAdapter,
  buildEvaluationSuite,
  computeWilsonInterval,
  computeMcNemarTest,
  computeCalibration,
  analyzeLatencies,
  generatePerturbations,
  probeActionScalability
} from './common.js';

function parseCliArgs() {
  const args = process.argv.slice(2);
  let samplesPerSource = 50;
  let seed = 42;
  let runPerturbations = true;
  let runScalability = true;
  let includeDevSet = true;
  let s1stemOnly = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--samples' && args[i + 1]) {
      samplesPerSource = parseInt(args[i + 1], 10) || 50;
      i++;
    } else if (args[i] === '--seed' && args[i + 1]) {
      seed = parseInt(args[i + 1], 10) || 42;
      i++;
    } else if (args[i] === '--no-perturbations') {
      runPerturbations = false;
    } else if (args[i] === '--no-scalability') {
      runScalability = false;
    } else if (args[i] === '--no-dev-set') {
      includeDevSet = false;
    } else if (args[i] === '--s1stem-only') {
      s1stemOnly = true;
    }
  }

  return { samplesPerSource, seed, runPerturbations, runScalability, includeDevSet, s1stemOnly };
}

function cleanMatch(a, b) {
  if (!a || !b) return false;
  return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
}

async function main() {
  const args = parseCliArgs();
  const config = loadBenchmarkConfig();
  const af = config.azure_foundry;
  const oai = config.openai;

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultsDir = path.resolve('Benchmark/results', timestamp);
  fs.mkdirSync(resultsDir, { recursive: true });

  console.log('╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗');
  console.log('║   BENCHMARK: S1STEM v1.2.0 vs MICROSOFT-DECISION-1 vs GPT-6-LUNA (DECISIONS) vs GPT-6.1-SOL             ║');
  console.log('║   Evaluation Protocol: Stratified Public Test Sets + Held-Out Stress Set + Invariant Perturbations      ║');
  console.log('╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n');

  console.log(`Configured Samples/Source: ${args.samplesPerSource}`);
  console.log(`Random Seed:               ${args.seed}`);
  console.log(`Azure Foundry Configured:  ${af.is_configured ? 'YES (Live API Calls)' : 'NO'}`);
  console.log(`OpenAI API Configured:     ${oai.is_configured ? 'YES (Live Decisions Calls)' : 'NO'}`);
  console.log(`Results Output Directory:  ${resultsDir}\n`);

  // Initialize model adapters
  const s1stem = new S1stemAdapter(config.s1stem);
  await s1stem.init();

  let decision1 = null;
  let luna = null;
  let gpt6 = null;

  if (af.is_configured && !args.s1stemOnly) {
    decision1 = new MicrosoftDecision1Adapter({
      endpoint: af.endpoint,
      apiKey: af.api_key,
      deploymentName: af.microsoft_decision_1.deployment_name,
      timeoutMs: config.benchmark_settings.timeout_ms
    });
    gpt6 = new Gpt61SolAdapter({
      endpoint: af.endpoint,
      apiKey: af.api_key,
      deploymentName: af.gpt_6_1_sol.deployment_name,
      timeoutMs: config.benchmark_settings.timeout_ms
    });
  }

  const lunaConf = config.gpt_6_luna;
  if (lunaConf.is_configured && !args.s1stemOnly) {
    luna = new Gpt6LunaDecisionsAdapter({
      endpoint: lunaConf.endpoint,
      apiKey: lunaConf.api_key,
      model: lunaConf.deployment_name,
      path: lunaConf.path,
      timeoutMs: config.benchmark_settings.timeout_ms
    });
  }

  // Load datasets
  const { testSet, devSet } = buildEvaluationSuite({
    samplesPerSource: args.samplesPerSource,
    seed: args.seed,
    includeDevSet: args.includeDevSet
  });

  console.log(`Loaded ${testSet.length} items in Held-Out Test Set (Headline Benchmark).`);
  if (devSet.length > 0) {
    console.log(`Loaded ${devSet.length} items in Contaminated Dev Set (Reported separately for transparency).\n`);
  }

  const allItems = [...testSet, ...devSet];
  const detailedRecords = [];

  const rawStats = {
    test: {
      total: testSet.length,
      s1stem: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0 },
      decision1: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, active: !!decision1 },
      luna: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, contentFiltered: 0, active: !!luna },
      gpt6: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, contentFiltered: 0, active: !!gpt6 }
    },
    dev: {
      total: devSet.length,
      s1stem: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0 },
      decision1: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, active: !!decision1 },
      luna: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, contentFiltered: 0, active: !!luna },
      gpt6: { correct: 0, latencies: [], predictions: [], costUsd: 0, errors: 0, contentFiltered: 0, active: !!gpt6 }
    }
  };

  // ---------------------------------------------------------------------------
  // STEP 1: RUN SUITE EVALUATION
  // ---------------------------------------------------------------------------
  console.log('▶ [1/3] EVALUATING TEST & DEV DATASET ITEMS:\n');

  for (let i = 0; i < allItems.length; i++) {
    const item = allItems[i];
    const bucket = item.isDevSet ? rawStats.dev : rawStats.test;
    const itemPrefix = item.isDevSet ? '[DEV - Contaminated]' : '[TEST]';

    process.stdout.write(`[${i + 1}/${allItems.length}] ${itemPrefix} ${item.source} (${item.category}): `);

    const record = {
      index: i + 1,
      id: item.id,
      source: item.source,
      category: item.category,
      isDevSet: item.isDevSet,
      observation: item.observation,
      candidatesCount: item.candidates.length,
      candidates: item.candidates,
      expected: item.expected,
      models: {}
    };

    // 1. S1stem Evaluation
    try {
      const res = await s1stem.choice(item.observation, item.candidates);
      const isCorrect = cleanMatch(res.winner, item.expected);
      if (isCorrect) bucket.s1stem.correct++;
      bucket.s1stem.latencies.push(res.latencyMs);
      bucket.s1stem.predictions.push({ isCorrect, confidence: res.confidence });

      record.models.s1stem = {
        winner: res.winner,
        isCorrect,
        confidence: res.confidence,
        latencyMs: res.latencyMs,
        costUsd: 0.0
      };
      process.stdout.write(`S1:${isCorrect ? '✓' : '✗'}(${res.latencyMs}ms) `);
    } catch (err) {
      bucket.s1stem.errors++;
      bucket.s1stem.predictions.push({ isCorrect: false, confidence: 0 });
      record.models.s1stem = { error: err.message, isCorrect: false };
      process.stdout.write(`S1:ERR `);
    }

    // 2. Microsoft-Decision-1 Evaluation
    if (decision1) {
      try {
        const res = await decision1.choice(item.observation, item.candidates);
        if (res.isError) {
          bucket.decision1.errors++;
          bucket.decision1.predictions.push({ isCorrect: false, confidence: null });
          record.models.decision1 = { error: res.error, isCorrect: false, latencyMs: res.latencyMs };
          process.stdout.write(`D1:ERR `);
        } else {
          const isCorrect = cleanMatch(res.winner, item.expected);
          if (isCorrect) bucket.decision1.correct++;
          bucket.decision1.latencies.push(res.latencyMs);
          bucket.decision1.costUsd += res.costUsd;
          bucket.decision1.predictions.push({ isCorrect, confidence: res.confidence });

          record.models.decision1 = {
            winner: res.winner,
            isCorrect,
            confidence: res.confidence,
            latencyMs: res.latencyMs,
            costUsd: res.costUsd
          };
          process.stdout.write(`D1:${isCorrect ? '✓' : '✗'}(${res.latencyMs}ms) `);
        }
      } catch (err) {
        bucket.decision1.errors++;
        bucket.decision1.predictions.push({ isCorrect: false, confidence: null });
        record.models.decision1 = { error: err.message, isCorrect: false };
        process.stdout.write(`D1:EXC `);
      }
    }

    // 3. gpt-6-luna (Decisions API) Evaluation
    if (luna) {
      try {
        const res = await luna.choice(item.observation, item.candidates);
        if (res.isError) {
          bucket.luna.errors++;
          if (res.contentFiltered) bucket.luna.contentFiltered++;
          bucket.luna.predictions.push({ isCorrect: false, confidence: null });
          record.models.luna = { error: res.error, contentFiltered: !!res.contentFiltered, isCorrect: false, latencyMs: res.latencyMs };
          process.stdout.write(res.contentFiltered ? `LUNA:FILTER ` : `LUNA:ERR `);
        } else if (res.isRefusal) {
          bucket.luna.errors++;
          if (res.contentFiltered) bucket.luna.contentFiltered++;
          bucket.luna.predictions.push({ isCorrect: false, confidence: 0 });
          record.models.luna = { isRefusal: true, contentFiltered: true, isCorrect: false, latencyMs: res.latencyMs };
          process.stdout.write(`LUNA:REFUSAL `);
        } else {
          const isCorrect = cleanMatch(res.winner, item.expected);
          if (isCorrect) bucket.luna.correct++;
          bucket.luna.latencies.push(res.latencyMs);
          bucket.luna.costUsd += res.costUsd;
          bucket.luna.predictions.push({ isCorrect, confidence: res.confidence });

          record.models.luna = {
            winner: res.winner,
            isCorrect,
            confidence: res.confidence,
            latencyMs: res.latencyMs,
            costUsd: res.costUsd
          };
          process.stdout.write(`LUNA:${isCorrect ? '✓' : '✗'}(${res.latencyMs}ms) `);
        }
      } catch (err) {
        bucket.luna.errors++;
        bucket.luna.predictions.push({ isCorrect: false, confidence: null });
        record.models.luna = { error: err.message, isCorrect: false };
        process.stdout.write(`LUNA:EXC `);
      }
    }

    // 4. gpt-6.1-sol Evaluation
    if (gpt6) {
      try {
        const res = await gpt6.choice(item.observation, item.candidates);
        if (res.isError) {
          bucket.gpt6.errors++;
          if (res.contentFiltered) bucket.gpt6.contentFiltered++;
          bucket.gpt6.predictions.push({ isCorrect: false, confidence: null });
          record.models.gpt6 = { error: res.error, contentFiltered: res.contentFiltered, isCorrect: false, latencyMs: res.latencyMs };
          process.stdout.write(`GPT:${res.contentFiltered ? 'FILTER' : 'ERR'} `);
        } else {
          const isCorrect = cleanMatch(res.winner, item.expected);
          if (isCorrect) bucket.gpt6.correct++;
          bucket.gpt6.latencies.push(res.latencyMs);
          bucket.gpt6.costUsd += res.costUsd;
          bucket.gpt6.predictions.push({ isCorrect, confidence: null });

          record.models.gpt6 = {
            winner: res.winner,
            rawText: res.rawText,
            isCorrect,
            latencyMs: res.latencyMs,
            costUsd: res.costUsd
          };
          process.stdout.write(`GPT:${isCorrect ? '✓' : '✗'}(${res.latencyMs}ms) `);
        }
      } catch (err) {
        bucket.gpt6.errors++;
        bucket.gpt6.predictions.push({ isCorrect: false, confidence: null });
        record.models.gpt6 = { error: err.message, isCorrect: false };
        process.stdout.write(`GPT:EXC `);
      }
    }

    process.stdout.write('\n');
    detailedRecords.push(record);
  }

  // ---------------------------------------------------------------------------
  // STEP 2: ROBUSTNESS & FLIP RATE EVALUATION (PERTURBATIONS)
  // ---------------------------------------------------------------------------
  const perturbationStats = {
    s1stem: { total: 0, flips: 0, orderFlips: 0 },
    decision1: { total: 0, flips: 0, orderFlips: 0, active: !!decision1 },
    luna: { total: 0, flips: 0, orderFlips: 0, active: !!luna },
    gpt6: { total: 0, flips: 0, orderFlips: 0, active: !!gpt6 }
  };

  const perturbationRecords = [];

  if (args.runPerturbations) {
    console.log('\n─────────────────────────────────────────────────────────────────────────────────────────────────────────');
    console.log('▶ [2/3] EVALUATING ROBUSTNESS & FLIP RATE (8 INVARIANT PERTURBATIONS):');
    console.log('  Testing Order Reverse, Cyclic Rotation, Whitespace, Paraphrase, Casing, Punctuation...\n');

    // Select 5 diverse representative items from testSet for perturbation analysis
    const pertBaseItems = testSet.filter(t => t.candidates.length >= 2).slice(0, 5);

    for (const baseItem of pertBaseItems) {
      console.log(`  Evaluating perturbation set for: "${baseItem.observation.slice(0, 60)}..."`);

      // Base evaluations
      const s1Base = await s1stem.choice(baseItem.observation, baseItem.candidates);
      const d1Base = decision1 ? await decision1.choice(baseItem.observation, baseItem.candidates) : null;
      const lunaBase = luna ? await luna.choice(baseItem.observation, baseItem.candidates) : null;
      const gptBase = gpt6 ? await gpt6.choice(baseItem.observation, baseItem.candidates) : null;

      const pList = generatePerturbations(baseItem);

      for (const p of pList) {
        const pRecord = {
          baseObservation: baseItem.observation,
          perturbedObservation: p.observation,
          perturbationName: p.name,
          isOrderPermutation: p.isOrder,
          models: {}
        };

        // S1stem
        perturbationStats.s1stem.total++;
        const s1P = await s1stem.choice(p.observation, p.candidates);
        const s1Flipped = s1P.winner !== s1Base.winner;
        if (s1Flipped) {
          perturbationStats.s1stem.flips++;
          if (p.isOrder) perturbationStats.s1stem.orderFlips++;
        }
        pRecord.models.s1stem = { baseWinner: s1Base.winner, perturbedWinner: s1P.winner, flipped: s1Flipped };

        // Decision-1
        if (decision1 && d1Base && !d1Base.isError) {
          perturbationStats.decision1.total++;
          const d1P = await decision1.choice(p.observation, p.candidates);
          const d1Flipped = !d1P.isError && d1P.winner !== d1Base.winner;
          if (d1Flipped) {
            perturbationStats.decision1.flips++;
            if (p.isOrder) perturbationStats.decision1.orderFlips++;
          }
          pRecord.models.decision1 = { baseWinner: d1Base.winner, perturbedWinner: d1P.winner, flipped: d1Flipped };
        }

        // GPT-6 Luna
        if (luna && lunaBase && !lunaBase.isError) {
          perturbationStats.luna.total++;
          const lunaP = await luna.choice(p.observation, p.candidates);
          const lunaFlipped = !lunaP.isError && lunaP.winner !== lunaBase.winner;
          if (lunaFlipped) {
            perturbationStats.luna.flips++;
            if (p.isOrder) perturbationStats.luna.orderFlips++;
          }
          pRecord.models.luna = { baseWinner: lunaBase.winner, perturbedWinner: lunaP.winner, flipped: lunaFlipped };
        }

        // GPT-6
        if (gpt6 && gptBase && !gptBase.isError) {
          perturbationStats.gpt6.total++;
          const gptP = await gpt6.choice(p.observation, p.candidates);
          const gptFlipped = !gptP.isError && gptP.winner !== gptBase.winner;
          if (gptFlipped) {
            perturbationStats.gpt6.flips++;
            if (p.isOrder) perturbationStats.gpt6.orderFlips++;
          }
          pRecord.models.gpt6 = { baseWinner: gptBase.winner, perturbedWinner: gptP.winner, flipped: gptFlipped };
        }

        perturbationRecords.push(pRecord);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // STEP 3: ACTION SPACE SCALABILITY LIMIT PROBE (K = 32 to 3,000)
  // ---------------------------------------------------------------------------
  let scalabilityResults = [];
  if (args.runScalability) {
    console.log('\n─────────────────────────────────────────────────────────────────────────────────────────────────────────');
    console.log('▶ [3/3] PROBING ACTION SPACE SCALABILITY (K = 32, 64, 128, 255, 256, 1024, 3000):');
    scalabilityResults = await probeActionScalability({
      s1stemClient: s1stem,
      decision1Client: decision1,
      lunaClient: luna,
      gptClient: gpt6
    });
    for (const r of scalabilityResults) {
      console.log(`  K = ${String(r.K).padStart(4)}: S1stem: ${r.s1stem.status} (${r.s1stem.latencyMs}ms) | MD1: ${r.decision1.status || 'N/A'}${r.decision1.isLimit ? ' [BLOCKED HTTP 400]' : ''} | LUNA: ${r.luna?.status || 'N/A'}${r.luna?.isLimit ? ' [BLOCKED]' : ''} | GPT6: ${r.gpt.status || 'N/A'}`);
    }
  }


  // ---------------------------------------------------------------------------
  // STEP 4: COMPUTE STATISTICAL METRICS & SUMMARY
  // ---------------------------------------------------------------------------
  const testTotal = rawStats.test.total;
  const testGroundTruths = testSet.map(t => t.expected);
  const testPredS1 = detailedRecords.filter(r => !r.isDevSet).map(r => r.models.s1stem);
  const testPredD1 = detailedRecords.filter(r => !r.isDevSet).map(r => r.models.decision1 || {});
  const testPredLuna = detailedRecords.filter(r => !r.isDevSet).map(r => r.models.luna || {});
  const testPredGpt = detailedRecords.filter(r => !r.isDevSet).map(r => r.models.gpt6 || {});

  // Wilson Score intervals
  const s1Wilson = computeWilsonInterval(rawStats.test.s1stem.correct, testTotal);
  const d1Wilson = decision1 ? computeWilsonInterval(rawStats.test.decision1.correct, testTotal) : null;
  const lunaWilson = luna ? computeWilsonInterval(rawStats.test.luna.correct, testTotal) : null;
  const gptWilson = gpt6 ? computeWilsonInterval(rawStats.test.gpt6.correct, testTotal) : null;

  // McNemar tests
  const mcNemarVsD1 = decision1 ? computeMcNemarTest(testPredS1, testPredD1, testGroundTruths) : null;
  const mcNemarVsLuna = luna ? computeMcNemarTest(testPredS1, testPredLuna, testGroundTruths) : null;
  const mcNemarVsGpt = gpt6 ? computeMcNemarTest(testPredS1, testPredGpt, testGroundTruths) : null;

  // Calibration
  const s1Calib = computeCalibration(rawStats.test.s1stem.predictions);
  const d1Calib = decision1 ? computeCalibration(rawStats.test.decision1.predictions) : null;
  const lunaCalib = luna ? computeCalibration(rawStats.test.luna.predictions) : null;

  // Latencies
  const s1Lat = analyzeLatencies(rawStats.test.s1stem.latencies);
  const d1Lat = analyzeLatencies(rawStats.test.decision1.latencies);
  const lunaLat = analyzeLatencies(rawStats.test.luna.latencies);
  const gptLat = analyzeLatencies(rawStats.test.gpt6.latencies);

  // Flip Rates
  const s1Flip = perturbationStats.s1stem.total > 0 ? Number(((perturbationStats.s1stem.flips / perturbationStats.s1stem.total) * 100).toFixed(1)) : 'N/A';
  const d1Flip = perturbationStats.decision1.total > 0 ? Number(((perturbationStats.decision1.flips / perturbationStats.decision1.total) * 100).toFixed(1)) : 'N/A';
  const lunaFlip = perturbationStats.luna.total > 0 ? Number(((perturbationStats.luna.flips / perturbationStats.luna.total) * 100).toFixed(1)) : 'N/A';
  const gptFlip = perturbationStats.gpt6.total > 0 ? Number(((perturbationStats.gpt6.flips / perturbationStats.gpt6.total) * 100).toFixed(1)) : 'N/A';

  const summary = {
    metadata: {
      timestamp: new Date().toISOString(),
      hostname: os.hostname(),
      platform: os.platform(),
      cpuModel: os.cpus()[0]?.model || 'Unknown CPU',
      nodeVersion: process.version,
      samplesPerSource: args.samplesPerSource,
      seed: args.seed,
      heldOutTestCount: testTotal,
      contaminatedDevCount: rawStats.dev.total
    },
    headlineTestResults: {
      s1stem: {
        name: 'S1stem v1.2.0 (BGE-Base CLM)',
        runtime: 'Local WebGPU / ONNX (In-Process CPU/GPU)',
        accuracy: s1Wilson.point,
        wilsonCi95: [s1Wilson.low, s1Wilson.high],
        correct: rawStats.test.s1stem.correct,
        total: testTotal,
        latency: s1Lat,
        calibration: s1Calib.isApplicable ? { ece: s1Calib.ece, brier: s1Calib.brier, score: s1Calib.score } : 'N/A',
        flipRate: s1Flip,
        orderShuffleFlips: perturbationStats.s1stem.orderFlips,
        costUsdTotal: 0.0,
        costPer1k: 0.0
      },
      microsoft_decision_1: decision1 ? {
        name: 'Microsoft-Decision-1',
        runtime: 'Azure Foundry SystemOne Decision API',
        accuracy: d1Wilson.point,
        wilsonCi95: [d1Wilson.low, d1Wilson.high],
        correct: rawStats.test.decision1.correct,
        total: testTotal,
        latency: d1Lat,
        calibration: d1Calib.isApplicable ? { ece: d1Calib.ece, brier: d1Calib.brier, score: d1Calib.score } : 'N/A',
        flipRate: d1Flip,
        orderShuffleFlips: perturbationStats.decision1.orderFlips,
        mcNemarVsS1stem: mcNemarVsD1,
        costUsdTotal: Number(rawStats.test.decision1.costUsd.toFixed(6)),
        costPer1k: 0.042
      } : { status: 'Not Evaluated (No API Key)' },
      gpt_6_luna: luna ? {
        name: 'gpt-6-luna',
        runtime: luna.runtime,
        accuracy: lunaWilson.point,
        wilsonCi95: [lunaWilson.low, lunaWilson.high],
        correct: rawStats.test.luna.correct,
        total: testTotal,
        latency: lunaLat,
        calibration: lunaCalib?.isApplicable ? { ece: lunaCalib.ece, brier: lunaCalib.brier, score: lunaCalib.score } : 'N/A',
        flipRate: lunaFlip,
        orderShuffleFlips: perturbationStats.luna.orderFlips,
        contentFilteredCount: rawStats.test.luna.contentFiltered,
        mcNemarVsS1stem: mcNemarVsLuna,
        costUsdTotal: Number(rawStats.test.luna.costUsd.toFixed(6)),
        costPer1k: Number(((rawStats.test.luna.costUsd / testTotal) * 1000).toFixed(4))
      } : { status: 'Not Evaluated (No API Key)' },
      gpt_6_1_sol: gpt6 ? {
        name: 'gpt-6.1-sol',
        runtime: 'Azure Foundry OpenAI Chat Completions API',
        accuracy: gptWilson.point,
        wilsonCi95: [gptWilson.low, gptWilson.high],
        correct: rawStats.test.gpt6.correct,
        total: testTotal,
        latency: gptLat,
        calibration: 'N/A (Generative output without calibrated probabilities)',
        flipRate: gptFlip,
        orderShuffleFlips: perturbationStats.gpt6.orderFlips,
        contentFilteredCount: rawStats.test.gpt6.contentFiltered,
        mcNemarVsS1stem: mcNemarVsGpt,
        costUsdTotal: Number(rawStats.test.gpt6.costUsd.toFixed(6)),
        costPer1k: Number(((rawStats.test.gpt6.costUsd / testTotal) * 1000).toFixed(3))
      } : { status: 'Not Evaluated (No API Key)' }
    },
    contaminatedDevSetResults: rawStats.dev.total > 0 ? {
      warning: 'Contains environmental cues matching internal heuristics. Reported solely for transparency.',
      count: rawStats.dev.total,
      s1stemAccuracy: Number(((rawStats.dev.s1stem.correct / rawStats.dev.total) * 100).toFixed(1)),
      decision1Accuracy: decision1 ? Number(((rawStats.dev.decision1.correct / rawStats.dev.total) * 100).toFixed(1)) : 'N/A',
      lunaAccuracy: luna ? Number(((rawStats.dev.luna.correct / rawStats.dev.total) * 100).toFixed(1)) : 'N/A',
      gpt6Accuracy: gpt6 ? Number(((rawStats.dev.gpt6.correct / rawStats.dev.total) * 100).toFixed(1)) : 'N/A'
    } : null,
    scalabilityProbe: scalabilityResults
  };

  // ---------------------------------------------------------------------------
  // STEP 5: WRITE EXPORTS
  // ---------------------------------------------------------------------------
  // 1. records.jsonl
  const jsonlPath = path.join(resultsDir, 'records.jsonl');
  const jsonlStream = fs.createWriteStream(jsonlPath);
  for (const r of detailedRecords) {
    jsonlStream.write(JSON.stringify(r) + '\n');
  }
  jsonlStream.end();

  // 2. summary.json
  const summaryPath = path.join(resultsDir, 'summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

  // Mirror to legacy locations for compatibility
  fs.writeFileSync('Benchmark/results.json', JSON.stringify(summary, null, 2));

  // 3. report.md
  const reportMd = generateMarkdownReport(summary);
  fs.writeFileSync(path.join(resultsDir, 'report.md'), reportMd);
  fs.writeFileSync('Benchmark/results.md', reportMd);

  // Print final summary to console
  console.log('\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗');
  console.log('║   FINAL BENCHMARK LEADERBOARD (HELD-OUT TEST SET)                                                       ║');
  console.log('╠═════════════════════════════════════════════════════════════════════════════════════════════════════════╣');
  console.log(`║ S1stem v1.2.0 (CLM):    Accuracy: ${s1Wilson.point}% [${s1Wilson.low}% - ${s1Wilson.high}%] | p50: ${s1Lat.p50} ms | Cost: $0.000 / 1k`);
  if (decision1) {
    console.log(`║ Microsoft-Decision-1:   Accuracy: ${d1Wilson.point}% [${d1Wilson.low}% - ${d1Wilson.high}%] | p50: ${d1Lat.p50} ms | Cost: $0.042 / 1k | McNemar p=${mcNemarVsD1.pValue}`);
  }
  if (luna) {
    console.log(`║ gpt-6-luna (Decisions): Accuracy: ${lunaWilson.point}% [${lunaWilson.low}% - ${lunaWilson.high}%] | p50: ${lunaLat.p50} ms | Cost: ~$${summary.headlineTestResults.gpt_6_luna.costPer1k} / 1k | McNemar p=${mcNemarVsLuna.pValue}`);
  }
  if (gpt6) {
    console.log(`║ gpt-6.1-sol (Frontier): Accuracy: ${gptWilson.point}% [${gptWilson.low}% - ${gptWilson.high}%] | p50: ${gptLat.p50} ms | Cost: ~$${summary.headlineTestResults.gpt_6_1_sol.costPer1k} / 1k | McNemar p=${mcNemarVsGpt.pValue}`);
  }
  console.log('╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n');

  console.log(`[Complete Audit Records] -> ${jsonlPath}`);
  console.log(`[Summary Metrics JSON]   -> ${summaryPath}`);
  console.log(`[Report MD]     -> ${path.join(resultsDir, 'report.md')}\n`);
}

function generateMarkdownReport(summary) {
  const m = summary.metadata;
  const h = summary.headlineTestResults;
  const s1 = h.s1stem;
  const d1 = h.microsoft_decision_1;
  const luna = h.gpt_6_luna;
  const gpt = h.gpt_6_1_sol;

  return `# Empirical Decision Benchmark: S1stem vs Microsoft-Decision-1 vs GPT-6 Luna (Decisions) vs GPT-6.1 Sol

**Execution Date**: ${m.timestamp}  
**Hardware & Platform**: ${m.cpuModel} (${m.platform}, Node ${m.nodeVersion})  
**Evaluation Protocol**: Stratified Sampling (Seed: ${m.seed}, ${m.samplesPerSource} samples/source)  
**Evaluated Held-Out Cases**: ${m.heldOutTestCount} items  

## 1. Headline Comparative Matrix (Held-Out Test Set)

| Model / Architecture | Runtime & Deployment | Top-1 Accuracy (95% Wilson CI) | Median Latency (p50) | Tail Latency (p95) | Calibration (ECE / Brier) | Invariant Flip Rate | Cost / 1k Ops |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1stem v1.2.0 (CLM)** | ${s1.runtime} | **${s1.accuracy}%** [${s1.wilsonCi95[0]}% - ${s1.wilsonCi95[1]}%] | **${s1.latency.p50} ms** | **${s1.latency.p95} ms** | ${s1.calibration.ece ? `${s1.calibration.ece} / ${s1.calibration.brier}` : 'N/A'} | **${s1.flipRate}%** | **$0.000** |
| **Microsoft-Decision-1** | ${d1.runtime || 'Azure Cloud'} | ${d1.accuracy ? `${d1.accuracy}% [${d1.wilsonCi95[0]}% - ${d1.wilsonCi95[1]}%]` : 'N/A'} | ${d1.latency?.p50 || 'N/A'} ms | ${d1.latency?.p95 || 'N/A'} ms | ${d1.calibration?.ece ? `${d1.calibration.ece} / ${d1.calibration.brier}` : 'N/A'} | ${d1.flipRate || 'N/A'}% | $0.042 |
| **gpt-6-luna (Decisions)** | ${luna.runtime || 'OpenAI Decisions API'} | ${luna.accuracy ? `${luna.accuracy}% [${luna.wilsonCi95[0]}% - ${luna.wilsonCi95[1]}%]` : 'N/A'} | ${luna.latency?.p50 || 'N/A'} ms | ${luna.latency?.p95 || 'N/A'} ms | ${luna.calibration?.ece ? `${luna.calibration.ece} / ${luna.calibration.brier}` : 'N/A'} | ${luna.flipRate || 'N/A'}% | ~$${luna.costPer1k || '0.001'} |
| **gpt-6.1-sol (Frontier)** | ${gpt.runtime || 'Azure Cloud'} | ${gpt.accuracy ? `${gpt.accuracy}% [${gpt.wilsonCi95[0]}% - ${gpt.wilsonCi95[1]}%]` : 'N/A'} | ${gpt.latency?.p50 || 'N/A'} ms | ${gpt.latency?.p95 || 'N/A'} ms | N/A *(Generative)* | ${gpt.flipRate || 'N/A'}% | ~$${gpt.costPer1k || '3.000'} |

## 2. Statistical Significance (Paired McNemar Test vs S1stem)

* **S1stem vs Microsoft-Decision-1**:
  * Contingency: ${d1.mcNemarVsS1stem ? `both correct: ${d1.mcNemarVsS1stem.bothCorrect}, S1stem only: ${d1.mcNemarVsS1stem.modelACorrectOnly}, MD1 only: ${d1.mcNemarVsS1stem.modelBCorrectOnly}, both wrong: ${d1.mcNemarVsS1stem.bothWrong}` : 'N/A'}
  * McNemar $\\chi^2$: ${d1.mcNemarVsS1stem?.chi2 ?? 'N/A'}, $p$-value: **${d1.mcNemarVsS1stem?.pValue ?? 'N/A'}** (${d1.mcNemarVsS1stem?.isSignificant ? 'Statistically Significant (p < 0.05)' : 'Not Statistically Significant'})

* **S1stem vs gpt-6-luna (Decisions API)**:
  * Contingency: ${luna.mcNemarVsS1stem ? `both correct: ${luna.mcNemarVsS1stem.bothCorrect}, S1stem only: ${luna.mcNemarVsS1stem.modelACorrectOnly}, Luna only: ${luna.mcNemarVsS1stem.modelBCorrectOnly}, both wrong: ${luna.mcNemarVsS1stem.bothWrong}` : 'N/A'}
  * McNemar $\\chi^2$: ${luna.mcNemarVsS1stem?.chi2 ?? 'N/A'}, $p$-value: **${luna.mcNemarVsS1stem?.pValue ?? 'N/A'}** (${luna.mcNemarVsS1stem?.isSignificant ? 'Statistically Significant (p < 0.05)' : 'Not Statistically Significant'})

* **S1stem vs gpt-6.1-sol (Chat Completions)**:
  * Contingency: ${gpt.mcNemarVsS1stem ? `both correct: ${gpt.mcNemarVsS1stem.bothCorrect}, S1stem only: ${gpt.mcNemarVsS1stem.modelACorrectOnly}, GPT-6 only: ${gpt.mcNemarVsS1stem.modelBCorrectOnly}, both wrong: ${gpt.mcNemarVsS1stem.bothWrong}` : 'N/A'}
  * McNemar $\\chi^2$: ${gpt.mcNemarVsS1stem?.chi2 ?? 'N/A'}, $p$-value: **${gpt.mcNemarVsS1stem?.pValue ?? 'N/A'}** (${gpt.mcNemarVsS1stem?.isSignificant ? 'Statistically Significant (p < 0.05)' : 'Not Statistically Significant'})

## 3. Action Space Scalability Limit ($K = 32$ to $K = 3,000$ Candidates)

| Candidate Count ($K$) | S1stem Caching | S1stem Latency | Microsoft-Decision-1 Status | gpt-6-luna Status | gpt-6.1-sol Status |
| :---: | :---: | :---: | :--- | :--- | :--- |
${summary.scalabilityProbe.map(r => `| **K = ${r.K}** | ${r.s1stem.registrationMs ? `${r.s1stem.registrationMs} ms` : 'N/A'} | **${r.s1stem.latencyMs ? `${r.s1stem.latencyMs} ms` : 'N/A'}** | ${r.decision1.status || 'N/A'}${r.decision1.isLimit ? ' **(BLOCKED HTTP 400)**' : ''} | ${r.luna?.status || 'N/A'}${r.luna?.isLimit ? ' **(BLOCKED)**' : ''} | ${r.gpt.status || 'N/A'} |`).join('\n')}

## 4. Contaminated Dev Set Disclosure

${summary.contaminatedDevSetResults ? `
> [!WARNING]
> The dev set contains historical operational testbed items where environmental tokens partially overlap with contextual polarity rules.
> S1stem Dev Accuracy: **${summary.contaminatedDevSetResults.s1stemAccuracy}%** vs MD1: **${summary.contaminatedDevSetResults.decision1Accuracy}%** vs Luna: **${summary.contaminatedDevSetResults.lunaAccuracy || 'N/A'}%** vs GPT-6: **${summary.contaminatedDevSetResults.gpt6Accuracy}%** (${summary.contaminatedDevSetResults.count} items).
` : 'Dev set was excluded from this run.'}
`;
}

main().catch(console.error);
