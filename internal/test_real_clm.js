import fs from 'fs';
import path from 'path';
import { pipeline } from '@huggingface/transformers';

async function run() {
  console.log('Loading real neural embedding pipeline...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const csvPath = path.resolve('internal/fixtures/banking77_test.csv');
  const allLines = fs.readFileSync(csvPath, 'utf-8').trim().split('\n').slice(1);
  
  const parsed = [];
  const intentSet = new Set();
  for (const line of allLines) {
    const lastComma = line.lastIndexOf(',');
    if (lastComma === -1) continue;
    let text = line.slice(0, lastComma).trim();
    if (text.startsWith('"') && text.endsWith('"')) text = text.slice(1, -1);
    const expected = line.slice(lastComma + 1).trim();
    intentSet.add(expected);
    parsed.push({ text, expected });
  }

  const allIntents = Array.from(intentSet);
  console.log(`Pre-caching action matrix for ALL ${allIntents.length} distinct banking intents...`);

  // One-Shot Action Pre-Caching (CLM Action Encoder E_a)
  const t0 = performance.now();
  const intentMap = [];
  for (const intent of allIntents) {
    const descriptiveText = intent.replace(/_/g, ' ');
    const res = await extractor(descriptiveText, { pooling: 'mean', normalize: true });
    intentMap.push({ intent, vec: res.data });
  }
  const regMs = performance.now() - t0;
  console.log(`Pre-cached ${allIntents.length} intents in ${regMs.toFixed(2)} ms.`);

  // Test real state queries (CLM State Encoder E_s + fast dot-product)
  let correct = 0;
  const latencies = [];

  const testCount = Math.min(50, parsed.length);
  for (let i = 0; i < testCount; i++) {
    const { text, expected } = parsed[i];
    const tQuery0 = performance.now();
    const qRes = await extractor(text, { pooling: 'mean', normalize: true });
    const qVec = qRes.data;

    let bestIntent = '';
    let bestScore = -Infinity;
    for (const item of intentMap) {
      let dot = 0;
      for (let d = 0; d < qVec.length; d++) {
        dot += qVec[d] * item.vec[d];
      }
      if (dot > bestScore) {
        bestScore = dot;
        bestIntent = item.intent;
      }
    }
    const qLatency = performance.now() - tQuery0;
    latencies.push(qLatency);

    const isMatch = (bestIntent === expected);
    if (isMatch) correct++;

    if (i < 8) {
      console.log(`[${i + 1}] Query: "${text.slice(0, 45)}..."`);
      console.log(`    Expected: ${expected} | Predicted: ${bestIntent} (${bestScore.toFixed(3)}) | Latency: ${qLatency.toFixed(2)} ms | Match: ${isMatch ? 'YES' : 'NO'}`);
    }
  }

  const meanLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const acc = (correct / testCount) * 100;
  console.log('\n--- Real Neural Results (Zero Mocks) ---');
  console.log(`Evaluated: ${testCount} queries`);
  console.log(`Correct:   ${correct}/${testCount} (${acc.toFixed(1)}%)`);
  console.log(`Mean Lat:  ${meanLat.toFixed(2)} ms per state inference`);
}

run().catch(console.error);
