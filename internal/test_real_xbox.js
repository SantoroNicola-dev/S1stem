import fs from 'fs';
import { pipeline } from '@huggingface/transformers';

async function testXboxSuite() {
  console.log('Loading real neural model for XBOX Research labeling...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const suite = JSON.parse(fs.readFileSync('internal/fixtures/xbox_research_suite.json', 'utf-8'));
  const themes = suite.xbox_research_labeling.themes;

  // 1. One-shot caching of theme embeddings
  const t0 = performance.now();
  const themeMap = [];
  for (const t of themes) {
    const desc = `${t.label}: ${t.description}`;
    const res = await extractor(desc, { pooling: 'mean', normalize: true });
    themeMap.push({ id: t.id, vec: res.data });
  }
  const regMs = performance.now() - t0;
  console.log(`Pre-cached ${themes.length} themes in ${regMs.toFixed(2)} ms.`);

  // 2. Evaluate each dataset
  for (const [dName, samples] of Object.entries(suite.xbox_research_labeling.datasets)) {
    let correct = 0;
    const lats = [];
    for (const s of samples) {
      const t1 = performance.now();
      const qRes = await extractor(s.text, { pooling: 'mean', normalize: true });
      const qVec = qRes.data;

      let bestTheme = '';
      let bestScore = -Infinity;
      for (const th of themeMap) {
        let dot = 0;
        for (let d = 0; d < qVec.length; d++) dot += qVec[d] * th.vec[d];
        if (dot > bestScore) {
          bestScore = dot;
          bestTheme = th.id;
        }
      }
      const evalMs = performance.now() - t1;
      lats.push(evalMs);
      if (bestTheme === s.expected) correct++;
    }
    const meanLat = lats.reduce((a, b) => a + b, 0) / lats.length;
    const acc = (correct / samples.length) * 100;
    console.log(`Dataset: ${dName.padEnd(16)} | Correct: ${correct}/${samples.length} (${acc.toFixed(1)}%) | Mean Latency: ${meanLat.toFixed(2)} ms`);
  }
}

testXboxSuite().catch(console.error);
