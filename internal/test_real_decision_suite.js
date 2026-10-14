import fs from 'fs';
import path from 'path';
import { pipeline } from '@huggingface/transformers';

async function testDecisionSuite() {
  console.log('Loading real neural embedding model...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const suite = JSON.parse(fs.readFileSync('internal/fixtures/decision_suite.json', 'utf-8'));
  console.log('Testing Reasoning Multi-Choice...');

  let correct = 0;
  let total = 0;
  for (const item of suite.reasoning_multichoice) {
    total++;
    const qVec = (await extractor(item.question, { pooling: 'mean', normalize: true })).data;
    
    let bestOpt = null;
    let bestScore = -Infinity;
    for (const opt of item.options) {
      const optVec = (await extractor(opt.label || opt.id, { pooling: 'mean', normalize: true })).data;
      let dot = 0;
      for (let d = 0; d < qVec.length; d++) dot += qVec[d] * optVec[d];
      if (dot > bestScore) {
        bestScore = dot;
        bestOpt = opt;
      }
    }
    const isOk = (bestOpt.id === item.correct);
    if (isOk) correct++;
    console.log(`Q: "${item.question.slice(0, 50)}..." -> Got: ${bestOpt.id}, Expected: ${item.correct}, Match: ${isOk ? 'YES' : 'NO'}`);
  }
  console.log(`Score: ${correct}/${total} (${(correct/total*100).toFixed(1)}%)`);
}

testDecisionSuite().catch(console.error);
