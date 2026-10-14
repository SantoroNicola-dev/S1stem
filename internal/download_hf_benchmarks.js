/**
 * Download & Curate Hugging Face Benchmark Datasets for S1stem
 * =============================================================
 * Fetches 500 validated samples for each of the 4 Generalization Benchmark Pillars:
 *   1. Amazon MASSIVE (SetFit/amazon_massive_intent_en-US) - Multi-Domain Intent Routing (60 intents)
 *   2. Glaive Function Calling v2 (glaiveai/glaive-function-calling-v2) - Agent Tool Selection Dispatch
 *   3. HellaSwag (Rowan/hellaswag) - Situated Action & Physical Continuation
 *   4. When2Call / Abstention (nvidia/When2Call) - Rejection, Margin Calibration & Escalation
 * 
 * Sources & Citations:
 *   - MASSIVE: https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US
 *   - Glaive Tool Calling: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
 *   - HellaSwag: https://huggingface.co/datasets/Rowan/hellaswag
 *   - When2Call: https://huggingface.co/datasets/nvidia/When2Call
 * 
 * License: Apache-2.0
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const FIXTURES_DIR = path.resolve('internal/fixtures/huggingface');
if (!fs.existsSync(FIXTURES_DIR)) {
  fs.mkdirSync(FIXTURES_DIR, { recursive: true });
}

async function fetchHfRows(dataset, config, split, offset, limit) {
  const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} from Hugging Face for ${dataset}: ${await res.text()}`);
  }
  const data = await res.json();
  return (data.rows || []).map(r => r.row);
}

function computeSha256(content) {
  return crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * 1. Amazon MASSIVE
 */
async function downloadMassive(targetCount = 500) {
  console.log(`[HF Downloader] Fetching Amazon MASSIVE (${targetCount} samples)...`);
  const samples = [];
  const allIntents = new Set();
  let offset = 0;
  const pageSize = 100;

  while (samples.length < targetCount) {
    const rows = await fetchHfRows('SetFit/amazon_massive_intent_en-US', 'default', 'test', offset, pageSize);
    if (!rows.length) break;
    for (const r of rows) {
      if (r.text && r.label_text) {
        allIntents.add(r.label_text);
        samples.push({
          id: String(r.id || samples.length),
          text: r.text.trim(),
          ground_truth: r.label_text.trim()
        });
        if (samples.length >= targetCount) break;
      }
    }
    offset += pageSize;
  }

  const payload = {
    benchmark: 'Amazon MASSIVE en-US',
    pillar: 'Multi-Domain Real Intent Routing',
    source: 'https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US',
    split: 'test',
    candidate_actions: Array.from(allIntents).sort(),
    total_samples: samples.length,
    samples
  };

  const filePath = path.join(FIXTURES_DIR, 'massive_500.json');
  const jsonStr = JSON.stringify(payload, null, 2);
  fs.writeFileSync(filePath, jsonStr, 'utf-8');
  console.log(`[HF Downloader] Saved massive_500.json (${samples.length} items, SHA256: ${computeSha256(jsonStr).slice(0, 12)}...)`);
  return payload;
}

/**
 * 2. Glaive Function Calling v2
 */
async function downloadToolCalling(targetCount = 500) {
  console.log(`[HF Downloader] Fetching Glaive Function Calling v2 (${targetCount} samples)...`);
  const samples = [];
  const globalToolSet = new Set();
  let offset = 0;
  const pageSize = 100;

  while (samples.length < targetCount) {
    const rows = await fetchHfRows('glaiveai/glaive-function-calling-v2', 'default', 'train', offset, pageSize);
    if (!rows.length) break;
    for (const r of rows) {
      const sys = r.system || '';
      const chat = r.chat || '';
      const userMatch = chat.match(/USER:\s*([\s\S]*?)\n\n/);
      const fnMatch = chat.match(/<functioncall>[\s\S]*?"name":\s*"([^"]+)"/);

      if (userMatch && fnMatch) {
        const query = userMatch[1].trim();
        const targetTool = fnMatch[1].trim();
        if (query && targetTool) {
          // Extract tools mentioned in system
          const tools = [];
          const fnMatches = sys.matchAll(/"name":\s*"([^"]+)"/g);
          for (const m of fnMatches) {
            tools.push(m[1].trim());
            globalToolSet.add(m[1].trim());
          }
          if (!tools.includes(targetTool)) {
            tools.push(targetTool);
            globalToolSet.add(targetTool);
          }

          samples.push({
            id: `tool_${samples.length}`,
            query,
            target_tool: targetTool,
            local_tools: Array.from(new Set(tools))
          });
          if (samples.length >= targetCount) break;
        }
      }
    }
    offset += pageSize;
  }

  const payload = {
    benchmark: 'Glaive Function Calling v2',
    pillar: 'Agent Tool Selection Dispatch',
    source: 'https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2',
    split: 'train',
    global_tool_catalog: Array.from(globalToolSet).sort(),
    total_samples: samples.length,
    samples
  };

  const filePath = path.join(FIXTURES_DIR, 'tool_calling_500.json');
  const jsonStr = JSON.stringify(payload, null, 2);
  fs.writeFileSync(filePath, jsonStr, 'utf-8');
  console.log(`[HF Downloader] Saved tool_calling_500.json (${samples.length} items, SHA256: ${computeSha256(jsonStr).slice(0, 12)}...)`);
  return payload;
}

/**
 * 3. HellaSwag
 */
async function downloadHellaSwag(targetCount = 500) {
  console.log(`[HF Downloader] Fetching HellaSwag (${targetCount} samples)...`);
  const samples = [];
  let offset = 0;
  const pageSize = 100;

  while (samples.length < targetCount) {
    const rows = await fetchHfRows('Rowan/hellaswag', 'default', 'validation', offset, pageSize);
    if (!rows.length) break;
    for (const r of rows) {
      if (r.ctx && Array.isArray(r.endings) && r.endings.length === 4 && r.label !== undefined) {
        samples.push({
          id: String(r.ind || samples.length),
          activity: r.activity_label || 'activity',
          context: r.ctx.trim(),
          candidate_endings: r.endings.map(e => e.trim()),
          ground_truth_index: Number(r.label)
        });
        if (samples.length >= targetCount) break;
      }
    }
    offset += pageSize;
  }

  const payload = {
    benchmark: 'HellaSwag',
    pillar: 'Situated Action & Physical Continuation',
    source: 'https://huggingface.co/datasets/Rowan/hellaswag',
    split: 'validation',
    total_samples: samples.length,
    samples
  };

  const filePath = path.join(FIXTURES_DIR, 'hellaswag_500.json');
  const jsonStr = JSON.stringify(payload, null, 2);
  fs.writeFileSync(filePath, jsonStr, 'utf-8');
  console.log(`[HF Downloader] Saved hellaswag_500.json (${samples.length} items, SHA256: ${computeSha256(jsonStr).slice(0, 12)}...)`);
  return payload;
}

/**
 * 4. When2Call / Abstention Benchmark
 */
async function downloadAbstention(targetCount = 500) {
  console.log(`[HF Downloader] Fetching When2Call & Abstention (${targetCount} samples)...`);
  const samples = [];
  const actionChoices = ['execute_tool', 'direct_answer', 'request_clarification', 'abstain_cannot_answer'];
  let offset = 0;
  const pageSize = 100;

  while (samples.length < targetCount) {
    const rows = await fetchHfRows('nvidia/When2Call', 'test', 'mcq', offset, pageSize);
    if (!rows.length) break;
    for (const r of rows) {
      if (r.question && r.correct_answer) {
        let normalizedAnswer = 'abstain_cannot_answer';
        if (r.correct_answer === 'cannot_answer') {
          normalizedAnswer = 'abstain_cannot_answer';
        } else if (r.correct_answer === 'tool_call') {
          normalizedAnswer = 'execute_tool';
        } else if (r.correct_answer === 'direct') {
          normalizedAnswer = 'direct_answer';
        } else if (r.correct_answer === 'request_for_info') {
          normalizedAnswer = 'request_clarification';
        }

        samples.push({
          id: String(r.uuid || samples.length),
          question: r.question.trim(),
          ground_truth_action: normalizedAnswer,
          original_answer: r.correct_answer,
          is_abstention: normalizedAnswer === 'abstain_cannot_answer'
        });
        if (samples.length >= targetCount) break;
      }
    }
    offset += pageSize;
  }

  const payload = {
    benchmark: 'When2Call (Nvidia Research)',
    pillar: 'Action Abstention & Confidence Calibration',
    source: 'https://huggingface.co/datasets/nvidia/When2Call',
    split: 'test/mcq',
    action_space: actionChoices,
    total_samples: samples.length,
    samples
  };

  const filePath = path.join(FIXTURES_DIR, 'abstention_500.json');
  const jsonStr = JSON.stringify(payload, null, 2);
  fs.writeFileSync(filePath, jsonStr, 'utf-8');
  console.log(`[HF Downloader] Saved abstention_500.json (${samples.length} items, SHA256: ${computeSha256(jsonStr).slice(0, 12)}...)`);
  return payload;
}

async function run() {
  console.log('=============================================================================');
  console.log(' S1STEM HUGGING FACE BENCHMARK FIXTURE GENERATOR (2,000 TEST SAMPLES)       ');
  console.log('=============================================================================');
  const t0 = performance.now();
  await downloadMassive(500);
  await downloadToolCalling(500);
  await downloadHellaSwag(500);
  await downloadAbstention(500);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`\nAll 4 benchmark fixtures downloaded successfully in ${elapsed}s.`);
}

run().catch(err => {
  console.error('Fatal error downloading Hugging Face fixtures:', err);
  process.exit(1);
});
