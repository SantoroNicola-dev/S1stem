#!/usr/bin/env node
/**
 * S1stem Model Exporter & Standalone Offline Packager
 * ===================================================
 * Downloads / exports the full precision FP32/FP16 ONNX model (Xenova/bge-large-en-v1.5, 1024-dim)
 * and its tokenizer files to a standalone local directory.
 * 
 * Enables 100% offline, air-gapped, zero-network execution across:
 * - Docker containers
 * - Local Node.js / TypeScript agents
 * - Python ONNX Runtime (LangChain, AutoGen, CrewAI)
 * - C++ / Rust / Go native runtimes
 * 
 * Usage:
 *   node scripts/download-model.js [--dest ./models/s1stem-bge-large]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Parse CLI arguments
const args = process.argv.slice(2);
let destDir = path.join(rootDir, 'models', 's1stem-bge-large');
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--dest' && args[i + 1]) {
    destDir = path.resolve(process.cwd(), args[i + 1]);
    i++;
  }
}

const MODEL_ID = 'Xenova/bge-large-en-v1.5';
const EXPECTED_FILES = [
  'config.json',
  'tokenizer_config.json',
  'tokenizer.json',
  path.join('onnx', 'model.onnx'),
];

console.log('╔════════════════════════════════════════════════════════════════════════════════╗');
console.log('║        S1STEM STANDALONE ONNX EXPORTER & OFFLINE PACKAGER                      ║');
console.log('╚════════════════════════════════════════════════════════════════════════════════╝\n');
console.log(`• Target Model:       ${MODEL_ID} (1024-dim, cls pooling)`);
console.log(`• Output Directory:   ${destDir}\n`);

async function main() {
  fs.mkdirSync(destDir, { recursive: true });
  fs.mkdirSync(path.join(destDir, 'onnx'), { recursive: true });

  const { pipeline, env } = await import('@huggingface/transformers');
  
  // Cache location used by Transformers.js
  const cacheBase = env.cacheDir || path.join(rootDir, 'node_modules', '@huggingface', 'transformers', '.cache');
  const sourceCache = path.join(cacheBase, ...MODEL_ID.split('/'));

  console.log('[1/3] Checking local cache for pre-downloaded weights...');
  let hasCached = fs.existsSync(sourceCache) && fs.existsSync(path.join(sourceCache, 'onnx', 'model.onnx'));

  if (!hasCached) {
    console.log('[1/3] Model not found in local cache. Initiating full download from HuggingFace Hub...');
    console.log('      (Downloading model.onnx ~1.27 GB. This happens once)...');
    const t0 = performance.now();
    await pipeline('feature-extraction', MODEL_ID, {
      progress_callback: (p) => {
        if (p.file && p.progress !== undefined) {
          process.stdout.write(`\r      Downloading ${p.file}: ${Math.round(p.progress)}%`);
        }
      },
    });
    console.log(`\n      ✓ Download complete in ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
  } else {
    console.log(`      ✓ Found pre-cached weights in ${sourceCache}`);
  }

  console.log('\n[2/3] Exporting standalone files to target distribution directory...');
  let totalBytes = 0;
  for (const relFile of EXPECTED_FILES) {
    const src = path.join(sourceCache, relFile);
    const dst = path.join(destDir, relFile);

    if (!fs.existsSync(src)) {
      throw new Error(`Critical model file missing from cache: ${src}`);
    }

    fs.copyFileSync(src, dst);
    const stat = fs.statSync(dst);
    totalBytes += stat.size;
    const mb = (stat.size / (1024 * 1024)).toFixed(2);
    console.log(`      ✓ Copied ${relFile.padEnd(24)} (${mb.padStart(8)} MB) -> ${dst}`);
  }

  // Also write a standalone metadata manifest
  const manifest = {
    modelId: MODEL_ID,
    architecture: 'BGE-Large CLM Two-Tower Dual Encoder',
    embeddingDim: 1024,
    pooling: 'cls',
    precision: 'FP32/FP16 (Full Precision)',
    totalSizeBytes: totalBytes,
    exportedAt: new Date().toISOString(),
    files: EXPECTED_FILES,
  };
  fs.writeFileSync(path.join(destDir, 's1stem-manifest.json'), JSON.stringify(manifest, null, 2));

  console.log('\n[3/3] Verifying standalone offline integrity...');
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  const testExtractor = await pipeline('feature-extraction', destDir, { local_files_only: true });
  const sample = await testExtractor('S1stem offline verification test vector', { pooling: 'cls', normalize: true });

  if (sample.data.length !== 1024) {
    throw new Error(`Embedding dimension mismatch: expected 1024, got ${sample.data.length}`);
  }

  console.log(`      ✓ Pure offline feature extraction verified!`);
  console.log(`      ✓ Generated 1024-dimensional normalized vector: [${sample.data.slice(0, 3).map(v => v.toFixed(4)).join(', ')}, ...]`);

  console.log('\n╔════════════════════════════════════════════════════════════════════════════════╗');
  console.log('║                         STANDALONE EXPORT READY                                ║');
  console.log('╚════════════════════════════════════════════════════════════════════════════════╝');
  console.log(`• Destination:     ${destDir}`);
  console.log(`• Total Footprint: ${(totalBytes / (1024 * 1024)).toFixed(1)} MB (~${(totalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB)`);
  console.log(`• Usage (Node.js): new S1stemEngine({ localModelPath: '${destDir.replace(/\\/g, '/')}' })`);
  console.log(`• Usage (Docker):  Mount as volume: -v ${destDir}:/app/models:ro\n`);
}

main().catch((err) => {
  console.error('\n❌ Export failed:', err);
  process.exit(1);
});
