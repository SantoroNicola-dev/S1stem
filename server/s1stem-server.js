/**
 * S1stem High-Performance Microservice Server
 * ============================================
 * Exposes System 1 Reflex Decision Engine over a zero-overhead HTTP/REST interface.
 * Designed for frictionless integration with Python (LangChain, AutoGen, CrewAI),
 * Node.js, Go, Rust, and containerized agent stacks.
 * 
 * Endpoints:
 *   GET  /health           - System health, uptime, memory, and model metadata
 *   POST /choice           - Polymorphic choice resolution (state + candidate actions)
 *   POST /register-actions - One-shot pre-caching of action matrix A in RAM
 *   POST /step             - O(1) ultra-fast sub-millisecond query against pre-cached actions
 *   POST /reset            - Reset cached actions or telemetry stats
 * 
 * @license Apache-2.0
 */

import http from 'http';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { S1stemEngine } from '../core/s1stem.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const PORT = parseInt(process.env.PORT || process.env.S1STEM_PORT || '8000', 10);
const HOST = process.env.HOST || '0.0.0.0';

// Check for local offline model
let localModelPath = process.env.S1STEM_MODEL_PATH || null;
if (!localModelPath) {
  const defaultLocal = path.join(rootDir, 'models', 's1stem-bge-large');
  if (fs.existsSync(defaultLocal)) {
    localModelPath = defaultLocal;
  }
}

console.log('╔════════════════════════════════════════════════════════════════════════════════╗');
console.log('║               S1STEM SYSTEM 1 DECISION MICROSERVICE SERVER                     ║');
console.log('╚════════════════════════════════════════════════════════════════════════════════╝\n');
console.log(`• Port:             ${PORT}`);
console.log(`• Host:             ${HOST}`);
console.log(`• Offline Model:    ${localModelPath || 'Remote HuggingFace Hub (Xenova/bge-large-en-v1.5)'}\n`);

const engine = new S1stemEngine({
  localModelPath,
  autoLoadModel: true,
});

// Warm up model during boot
console.log('[boot] Initializing neural extractor...');
const tBoot0 = performance.now();
await engine.initExtractor();
console.log(`[boot] Neural extractor ready in ${(performance.now() - tBoot0).toFixed(1)} ms.`);

function sendJson(res, statusCode, data) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });
  res.end(JSON.stringify(data));
}

function parseJsonBody(req, maxBytes = 10 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > maxBytes) {
        req.destroy();
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      if (!body.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (err) {
        reject(new Error(`Invalid JSON: ${err.message}`));
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  // CORS Preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    res.end();
    return;
  }

  const url = req.url.split('?')[0];

  try {
    // 1. Health & Status Check
    if (req.method === 'GET' && (url === '/' || url === '/health')) {
      const mem = process.memoryUsage();
      sendJson(res, 200, {
        status: 'ready',
        engine: 'S1stem v1.3.0',
        modelId: engine.config.embeddingModelId,
        embeddingDim: engine.embeddingDim,
        isOfflineModel: !!localModelPath,
        cachedActionsCount: engine.actionCache ? engine.actionCache.length : 0,
        stats: engine.stats,
        uptimeSeconds: Math.round(process.uptime()),
        memoryMB: {
          rss: Math.round(mem.rss / 1024 / 1024),
          heapUsed: Math.round(mem.heapUsed / 1024 / 1024),
        },
      });
      return;
    }

    // 2. Full Choice Resolution: POST /choice
    if (req.method === 'POST' && url === '/choice') {
      const payload = await parseJsonBody(req);
      const state = payload.state || payload.prompt || payload.query || payload.text;
      const actions = payload.actions || payload.choices || null;

      if (!state) {
        sendJson(res, 400, { error: 'Missing required field: "state"' });
        return;
      }

      if (actions && Array.isArray(actions) && actions.length > 0) {
        await engine.registerActions(actions);
      }

      if (!engine.actionCache || engine.actionCache.length === 0) {
        sendJson(res, 400, {
          error: 'No actions available. Provide "actions" array in payload or call POST /register-actions first.',
        });
        return;
      }

      const options = {
        mask: payload.mask || null,
        biases: payload.biases || null,
        temperature: payload.temperature ?? engine.config.temperature,
      };

      const decision = await engine.choice(state, options);

      sendJson(res, 200, {
        winner: decision.winner,
        selectedAction: decision.selectedAction,
        confidence: decision.confidence,
        ranking: decision.ranking,
        latencyMs: decision.latencyMs,
        cachedActionsCount: engine.actionCache.length,
      });
      return;
    }

    // 3. One-Shot Action Space Pre-caching: POST /register-actions
    if (req.method === 'POST' && url === '/register-actions') {
      const payload = await parseJsonBody(req);
      const actions = payload.actions || payload.choices;

      if (!Array.isArray(actions) || actions.length === 0) {
        sendJson(res, 400, { error: 'Payload must include a non-empty "actions" array.' });
        return;
      }

      const t0 = performance.now();
      const count = await engine.registerActions(actions);
      const latencyMs = performance.now() - t0;

      sendJson(res, 200, {
        status: 'cached',
        actionsCount: count,
        embeddingDim: engine.embeddingDim,
        latencyMs,
        message: `Cached ${count} candidate actions in RAM for O(1) query matching.`,
      });
      return;
    }

    // 4. O(1) Sub-Millisecond Decision: POST /step
    if (req.method === 'POST' && (url === '/step' || url === '/choice-fast')) {
      const payload = await parseJsonBody(req);
      const state = payload.state || payload.prompt || payload.query || payload.text;

      if (!state) {
        sendJson(res, 400, { error: 'Missing required field: "state"' });
        return;
      }

      if (!engine.actionCache || engine.actionCache.length === 0) {
        sendJson(res, 400, {
          error: 'Action matrix is empty. Register actions first via POST /register-actions.',
        });
        return;
      }

      const options = {
        mask: payload.mask || null,
        biases: payload.biases || null,
        temperature: payload.temperature ?? engine.config.temperature,
      };

      const decision = await engine.choice(state, options);

      sendJson(res, 200, {
        winner: decision.winner,
        selectedAction: decision.selectedAction,
        confidence: decision.confidence,
        ranking: decision.ranking,
        latencyMs: decision.latencyMs,
      });
      return;
    }

    // 5. Reset Actions / Stats: POST /reset
    if (req.method === 'POST' && url === '/reset') {
      engine.actionCache = [];
      engine.actionEmbeddings = null;
      sendJson(res, 200, { status: 'reset', message: 'Action matrix cleared.' });
      return;
    }

    // 404
    sendJson(res, 404, { error: `Endpoint "${url}" not found.` });
  } catch (err) {
    console.error(`[error] Request to ${url} failed:`, err);
    sendJson(res, 500, { error: err.message || 'Internal Server Error' });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`\n🚀 S1stem Server listening at http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
  console.log('   Ready for low-latency System 1 discrete agent decisions.\n');
});

// Graceful shutdown
function shutdown() {
  console.log('\n[shutdown] Terminating S1stem server...');
  server.close(() => {
    console.log('[shutdown] Server closed.');
    process.exit(0);
  });
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
