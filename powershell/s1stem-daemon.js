/**
 * S1stem Background Daemon for PowerShell
 * ========================================
 * High-performance lightweight resident service listening on 127.0.0.1:9191.
 * Holds pre-cached S1stem neural embeddings in RAM to serve sub-15ms resolutions
 * directly to PowerShell sessions.
 */

import http from 'http';
import { PowerShellS1stemResolver } from './s1stem-resolver.js';

const PORT = parseInt(process.env.S1STEM_PORT || '9191', 10);
const HOST = '127.0.0.1';

const resolver = new PowerShellS1stemResolver();

console.log('[s1stem-daemon] Initializing neural action matrix in RAM...');
await resolver.init();
console.log(`[s1stem-daemon] Neural weights cached (${resolver.patterns.length} patterns ready).`);

const server = http.createServer(async (req, res) => {
  // CORS & headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = req.url.split('?')[0];

  if (req.method === 'GET' && (url === '/' || url === '/health')) {
    res.writeHead(200);
    res.end(JSON.stringify({
      status: 'ready',
      version: '1.1.0',
      pid: process.pid,
      patternsCount: resolver.patterns.length,
      uptime: process.uptime()
    }));
    return;
  }

  if (req.method === 'POST' && url === '/shutdown') {
    res.writeHead(200);
    res.end(JSON.stringify({ message: 'Shutting down S1stem daemon.' }));
    console.log('[s1stem-daemon] Graceful shutdown requested.');
    setTimeout(() => {
      server.close(() => process.exit(0));
    }, 100);
    return;
  }

  if (req.method === 'POST' && url === '/resolve') {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      // Protect against gigantic payloads
      if (body.length > 50000) {
        req.destroy();
      }
    });

    req.on('end', async () => {
      try {
        let prompt = '';
        try {
          const parsed = JSON.parse(body);
          prompt = parsed.prompt || parsed.query || parsed.text || '';
        } catch {
          prompt = body.trim();
        }

        if (!prompt) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'Missing prompt in request body.' }));
          return;
        }

        const result = await resolver.resolve(prompt);
        res.writeHead(200);
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error('[s1stem-daemon] Resolution error:', err);
        res.writeHead(500);
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end(JSON.stringify({ error: 'Endpoint not found' }));
});

server.listen(PORT, HOST, () => {
  console.log(`[s1stem-daemon] Running at http://${HOST}:${PORT} (PID: ${process.pid})`);
});

process.on('SIGINT', () => {
  console.log('[s1stem-daemon] Terminating...');
  server.close(() => process.exit(0));
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});
