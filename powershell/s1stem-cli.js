/**
 * S1stem CLI Runner & Fallback for PowerShell
 * ============================================
 * Dispatches query to local daemon http://127.0.0.1:9191/resolve for sub-15ms speed;
 * falls back automatically to direct in-process resolution if daemon is not running.
 */

import { PowerShellS1stemResolver } from './s1stem-resolver.js';

const PORT = parseInt(process.env.S1STEM_PORT || '9191', 10);
const HOST = '127.0.0.1';

const args = process.argv.slice(2);
const isHuman = args.includes('--human') || args.includes('-h');
const cleanArgs = args.filter(a => a !== '--human' && a !== '-h');
const prompt = cleanArgs.join(' ').trim();

if (!prompt) {
  console.error(JSON.stringify({
    error: 'Usage: node s1stem-cli.js "<your command request in natural language>"'
  }));
  process.exit(1);
}

async function tryDaemon() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 250);

    const res = await fetch(`http://${HOST}:${PORT}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
      signal: controller.signal
    });

    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return { data, source: 'daemon' };
  } catch {
    return null;
  }
}

async function runDirect() {
  const resolver = new PowerShellS1stemResolver();
  await resolver.init();
  const data = await resolver.resolve(prompt);
  return { data, source: 'in-process' };
}

try {
  let outcome = await tryDaemon();
  if (!outcome) {
    outcome = await runDirect();
  }

  const { data, source } = outcome;
  data.source = source;

  if (isHuman) {
    console.log(`\nCommand     : ${data.command}`);
    console.log(`Explanation : ${data.explanation}`);
    console.log(`Pattern     : ${data.patternLabel} (${data.confidence}% conf)`);
    console.log(`Security    : ${data.security.decision} (Risk: ${data.risk.score}/5)`);
    console.log(`Latency     : ${data.latencyMs} ms [${source}]\n`);
  } else {
    console.log(JSON.stringify(data));
  }
} catch (err) {
  console.error(JSON.stringify({ error: err.message }));
  process.exit(1);
}
