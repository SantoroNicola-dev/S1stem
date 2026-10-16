/**
 * S1stem Node.js / JavaScript Client Example
 * ==========================================
 * Demonstrates frictionless discrete decision resolution from Node.js,
 * Next.js, Electron, or browser apps using standard fetch.
 */

const S1STEM_URL = process.env.S1STEM_URL || 'http://localhost:8000';

async function main() {
  console.log(`Connecting to S1stem at ${S1STEM_URL}...`);

  // 1. Health check
  const health = await fetch(`${S1STEM_URL}/health`).then((r) => r.json());
  console.log(`✓ S1stem Online: ${health.engine} (dim: ${health.embeddingDim}, offline: ${health.isOfflineModel})`);

  // 2. Pre-cache tool candidates (useful for autonomous agents)
  console.log('\n[1] Registering tool candidate actions...');
  const tools = [
    { id: 'TOOL_SQL_QUERY',  label: 'sql_query',  description: 'Execute SQL statement on customer database.' },
    { id: 'TOOL_SEND_EMAIL', label: 'send_email', description: 'Draft and send transactional email to customer.' },
    { id: 'TOOL_WEB_SEARCH', label: 'web_search', description: 'Search web documentation for external technical facts.' },
    { id: 'TOOL_CALCULATOR', label: 'calculator', description: 'Compute numerical calculations and math.' },
  ];

  const regRes = await fetch(`${S1STEM_URL}/register-actions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actions: tools }),
  }).then((r) => r.json());
  console.log(`✓ Pre-cached ${regRes.actionsCount} tools in ${regRes.latencyMs.toFixed(1)} ms.`);

  // 3. Fast O(1) Tool Dispatching (< 1 ms matrix query)
  console.log('\n[2] Executing fast tool dispatching query...');
  const userRequest = 'Find the top 10 users who signed up last week with active subscriptions';
  
  const decision = await fetch(`${S1STEM_URL}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state: userRequest }),
  }).then((r) => r.json());

  console.log(`\nUser Request: "${userRequest}"`);
  console.log(`Selected Tool: ${decision.winner} (${(decision.confidence * 100).toFixed(1)}% confidence)`);
  console.log(`Latency:       ${decision.latencyMs.toFixed(3)} ms`);
  console.log('Tool Ranking:');
  decision.ranking.forEach((r) => {
    console.log(`  [Rank ${r.rank}] ${r.id.padEnd(16)} ${(r.probability * 100).toFixed(1)}%`);
  });
}

main().catch(console.error);
