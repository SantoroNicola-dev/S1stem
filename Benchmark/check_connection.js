/**
 * Pre-Flight Connectivity Check for Competitor Models on Azure Foundry
 * =====================================================================
 * Verifies connectivity, authentication, and baseline latency for:
 *   1. S1stem local neural engine (in-process)
 *   2. Microsoft-Decision-1 endpoint (/providers/microsoft/v1/systemone)
 *   3. gpt-6.1-sol endpoint (/openai/v1/chat/completions)
 * 
 * Usage:
 *   node Benchmark/check_connection.js
 * 
 * License: Apache-2.0
 */

import {
  loadBenchmarkConfig,
  S1stemAdapter,
  MicrosoftDecision1Adapter,
  Gpt6LunaDecisionsAdapter,
  Gpt61SolAdapter
} from './common.js';

async function main() {
  console.log('╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗');
  console.log('║   COMPETITOR CONNECTIVITY & PRE-FLIGHT VERIFICATION (FOUNDRY & OPENAI DECISIONS)                        ║');
  console.log('╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n');

  const config = loadBenchmarkConfig();
  const af = config.azure_foundry;
  const oai = config.openai;

  const maskedAfKey = af.api_key && af.is_configured ? '••••••••' + af.api_key.slice(-4) : '[NOT CONFIGURED - PLACEHOLDER]';
  const maskedOaiKey = oai.api_key && oai.is_configured ? '••••••••' + oai.api_key.slice(-4) : '[NOT CONFIGURED - PLACEHOLDER]';
  
  console.log(`Azure Foundry:  ${af.endpoint || '[NOT SET]'} | Key: ${maskedAfKey}`);
  console.log(`OpenAI API:     ${oai.endpoint || '[NOT SET]'} | Key: ${maskedOaiKey}\n`);

  const pingCandidates = [
    { id: 'ok', label: 'Service operational' },
    { id: 'offline', label: 'Service degraded' }
  ];
  const pingObservation = 'System connectivity ping verification.';

  // 1. S1stem Local In-Process Check
  console.log('▶ [1/4] Verifying S1stem Local CLM Decision Engine:');
  try {
    const s1 = new S1stemAdapter(config.s1stem);
    const res = await s1.choice(pingObservation, pingCandidates);
    console.log(`  ✓ S1stem Local Engine: ONLINE | Winner: "${res.winner}" | Latency: ${res.latencyMs} ms | Cost: $0.00\n`);
  } catch (err) {
    console.error(`  ✗ S1stem Local Engine: ERROR — ${err.message}\n`);
  }

  // 2. Microsoft-Decision-1 Check
  console.log('▶ [2/4] Verifying Microsoft-Decision-1 (Azure Foundry):');
  if (af.is_configured) {
    try {
      const d1 = new MicrosoftDecision1Adapter({
        endpoint: af.endpoint,
        apiKey: af.api_key,
        deploymentName: af.microsoft_decision_1.deployment_name,
        timeoutMs: config.benchmark_settings.timeout_ms
      });
      const res = await d1.choice(pingObservation, pingCandidates);
      if (res.isError) {
        console.error(`  ✗ Microsoft-Decision-1: FAILED — ${res.error}\n`);
      } else {
        console.log(`  ✓ Microsoft-Decision-1: ONLINE | Winner: "${res.winner}" | Latency: ${res.latencyMs} ms | Cost: ~$${res.costUsd.toFixed(6)}\n`);
      }
    } catch (err) {
      console.error(`  ✗ Microsoft-Decision-1: FAILED — ${err.message}\n`);
    }
  } else {
    console.log('  ⏭ Skipped (Set active API key in Benchmark/config.json or AZURE_FOUNDRY_API_KEY)\n');
  }

  // 3. gpt-6-luna Decisions API Check
  const lunaConf = config.gpt_6_luna;
  console.log(`▶ [3/4] Verifying gpt-6-luna (Decisions API @ ${lunaConf.path}):`);
  if (lunaConf.is_configured) {
    try {
      const luna = new Gpt6LunaDecisionsAdapter({
        endpoint: lunaConf.endpoint,
        apiKey: lunaConf.api_key,
        model: lunaConf.deployment_name,
        path: lunaConf.path,
        timeoutMs: config.benchmark_settings.timeout_ms
      });
      const res = await luna.choice(pingObservation, pingCandidates);
      if (res.isError) {
        console.error(`  ✗ gpt-6-luna: FAILED — ${res.error}\n`);
      } else {
        console.log(`  ✓ gpt-6-luna: ONLINE | Winner: "${res.winner}" | Confidence: ${res.confidence} | Latency: ${res.latencyMs} ms | Cost: ~$${res.costUsd.toFixed(6)}\n`);
      }
    } catch (err) {
      console.error(`  ✗ gpt-6-luna: FAILED — ${err.message}\n`);
    }
  } else {
    console.log('  ⏭ Skipped (Set active API key in Benchmark/config.json under "azure_foundry" or "openai")\n');
  }

  // 4. gpt-6.1-sol Check
  console.log('▶ [4/4] Verifying gpt-6.1-sol (Azure Foundry):');
  if (af.is_configured) {
    try {
      const gpt = new Gpt61SolAdapter({
        endpoint: af.endpoint,
        apiKey: af.api_key,
        deploymentName: af.gpt_6_1_sol.deployment_name,
        timeoutMs: config.benchmark_settings.timeout_ms
      });
      const res = await gpt.choice(pingObservation, pingCandidates);
      if (res.isError) {
        console.error(`  ✗ gpt-6.1-sol: FAILED — ${res.error}\n`);
      } else {
        console.log(`  ✓ gpt-6.1-sol: ONLINE | Winner: "${res.winner}" | Latency: ${res.latencyMs} ms | Cost: ~$${res.costUsd.toFixed(6)}\n`);
      }
    } catch (err) {
      console.error(`  ✗ gpt-6.1-sol: FAILED — ${err.message}\n`);
    }
  } else {
    console.log('  ⏭ Skipped (Set active API key in Benchmark/config.json or AZURE_FOUNDRY_API_KEY)\n');
  }

  console.log('Pre-flight verification complete.');
}

main().catch(console.error);

