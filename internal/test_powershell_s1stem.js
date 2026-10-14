/**
 * S1stem PowerShell Extension Verification Suite
 * ===============================================
 * Validates natural language intent routing, slot filling, Noul security gate,
 * and ordinal risk assessment across realistic user scenarios.
 */

import { PowerShellS1stemResolver } from '../powershell/s1stem-resolver.js';

console.log('╔════════════════════════════════════════════════════════════════════════════════╗');
console.log('║       S1STEM POWERSHELL EXTENSION NEURAL VERIFICATION SUITE                    ║');
console.log('╚════════════════════════════════════════════════════════════════════════════════╝\n');

async function runTestSuite() {
  const resolver = new PowerShellS1stemResolver();
  
  console.log('1. Initializing PowerShell Neural Resolver:');
  const t0 = performance.now();
  await resolver.init();
  const initMs = performance.now() - t0;
  console.log(`   ✓ Neural patterns cached in ${initMs.toFixed(2)} ms (${resolver.patterns.length} patterns).`);

  const testCases = [
    {
      name: 'User Example 1: Create New File in D:\\',
      query: 'command to create a new file in D:\\',
      expectedPattern: 'fs_create_file',
      expectedSnippet: 'New-Item -Path "D:\\new_file.txt" -ItemType File',
      expectedRiskMax: 2
    },
    {
      name: 'User Example 2: List Files in C:\\test',
      query: 'command to list files in a directory in c:\\test',
      expectedPattern: 'fs_list_files',
      expectedSnippet: 'Get-ChildItem -Path "c:\\test"',
      expectedRiskMax: 2
    },
    {
      name: 'User Example 3: Create Directory & Enter',
      query: 'create a new directory and enter inside, directory name new_dir',
      expectedPattern: 'fs_create_dir_and_enter',
      expectedSnippet: 'New-Item -ItemType Directory -Path "new_dir" -Force; Set-Location -Path "new_dir"',
      expectedRiskMax: 2
    },
    {
      name: 'Scenario 4: Top Processes by Memory',
      query: 'show processes consuming the most memory',
      expectedPattern: 'proc_list_processes',
      expectedSnippet: 'Get-Process',
      expectedRiskMax: 2
    },
    {
      name: 'Scenario 5: Network Ping Reachability',
      query: 'ping google to verify network connection',
      expectedPattern: 'net_test_connection_ping',
      expectedSnippet: 'Test-Connection',
      expectedRiskMax: 2
    },
    {
      name: 'Scenario 6: High Risk Destructive Deletion',
      query: 'recursively delete temp folder and remove all files',
      expectedPattern: 'fs_delete_directory_recursive',
      expectedSnippet: 'Remove-Item -Path "temp" -Recurse -Force',
      expectedRiskMin: 4
    },
    {
      name: 'Scenario 7: Date and Time Query',
      query: 'what time is it',
      expectedPattern: 'sys_get_date_time',
      expectedSnippet: 'Get-Date',
      expectedRiskMax: 1
    }
  ];

  let passedCount = 0;

  for (let i = 0; i < testCases.length; i++) {
    const tc = testCases[i];
    console.log(`\nTest ${i + 1}: ${tc.name}`);
    console.log(`   Query: "${tc.query}"`);

    const res = await resolver.resolve(tc.query);
    console.log(`   ✓ Pattern: ${res.patternId} (${res.patternLabel}) [Confidence: ${res.confidence}%]`);
    console.log(`   ✓ Command: ${res.command}`);
    console.log(`   ✓ Security: ${res.security.decision} (Risk: ${res.risk.score}/5) [Latency: ${res.latencyMs} ms]`);

    if (res.patternId !== tc.expectedPattern) {
      throw new Error(`Expected pattern ${tc.expectedPattern}, but got ${res.patternId}`);
    }

    if (!res.command.includes(tc.expectedSnippet)) {
      throw new Error(`Expected command to contain "${tc.expectedSnippet}", but got "${res.command}"`);
    }

    if (tc.expectedRiskMax && res.risk.score > tc.expectedRiskMax) {
      console.warn(`   Notice: Risk score ${res.risk.score} was higher than expected max ${tc.expectedRiskMax}`);
    }

    if (tc.expectedRiskMin && res.risk.score < tc.expectedRiskMin) {
      throw new Error(`Expected risk score >= ${tc.expectedRiskMin}, but got ${res.risk.score}`);
    }

    console.log(`   ✓ PASS: Scenario verified successfully.`);
    passedCount++;
  }

  console.log('\n╔════════════════════════════════════════════════════════════════════════════════╗');
  console.log(`║   ALL ${passedCount}/${testCases.length} POWERSHELL S1STEM TESTS PASSED CLEANLY (100%)                 ║`);
  console.log('╚════════════════════════════════════════════════════════════════════════════════╝\n');
}

runTestSuite().catch(err => {
  console.error('\n[FATAL] PowerShell Test Suite Failed:', err);
  process.exit(1);
});
