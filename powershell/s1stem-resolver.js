/**
 * S1stem PowerShell Intent & Command Resolver
 * ============================================
 * Resolves natural language user queries into executable PowerShell commands
 * using S1stem Two-Tower CLM neural decision engine, slot-filling, Noul security
 * gate and ordinal risk scoring.
 * 
 * Supports both curated idiomatic templates AND the complete catalog of 1,580+
 * authentic PowerShell cmdlets (One-shot cached in RAM).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { S1stemEngine } from '../core/s1stem.js';
import { POWERSHELL_PATTERNS, extractSlots } from './powershell-catalog.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getCmdletDefaultRisk(verb) {
  switch (verb) {
    case 'Remove':
    case 'Clear':
      return 4;
    case 'Stop':
    case 'Restart':
    case 'Disable':
    case 'Uninstall':
      return 3;
    case 'Set':
    case 'New':
    case 'Move':
    case 'Rename':
    case 'Install':
    case 'Add':
      return 2;
    default:
      return 1;
  }
}

function generateCmdletTemplate(cmdlet, slots) {
  const name = cmdlet.name;
  const p = slots || {};

  switch (name) {
    case 'Get-FileHash':
      return `Get-FileHash -Path "${p.path || p.fileName || 'file.txt'}" -Algorithm SHA256`;
    case 'Compress-Archive':
      return `Compress-Archive -Path "${p.path || p.dirName || '.'}" -DestinationPath "${p.zipName || 'archive.zip'}" -Force`;
    case 'Expand-Archive':
      return `Expand-Archive -Path "${p.path || p.fileName || 'archive.zip'}" -DestinationPath "${p.targetDir || '.'}" -Force`;
    case 'Get-Service':
      return `Get-Service | Where-Object { $_.Status -eq 'Running' } | Select-Object -First 25`;
    case 'Test-Path':
      return `Test-Path -Path "${p.path || p.dirName || '.'}"`;
    case 'Get-Random':
      return `Get-Random -Minimum 1 -Maximum 100`;
    case 'Get-Clipboard':
      return `Get-Clipboard`;
    case 'Set-Clipboard':
      return `Set-Clipboard -Value "${p.value || 'text'}"`;
    case 'ConvertFrom-Json':
      return `Get-Content "${p.fileName || 'data.json'}" | ConvertFrom-Json`;
    case 'ConvertTo-Json':
      return `$data | ConvertTo-Json -Depth 4`;
    case 'Export-Csv':
      return `$data | Export-Csv -Path "${p.fileName || 'export.csv'}" -NoTypeInformation`;
    case 'Import-Csv':
      return `Import-Csv -Path "${p.fileName || 'data.csv'}"`;
    case 'Get-ScheduledTask':
      return `Get-ScheduledTask | Select-Object TaskName, State | Format-Table -AutoSize`;
    case 'Get-Volume':
      return `Get-Volume | Format-Table DriveLetter, FileSystemLabel, SizeRemaining, Size -AutoSize`;
    default:
      if (cmdlet.verb === 'Get') {
        return `${name} | Select-Object -First 25`;
      }
      if (cmdlet.verb === 'Test') {
        return `${name} -Target "${p.host || p.path || 'localhost'}"`;
      }
      if (cmdlet.verb === 'Start') {
        return `${name} -Name "${p.name || 'service'}"`;
      }
      if (cmdlet.verb === 'Stop') {
        return `${name} -Name "${p.name || 'service'}" -Force`;
      }
      if (cmdlet.verb === 'New') {
        return `${name} -Name "${p.name || 'new_item'}"`;
      }
      return `${name}`;
  }
}

export class PowerShellS1stemResolver {
  constructor(config = {}) {
    this.config = config;
    this.engine = null;
    this.isInitialized = false;
    this.patterns = [...POWERSHELL_PATTERNS];
    this.patternMap = new Map();
  }

  async init() {
    if (this.isInitialized) return;

    // 1. Populate Pattern Map with Curated Patterns
    for (const pat of POWERSHELL_PATTERNS) {
      this.patternMap.set(pat.id, pat);
    }

    // 2. Load Full PowerShell Cmdlet Catalog (pwsh-cmdlets.json)
    const jsonPath = path.join(__dirname, 'data', 'pwsh-cmdlets.json');
    if (fs.existsSync(jsonPath)) {
      try {
        const cmdlets = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        for (const c of cmdlets) {
          if (!this.patternMap.has(c.name)) {
            const cmdletPattern = {
              id: c.name,
              label: c.name,
              description: c.description || `${c.name} - ${c.verb} ${c.noun} ${c.module || 'PowerShell'} cmdlet`,
              template: (slots) => generateCmdletTemplate(c, slots),
              explanation: (slots) => `Executes cmdlet ${c.name} (${c.verb} ${c.noun}, module ${c.module || 'PowerShell'}).`,
              defaultRisk: getCmdletDefaultRisk(c.verb)
            };
            this.patterns.push(cmdletPattern);
            this.patternMap.set(c.name, cmdletPattern);
          }
        }
      } catch (err) {
        console.warn('[s1stem-resolver] Could not load pwsh-cmdlets.json:', err.message);
      }
    }

    this.engine = new S1stemEngine({
      temperature: 0.035,
      autoLoadModel: true,
      ...this.config
    });

    // Register all actions into Action Matrix A (One-Shot)
    const actionDescriptors = this.patterns.map(p => ({
      id: p.id,
      label: p.label,
      description: p.description
    }));

    await this.engine.registerActions(actionDescriptors);
    this.isInitialized = true;
  }

  /**
   * Resolves natural language user query into concrete PowerShell command
   * 
   * @param {string} userPrompt - e.g. "command to create a new file in D:\"
   * @returns {Promise<Object>} Resolved command, explanation, security & risk
   */
  async resolve(userPrompt) {
    if (!this.isInitialized) {
      await this.init();
    }

    const tStart = performance.now();
    const prompt = String(userPrompt || '').trim();
    if (!prompt) {
      throw new Error('Prompt cannot be empty.');
    }

    // 1. Two-Tower CLM Intent Matching across the full action space
    const decision = await this.engine.choice(prompt);
    const pattern = this.patternMap.get(decision.winner) || this.patterns[0];

    // 2. Intelligent Slot-Filling
    const slots = extractSlots(prompt);

    // If intent is "create file" and user gave path like "D:\" but no file name, ensure default name
    if (pattern.id === 'fs_create_file') {
      if (!slots.fileName && !slots.path?.match(/\.[a-zA-Z0-9]+$/)) {
        slots.fileName = 'new_file.txt';
      }
    }

    // If intent is "create dir and enter" or "create dir", ensure dirName
    if (pattern.id === 'fs_create_dir_and_enter' || pattern.id === 'fs_create_directory') {
      if (!slots.dirName && !slots.path) {
        slots.dirName = 'new_dir';
      }
    }

    // 3. Generate Concrete PowerShell Command
    const command = pattern.template(slots);
    const explanation = pattern.explanation(slots);

    // 4. Calibrated Security Gate (Noul)
    const securityGate = await this.engine.noul(
      `${prompt} -> Executing: ${command}`,
      'Is this command potentially dangerous, destructive, deleting files, killing processes, or modifying system state?',
      { labels: ['ALLOW', 'FLAG'] }
    );

    // 5. Ordinal Impact Risk Rating (Score 1-5)
    const riskAssessment = await this.engine.score(
      `${prompt} -> ${command}`,
      'Operational Execution Risk and Destruction Level',
      { min: 1, max: 5 }
    );

    const totalLatencyMs = performance.now() - tStart;

    return {
      command,
      explanation,
      patternId: pattern.id,
      patternLabel: pattern.label,
      confidence: Number((decision.confidence * 100).toFixed(1)),
      margin: Number(decision.margin.toFixed(3)),
      slots,
      security: {
        decision: securityGate.decision,
        confidence: Number((securityGate.confidence * 100).toFixed(1)),
        isFlagged: securityGate.decision === 'FLAG'
      },
      risk: {
        score: pattern.defaultRisk || 1,
        expectedScore: riskAssessment.expectedScore
      },
      latencyMs: Number(totalLatencyMs.toFixed(2)),
      totalPatternsCount: this.patterns.length,
      alternatives: (decision.ranking || []).slice(1, 4).map(r => ({
        id: r.id,
        label: this.patternMap.get(r.id)?.label || r.id,
        probability: Number((r.probability * 100).toFixed(1))
      }))
    };
  }
}
