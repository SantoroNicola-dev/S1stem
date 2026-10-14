/**
 * Agent Dispatcher // S1stem-CDR Interactive Testbed
 * Real Client-Side Two-Tower CLM Decision & Security Arbiter
 * 
 * Zero Mocks: All decisions, security gates, and risk scores are computed
 * live by S1stemEngine in local memory.
 */

import { S1stemEngine } from '../../core/s1stem.js';

// Default Real-World Action Library (28 Authentic DevOps / Cloud / Security Tools)
export const DEFAULT_TOOLS = [
  // DevOps & System Shell
  {
    id: 'run_command',
    label: 'Execute Shell Command',
    description: 'Execute a non-destructive CLI command in system terminal, check stdout/stderr, or inspect environment variables.',
    icon: '💻',
    category: 'system'
  },
  {
    id: 'kill_background_process',
    label: 'Kill Background Process',
    description: 'Terminate a running process or task by PID, kill zombie daemons, or abort stuck background jobs.',
    icon: '⏹',
    category: 'system'
  },
  {
    id: 'package_install_dependency',
    label: 'Install Package Dependency',
    description: 'Install software libraries, npm dependencies, python pip packages, or system package manager modules.',
    icon: '📦',
    category: 'system'
  },
  {
    id: 'read_system_logs',
    label: 'Read System Logs',
    description: 'Inspect application runtime logs, syslog, journalctl daemon traces, or container error outputs.',
    icon: '📜',
    category: 'system'
  },

  // Git & Version Control
  {
    id: 'git_status_diff',
    label: 'Git Status & Diff',
    description: 'Inspect repository status, view modified files, check branch tracking, and examine unstaged line differences.',
    icon: '🌿',
    category: 'git'
  },
  {
    id: 'git_commit_push',
    label: 'Git Commit and Push',
    description: 'Stage changed files, generate commit with descriptive log message, and push branch commits to upstream remote.',
    icon: '🚀',
    category: 'git'
  },
  {
    id: 'git_force_push',
    label: 'Git Force Push (Destructive)',
    description: 'Force overwrite remote git commit history using git push --force or lease, overwriting upstream collaboration tree.',
    icon: '⚠️',
    category: 'git'
  },

  // Database & Storage
  {
    id: 'database_query_readonly',
    label: 'Database Read Query (SELECT)',
    description: 'Execute read-only SQL SELECT queries to fetch table rows, inspect schemas, or verify data counts without writes.',
    icon: '🔍',
    category: 'database'
  },
  {
    id: 'database_migration',
    label: 'Run Database Migration',
    description: 'Apply schema migrations, add table columns, alter foreign keys, or update relational schema definitions.',
    icon: '🗄️',
    category: 'database'
  },
  {
    id: 'database_drop_table',
    label: 'Drop Database Table (Destructive)',
    description: 'Permanently drop, truncate, or wipe SQL database tables and purge all stored persistent records.',
    icon: '💥',
    category: 'database'
  },
  {
    id: 'data_backup_snapshot',
    label: 'Create Backup Snapshot',
    description: 'Trigger point-in-time database snapshot, dump volume to S3, or create encrypted backup archive.',
    icon: '💾',
    category: 'database'
  },

  // Cloud & Containers
  {
    id: 'aws_deploy_lambda',
    label: 'Deploy AWS Lambda Function',
    description: 'Build and deploy serverless function bundle to AWS Lambda or Cloudflare Workers edge environment.',
    icon: '☁️',
    category: 'cloud'
  },
  {
    id: 'kubernetes_scale_pod',
    label: 'Scale Kubernetes Deployment',
    description: 'Adjust replica count for Kubernetes pods, autoscale deployment, or resize cluster worker nodes.',
    icon: '☸️',
    category: 'cloud'
  },
  {
    id: 'docker_build_image',
    label: 'Build Docker Container Image',
    description: 'Build OCI Docker image from Dockerfile, tag release version, and verify container build cache layers.',
    icon: '🐳',
    category: 'cloud'
  },
  {
    id: 'cloud_storage_upload',
    label: 'Upload Asset to Cloud Storage',
    description: 'Upload files, media assets, dataset archives, or build artifacts to cloud object storage bucket.',
    icon: '📤',
    category: 'cloud'
  },

  // Security, IAM & Secrets
  {
    id: 'security_scan_vulnerability',
    label: 'Scan Vulnerabilities & CVEs',
    description: 'Run static application security testing (SAST), dependency vulnerability scan, or audit CVEs.',
    icon: '🛡️',
    category: 'security'
  },
  {
    id: 'security_firewall_block',
    label: 'Block IP in Firewall',
    description: 'Add malicious IP address or subnet to firewall blocklist or cloud WAF rate-limiting rule.',
    icon: '🚫',
    category: 'security'
  },
  {
    id: 'read_vault_secret',
    label: 'Retrieve Secret from Vault',
    description: 'Fetch encrypted credentials, database tokens, or TLS certificates from HashiCorp Vault or AWS Secrets Manager.',
    icon: '🔑',
    category: 'security'
  },
  {
    id: 'rotate_api_keys',
    label: 'Rotate Production API Keys',
    description: 'Invalidate old authentication credentials, generate new cryptographic tokens, and distribute rotated secrets.',
    icon: '🔄',
    category: 'security'
  },
  {
    id: 'user_permission_grant',
    label: 'Grant Admin IAM Permission',
    description: 'Escalate privileges, grant administrative roles, or modify IAM security policies with elevated permissions.',
    icon: '👑',
    category: 'security'
  },

  // File System & Codebase
  {
    id: 'file_search_grep',
    label: 'Search Files and Grep Code',
    description: 'Search repository text patterns with ripgrep, locate files by glob pattern, or find symbol usages.',
    icon: '🔎',
    category: 'file'
  },
  {
    id: 'file_read_content',
    label: 'Read File Content',
    description: 'Open and inspect the text contents of a file on the local filesystem or repository source tree.',
    icon: '📄',
    category: 'file'
  },
  {
    id: 'file_write_edit',
    label: 'Edit or Write File',
    description: 'Apply non-destructive edits, refactor code, or write content to a designated file path.',
    icon: '✏️',
    category: 'file'
  },
  {
    id: 'file_rm_recursive',
    label: 'Delete Files Recursively (Destructive)',
    description: 'Permanently remove files or delete directories recursively (rm -rf) from filesystem.',
    icon: '🗑️',
    category: 'file'
  },

  // Network & Diagnostics
  {
    id: 'network_ping_diagnostic',
    label: 'Network Ping & DNS Diagnostic',
    description: 'Test network reachability, latency ping, DNS resolution lookup, or traceroute to host endpoint.',
    icon: '📡',
    category: 'network'
  },
  {
    id: 'http_rest_request',
    label: 'Send HTTP / REST Request',
    description: 'Dispatch HTTP GET/POST/PUT API request with JSON payload and inspect response headers.',
    icon: '🌐',
    category: 'network'
  },

  // External Services & Notifications
  {
    id: 'stripe_refund_charge',
    label: 'Issue Stripe Payment Refund',
    description: 'Process monetary refund to customer credit card via Stripe Billing API for disputed or canceled charges.',
    icon: '💳',
    category: 'billing'
  },
  {
    id: 'slack_post_alert',
    label: 'Post Alert to Slack Channel',
    description: 'Send incident notification, deployment summary, or team update to Slack webhook or Discord channel.',
    icon: '📢',
    category: 'notification'
  }
];

class DispatcherApp {
  constructor() {
    this.engine = null;
    this.tools = [...DEFAULT_TOOLS];
    this.history = [];
    this.historyIndex = -1;
    this.isProcessing = false;

    // DOM Elements
    this.terminalStream = document.getElementById('terminal-stream');
    this.terminalForm = document.getElementById('terminal-form');
    this.commandInput = document.getElementById('command-input');
    this.btnExecute = document.getElementById('btn-execute');
    this.btnClearTerm = document.getElementById('btn-clear-term');
    this.termStatusText = document.getElementById('term-status-text');
    this.termLivePulse = document.getElementById('term-live-pulse');

    // Header Badges
    this.modelStatusPill = document.getElementById('model-status-pill');
    this.hdrLatency = document.getElementById('hdr-latency');

    // Telemetry Panel Elements
    this.winnerIcon = document.getElementById('winner-icon');
    this.winnerToolId = document.getElementById('winner-tool-id');
    this.winnerToolDesc = document.getElementById('winner-tool-desc');
    this.winnerConf = document.getElementById('winner-conf');
    this.winnerMargin = document.getElementById('winner-margin');
    this.decisionLatencyBadge = document.getElementById('decision-latency-badge');

    // Security (Noul) Elements
    this.noulBadge = document.getElementById('noul-badge');
    this.noulVerdict = document.getElementById('noul-verdict');
    this.noulBarAllow = document.getElementById('noul-bar-allow');
    this.noulBarFlag = document.getElementById('noul-bar-flag');
    this.noulRationale = document.getElementById('noul-rationale');

    // Risk (Score) Elements
    this.riskScoreVal = document.getElementById('risk-score-val');
    this.riskDots = document.querySelectorAll('.risk-dot');
    this.riskLabelText = document.getElementById('risk-label-text');

    // Ranking Elements
    this.rankingBarsList = document.getElementById('ranking-bars-list');
    this.rankingActionCount = document.getElementById('ranking-action-count');

    // Modal Elements
    this.btnOpenCatalog = document.getElementById('btn-open-catalog');
    this.btnToolCount = document.getElementById('btn-tool-count');
    this.toolModal = document.getElementById('tool-modal');
    this.btnCloseModal = document.getElementById('btn-close-modal');
    this.modalToolCount = document.getElementById('modal-tool-count');
    this.modalToolList = document.getElementById('modal-tool-list');
    this.toolSearchInput = document.getElementById('tool-search-input');

    // Modal Tabs
    this.tabBtnView = document.getElementById('tab-btn-view');
    this.tabBtnAdd = document.getElementById('tab-btn-add');
    this.tabBtnJson = document.getElementById('tab-btn-json');
    this.tabPaneView = document.getElementById('tab-pane-view');
    this.tabPaneAdd = document.getElementById('tab-pane-add');
    this.tabPaneJson = document.getElementById('tab-pane-json');

    // Custom Tool Form
    this.customToolForm = document.getElementById('custom-tool-form');
    this.newToolId = document.getElementById('new-tool-id');
    this.newToolLabel = document.getElementById('new-tool-label');
    this.newToolDesc = document.getElementById('new-tool-desc');

    // JSON Import
    this.jsonImportTextarea = document.getElementById('json-import-textarea');
    this.btnApplyJsonImport = document.getElementById('btn-apply-json-import');
    this.btnResetDefaultTools = document.getElementById('btn-reset-default-tools');
  }

  async init() {
    this.setupEventListeners();
    this.renderWelcomeBanner();
    this.updateToolCountDisplay();
    this.renderModalToolCatalog();

    try {
      this.setStatus('INITIALIZING...', 'warn');
      if (this.modelStatusPill) this.modelStatusPill.textContent = 'LOADING MODEL...';

      const t0 = performance.now();
      this.engine = new S1stemEngine({
        temperature: 0.035,
        autoLoadModel: true
      });

      // Register and pre-vectorize action space
      await this.engine.registerActions(this.tools);
      const initDuration = (performance.now() - t0).toFixed(1);

      const device = (typeof navigator !== 'undefined' && navigator.gpu) ? 'WEBGPU' : 'CPU';
      this.setStatus('READY', 'live');
      if (this.modelStatusPill) {
        this.modelStatusPill.textContent = `LOCAL ${device}`;
      }

      this.appendTerminalLog(
        `[s1stem-core] Model weights ready (${device}). ${this.tools.length} action embeddings cached in ${initDuration} ms.`,
        'sys'
      );
      this.appendTerminalLog(
        'Type your natural language command below or type :help for interactive commands.',
        'sys'
      );
    } catch (err) {
      console.error('Failed to initialize S1stemEngine:', err);
      this.setStatus('ERROR', 'error');
      if (this.modelStatusPill) this.modelStatusPill.textContent = 'FAIL';
      this.appendTerminalLog(`[fatal] Initialization failed: ${err.message}`, 'error');
    }
  }

  setupEventListeners() {
    // Form submit
    this.terminalForm?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const raw = this.commandInput.value.trim();
      if (!raw || this.isProcessing) return;

      this.history.push(raw);
      this.historyIndex = this.history.length;
      this.commandInput.value = '';

      await this.handleInput(raw);
    });

    // Arrow keys history traversal
    this.commandInput?.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') {
        if (this.historyIndex > 0) {
          this.historyIndex--;
          this.commandInput.value = this.history[this.historyIndex] || '';
        }
        e.preventDefault();
      } else if (e.key === 'ArrowDown') {
        if (this.historyIndex < this.history.length - 1) {
          this.historyIndex++;
          this.commandInput.value = this.history[this.historyIndex] || '';
        } else {
          this.historyIndex = this.history.length;
          this.commandInput.value = '';
        }
        e.preventDefault();
      }
    });

    // Clear Terminal Button
    this.btnClearTerm?.addEventListener('click', () => {
      this.clearTerminal();
    });

    // Modal Open / Close
    this.btnOpenCatalog?.addEventListener('click', () => {
      this.openModal();
    });

    this.btnCloseModal?.addEventListener('click', () => {
      this.closeModal();
    });

    this.toolModal?.addEventListener('click', (e) => {
      if (e.target === this.toolModal) this.closeModal();
    });

    // Modal Tabs
    this.tabBtnView?.addEventListener('click', () => this.switchTab('view'));
    this.tabBtnAdd?.addEventListener('click', () => this.switchTab('add'));
    this.tabBtnJson?.addEventListener('click', () => this.switchTab('json'));

    // Search filter in modal
    this.toolSearchInput?.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      this.renderModalToolCatalog(q);
    });

    // Custom tool form submission
    this.customToolForm?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const id = this.newToolId.value.trim().toLowerCase().replace(/\s+/g, '_');
      const label = this.newToolLabel.value.trim();
      const desc = this.newToolDesc.value.trim();

      if (!id || !label || !desc) return;

      const newTool = {
        id,
        label,
        description: desc,
        icon: '⚡',
        category: 'custom'
      };

      // Check if duplicate ID
      const existingIdx = this.tools.findIndex(t => t.id === id);
      if (existingIdx >= 0) {
        this.tools[existingIdx] = newTool;
      } else {
        this.tools.push(newTool);
      }

      await this.revectorizeActionSpace();
      this.customToolForm.reset();
      this.switchTab('view');
      this.renderModalToolCatalog();
      this.appendTerminalLog(`[action-space] Registered custom tool: ${id} (${label})`, 'sys');
    });

    // Apply JSON import
    this.btnApplyJsonImport?.addEventListener('click', async () => {
      const rawJson = this.jsonImportTextarea.value.trim();
      if (!rawJson) return;

      try {
        const parsed = JSON.parse(rawJson);
        if (!Array.isArray(parsed) || parsed.length === 0) {
          alert('JSON must be a non-empty array of tools: [{ id, label, description }]');
          return;
        }

        const validTools = parsed.map(p => ({
          id: (p.id || p.name || 'tool_' + Math.random().toString(36).substring(2, 6)).toLowerCase().replace(/\s+/g, '_'),
          label: p.label || p.name || p.id,
          description: p.description || p.label || '',
          icon: p.icon || '⚙',
          category: p.category || 'imported'
        }));

        this.tools = validTools;
        await this.revectorizeActionSpace();
        this.switchTab('view');
        this.renderModalToolCatalog();
        this.appendTerminalLog(`[action-space] Imported ${validTools.length} tools from JSON manifest.`, 'sys');
      } catch (err) {
        alert('Invalid JSON manifest: ' + err.message);
      }
    });

    // Reset default tools
    this.btnResetDefaultTools?.addEventListener('click', async () => {
      this.tools = [...DEFAULT_TOOLS];
      await this.revectorizeActionSpace();
      this.switchTab('view');
      this.renderModalToolCatalog();
      this.appendTerminalLog('[action-space] Reset action catalog to 28 default tools.', 'sys');
    });
  }

  async revectorizeActionSpace() {
    this.updateToolCountDisplay();
    if (this.engine) {
      const t0 = performance.now();
      await this.engine.registerActions(this.tools);
      const dt = (performance.now() - t0).toFixed(1);
      this.appendTerminalLog(`[cache] Action matrix updated: ${this.tools.length} tools vectorized in ${dt} ms.`, 'sys');
    }
  }

  updateToolCountDisplay() {
    const count = this.tools.length;
    if (this.btnToolCount) this.btnToolCount.textContent = count;
    if (this.modalToolCount) this.modalToolCount.textContent = count;
    if (this.rankingActionCount) this.rankingActionCount.textContent = `${count} Actions`;
  }

  openModal() {
    if (!this.toolModal) return;
    this.toolModal.classList.add('open');
    this.toolModal.setAttribute('aria-hidden', 'false');
    this.renderModalToolCatalog();
  }

  closeModal() {
    if (!this.toolModal) return;
    this.toolModal.classList.remove('open');
    this.toolModal.setAttribute('aria-hidden', 'true');
  }

  switchTab(tabId) {
    [this.tabBtnView, this.tabBtnAdd, this.tabBtnJson].forEach(b => b?.classList.remove('active'));
    [this.tabPaneView, this.tabPaneAdd, this.tabPaneJson].forEach(p => p?.classList.remove('active'));

    if (tabId === 'view') {
      this.tabBtnView?.classList.add('active');
      this.tabPaneView?.classList.add('active');
    } else if (tabId === 'add') {
      this.tabBtnAdd?.classList.add('active');
      this.tabPaneAdd?.classList.add('active');
    } else if (tabId === 'json') {
      this.tabBtnJson?.classList.add('active');
      this.tabPaneJson?.classList.add('active');
    }
  }

  renderModalToolCatalog(query = '') {
    if (!this.modalToolList) return;
    const filtered = query
      ? this.tools.filter(t => t.id.includes(query) || t.label.toLowerCase().includes(query) || t.description.toLowerCase().includes(query))
      : this.tools;

    if (filtered.length === 0) {
      this.modalToolList.innerHTML = `<div class="placeholder-text">No tools match "${query}".</div>`;
      return;
    }

    this.modalToolList.innerHTML = filtered.map(t => `
      <div class="tool-item-card">
        <div class="tool-item-header">
          <span class="tool-item-id">${t.icon || '⚙'} ${t.id}</span>
          <span class="badge" style="font-size: 10px;">${t.label}</span>
        </div>
        <div class="tool-item-desc">${t.description}</div>
      </div>
    `).join('');
  }

  setStatus(text, type = 'live') {
    if (this.termStatusText) this.termStatusText.textContent = text;
    if (this.termLivePulse) {
      this.termLivePulse.className = 'status-indicator ' + type;
    }
  }

  renderWelcomeBanner() {
    if (!this.terminalStream) return;
    this.terminalStream.innerHTML = `
      <div class="term-welcome-box">
        <div><strong>S1stem Two-Tower CLM Decision Engine v1.1.0</strong></div>
        <div style="margin-top: 4px;">Architecture: Contrastive Dual Encoder | Latency Target: <span class="hl">&lt;10 ms</span> | Cost: <span class="hl">$0.000</span></div>
        <div style="margin-top: 4px;">Commands: <span class="hl">:help</span>, <span class="hl">:tools</span>, <span class="hl">:add-tool</span>, <span class="hl">:benchmark</span>, <span class="hl">:clear</span></div>
      </div>
    `;
  }

  clearTerminal() {
    if (!this.terminalStream) return;
    this.terminalStream.innerHTML = '';
    this.renderWelcomeBanner();
  }

  appendTerminalLog(text, type = 'info') {
    if (!this.terminalStream) return;
    const line = document.createElement('div');
    line.className = 'term-meta-muted';
    if (type === 'sys') line.style.color = '#58a6ff';
    if (type === 'error') line.style.color = '#ef4444';
    if (type === 'success') line.style.color = '#10b981';
    line.textContent = text;
    this.terminalStream.appendChild(line);
    this.terminalStream.scrollTop = this.terminalStream.scrollHeight;
  }

  async handleInput(command) {
    // Check REPL Meta-Commands
    if (command.startsWith(':')) {
      await this.handleMetaCommand(command.toLowerCase().trim());
      return;
    }

    if (!this.engine) {
      this.appendTerminalLog('[error] Engine is still initializing. Please wait.', 'error');
      return;
    }

    this.isProcessing = true;
    if (this.btnExecute) this.btnExecute.disabled = true;
    this.setStatus('DISPATCHING...', 'warn');

    // Create User Command Entry in Terminal
    const entryEl = document.createElement('div');
    entryEl.className = 'term-entry';
    entryEl.innerHTML = `
      <div class="term-cmd-line">
        <span class="prompt-tag">s1stem-agent&gt;</span>
        <span class="cmd-text">${this.escapeHtml(command)}</span>
      </div>
      <div class="term-meta-muted" style="margin-left: 12px;">Evaluating neural action matrix & safety gate...</div>
    `;
    this.terminalStream.appendChild(entryEl);
    this.terminalStream.scrollTop = this.terminalStream.scrollHeight;

    const tStart = performance.now();

    try {
      // Step 1: Real Tool Routing via S1stemEngine
      const decision = await this.engine.choice(command);

      // Step 2: Calibrated Security Verification via Noul
      const securityVerdict = await this.engine.noul(
        command,
        'Is this command potentially dangerous, destructive, modifying production state, or unauthorized?',
        { labels: ['ALLOW', 'FLAG'] }
      );

      // Step 3: Ordinal Risk Rating via Score 1–5
      const riskRating = await this.engine.score(
        command,
        'Execution Risk and Operational Impact Level',
        { min: 1, max: 5 }
      );

      const totalLatency = (performance.now() - tStart).toFixed(1);

      // Update Header Badges
      if (this.hdrLatency) this.hdrLatency.textContent = `${totalLatency} ms`;
      if (this.decisionLatencyBadge) this.decisionLatencyBadge.textContent = `${totalLatency} ms`;

      // Find tool metadata
      const winningTool = this.tools.find(t => t.id === decision.winner) || {
        id: decision.winner,
        label: decision.winner,
        description: 'Custom registered action',
        icon: '⚙'
      };

      // Append Terminal Output Card
      const isFlagged = securityVerdict.decision === 'FLAG';
      const outputCard = document.createElement('div');
      outputCard.className = `term-output-card ${isFlagged ? 'flagged' : ''}`;
      outputCard.innerHTML = `
        <div class="term-output-row">
          <div>
            <span class="term-tool-name">${winningTool.icon || '⚙'} ${winningTool.id}</span>
            <span style="color: #8b949e; font-size: 11px; margin-left: 6px;">(${winningTool.label})</span>
          </div>
          <div>
            <span class="${isFlagged ? 'term-badge-flag' : 'term-badge-allow'}">
              ${securityVerdict.decision} (${(securityVerdict.confidence * 100).toFixed(0)}%)
            </span>
          </div>
        </div>
        <div class="term-output-row" style="margin-top: 3px;">
          <span class="term-meta-muted">
            Conf: <strong>${(decision.confidence * 100).toFixed(1)}%</strong> | 
            Margin Δ: <strong>${decision.margin.toFixed(3)}</strong> | 
            Impact: <strong>${riskRating.score}/5</strong>
          </span>
          <span class="term-meta-muted">Resolved in <strong>${totalLatency} ms</strong> ($0.000)</span>
        </div>
      `;

      // Replace intermediate message
      entryEl.querySelector('.term-meta-muted').replaceWith(outputCard);
      this.terminalStream.scrollTop = this.terminalStream.scrollHeight;

      // Update Right Telemetry Pane
      this.updateTelemetry(winningTool, decision, securityVerdict, riskRating, totalLatency);

    } catch (err) {
      console.error('Dispatch error:', err);
      entryEl.querySelector('.term-meta-muted')?.replaceWith(
        document.createTextNode(`[error] Dispatch failed: ${err.message}`)
      );
    } finally {
      this.isProcessing = false;
      if (this.btnExecute) this.btnExecute.disabled = false;
      this.setStatus('READY', 'live');
      this.commandInput.focus();
    }
  }

  updateTelemetry(tool, decision, security, risk, latencyMs) {
    // Card 1: Winning Action
    if (this.winnerIcon) this.winnerIcon.textContent = tool.icon || '⚙';
    if (this.winnerToolId) this.winnerToolId.textContent = tool.id;
    if (this.winnerToolDesc) this.winnerToolDesc.textContent = tool.description || tool.label;
    if (this.winnerConf) this.winnerConf.textContent = `${(decision.confidence * 100).toFixed(1)} %`;
    if (this.winnerMargin) this.winnerMargin.textContent = decision.margin.toFixed(3);

    // Card 2: Security Gate (Noul)
    const isFlagged = security.decision === 'FLAG';
    const allowPct = ((security.probability['ALLOW'] ?? 0.5) * 100).toFixed(0);
    const flagPct = ((security.probability['FLAG'] ?? 0.5) * 100).toFixed(0);

    if (this.noulBadge) {
      this.noulBadge.textContent = security.decision;
      this.noulBadge.className = isFlagged ? 'badge badge-red' : 'badge badge-emerald';
    }

    if (this.noulVerdict) {
      this.noulVerdict.textContent = isFlagged ? 'FLAGGED (HIGH RISK)' : 'AUTHORIZED (ALLOW)';
      this.noulVerdict.className = `verdict-val ${isFlagged ? 'flagged' : 'allow'}`;
    }

    if (this.noulBarAllow) {
      this.noulBarAllow.style.width = `${allowPct}%`;
      this.noulBarAllow.textContent = `ALLOW ${allowPct}%`;
    }

    if (this.noulBarFlag) {
      this.noulBarFlag.style.width = `${flagPct}%`;
      this.noulBarFlag.textContent = `FLAG ${flagPct}%`;
    }

    if (this.noulRationale) {
      this.noulRationale.textContent = isFlagged
        ? `Noul polar analysis flagged this operation (confidence ${(security.confidence * 100).toFixed(1)}%). Elevated scrutiny required before autonomous execution.`
        : `Noul polar analysis verified this operation as safe and routine (confidence ${(security.confidence * 100).toFixed(1)}%).`;
    }

    // Card 3: Impact Rating (Score 1-5)
    const scoreVal = risk.score || 1;
    if (this.riskScoreVal) {
      this.riskScoreVal.textContent = `${scoreVal} / 5`;
    }

    this.riskDots.forEach((dot, idx) => {
      const level = idx + 1;
      dot.className = 'risk-dot';
      if (level <= scoreVal) {
        dot.classList.add(`active-${scoreVal}`);
      }
    });

    const impactDescriptions = {
      1: 'Routine Read-Only / Diagnostic — Zero operational risk',
      2: 'Low Impact — Local development state modification',
      3: 'Moderate Impact — Deployment, container or branch modification',
      4: 'High Impact — Financial, access permission, or production change',
      5: 'Critical / Irreversible Impact — Permanent data drop or force overwrite'
    };

    if (this.riskLabelText) {
      this.riskLabelText.textContent = impactDescriptions[scoreVal] || 'Operational Impact Level';
    }

    // Card 4: Top Candidate Alternatives Softmax Distribution
    if (this.rankingBarsList && decision.ranking) {
      const topCandidates = decision.ranking.slice(0, 5);
      this.rankingBarsList.innerHTML = topCandidates.map((c, idx) => {
        const isWinner = (idx === 0);
        const pct = (c.probability * 100).toFixed(1);
        return `
          <div class="ranking-item ${isWinner ? 'winner' : ''}">
            <div class="ranking-item-meta">
              <span class="ranking-item-id">${idx + 1}. ${c.id}</span>
              <span class="ranking-item-pct">${pct}%</span>
            </div>
            <div class="ranking-track">
              <div class="ranking-fill" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  async handleMetaCommand(cmd) {
    const term = this.terminalStream;
    const addLine = (txt, cls = 'sys') => this.appendTerminalLog(txt, cls);

    if (cmd === ':help') {
      addLine('=== S1stem REPL Interactive Help ===', 'sys');
      addLine(':tools       - List all currently registered action candidates', 'info');
      addLine(':add-tool    - Open the action registration modal dialog', 'info');
      addLine(':benchmark   - Execute 10 consecutive real neural decisions to measure p50/p95 latency', 'info');
      addLine(':clear       - Clear the terminal history stream', 'info');
      addLine('', 'info');
      addLine('Real-World Natural Language Examples to Try:', 'sys');
      addLine('  - "Check repository status and inspect modified files"', 'info');
      addLine('  - "Drop all tables in production database and wipe records"', 'info');
      addLine('  - "Search in codebase where the choice function is defined"', 'info');
      addLine('  - "Deploy microservice update to AWS Lambda"', 'info');
      addLine('  - "Issue a 50 dollar refund to customer for disputed order"', 'info');
      addLine('  - "Run security dependency vulnerability scan for CVEs"', 'info');
      addLine('  - "Kill zombie background process with PID 8421"', 'info');
    } else if (cmd === ':tools') {
      addLine(`=== Registered Actions (${this.tools.length}) ===`, 'sys');
      this.tools.forEach((t, i) => {
        addLine(`  [${String(i + 1).padStart(2, '0')}] ${t.id.padEnd(28)} - ${t.label}`, 'info');
      });
    } else if (cmd === ':add-tool') {
      this.openModal();
      this.switchTab('add');
      addLine('[gui] Opened Action Registration Modal.', 'sys');
    } else if (cmd === ':clear') {
      this.clearTerminal();
    } else if (cmd === ':benchmark') {
      await this.runLiveBenchmark();
    } else {
      addLine(`Unknown command '${cmd}'. Type :help for command list.`, 'error');
    }
  }

  async runLiveBenchmark() {
    this.appendTerminalLog('--- Starting Live S1stem Neural Latency Benchmark (10 Real Iterations) ---', 'sys');
    this.setStatus('BENCHMARKING...', 'warn');
    if (this.btnExecute) this.btnExecute.disabled = true;

    const testPrompts = [
      'Inspect modified files in the git repository',
      'Execute a SELECT query to count active users in database',
      'Run SAST security vulnerability scan against known CVEs',
      'Recursively delete all temporary files with rm -rf',
      'Deploy new Docker container version onto Kubernetes cluster',
      'Search with ripgrep all references to the API key',
      'Send notification message to engineering Slack channel',
      'Terminate background daemon process to free up system memory',
      'Create snapshot database backup dump to S3 bucket before release',
      'Invalidate and rotate cryptographic authentication tokens in vault'
    ];

    const latencies = [];

    for (let i = 0; i < testPrompts.length; i++) {
      const prompt = testPrompts[i];
      const t0 = performance.now();
      const res = await this.engine.choice(prompt);
      const dt = performance.now() - t0;
      latencies.push(dt);
      this.appendTerminalLog(`[iter ${i + 1}/10] "${prompt.slice(0, 35)}..." -> ${res.winner} in ${dt.toFixed(2)} ms`, 'info');
    }

    latencies.sort((a, b) => a - b);
    const min = latencies[0].toFixed(2);
    const max = latencies[latencies.length - 1].toFixed(2);
    const p50 = latencies[Math.floor(latencies.length * 0.5)].toFixed(2);
    const p95 = latencies[Math.floor(latencies.length * 0.95)].toFixed(2);
    const avg = (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(2);
    const throughput = (1000 / avg).toFixed(1);

    this.appendTerminalLog('--- Benchmark Results ---', 'success');
    this.appendTerminalLog(`Min: ${min} ms | Avg: ${avg} ms | p50: ${p50} ms | p95: ${p95} ms | Max: ${max} ms`, 'success');
    this.appendTerminalLog(`Throughput: ~${throughput} decisions/second locally. Cost: $0.000. Zero server calls.`, 'success');

    if (this.hdrLatency) this.hdrLatency.textContent = `${avg} ms`;
    this.setStatus('READY', 'live');
    if (this.btnExecute) this.btnExecute.disabled = false;
  }

  escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  const app = new DispatcherApp();
  app.init();
});
