/**
 * S1stem Deep Nerd Telemetry & Real-Time Decision Stream
 * Computes information theory metrics (Shannon entropy, top-2 margins, cosine similarity),
 * latency jitter (min/avg/max), inference throughput, and manages real-time stream logs with % readouts.
 */

export class NerdTelemetry {
  /**
   * @param {Object} options
   * @param {string} [options.domain] - 'snake' | 'tetris' | 'doom'
   * @param {number} [options.temperature] - Softmax temperature tau
   * @param {number} [options.stateDim] - State embedding dimension (384)
   * @param {number} [options.projDim] - Latent projection dimension (128)
   * @param {string} [options.streamId] - ID of the terminal log container
   * @param {string} [options.statsContainerId] - ID of the nerd stats grid container
   * @param {number} [options.maxLogs] - Maximum entries in rolling log
   */
  constructor({
    domain = 'edge',
    temperature = 0.07,
    stateDim = 384,
    projDim = 128,
    streamId = 'log-stream',
    statsContainerId = 'nerd-stats-grid',
    maxLogs = 60,
  } = {}) {
    this.domain = domain;
    this.temperature = temperature;
    this.stateDim = stateDim;
    this.projDim = projDim;
    this.paramCount = stateDim * projDim * 2; // Two-tower (Ws + Wa)
    this.paramBytes = this.paramCount * 4; // Float32 IEEE-754

    this.streamEl = document.getElementById(streamId);
    this.statsContainer = document.getElementById(statsContainerId);
    this.maxLogs = maxLogs;

    // Rolling latency statistics (window of last 30 decisions)
    this.latencyHistory = [];
    this.decisionCount = 0;
    this.minLatency = Infinity;
    this.maxLatency = 0;
    this.avgLatency = 0;

    this.initStatsGrid();
  }

  /**
   * Renders the initial nerd telemetry DOM cards into the container
   */
  initStatsGrid() {
    if (!this.statsContainer) return;
    this.statsContainer.innerHTML = `
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">LATENCY JITTER</span>
        <span id="nerd-latency" class="nerd-stat-val cyan">0.00 / 0.00 ms</span>
        <span class="nerd-stat-sub">Min / Avg (rolling 30)</span>
      </div>
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">THROUGHPUT</span>
        <span id="nerd-throughput" class="nerd-stat-val emerald">&gt; 5,000 /s</span>
        <span class="nerd-stat-sub">Client Float32 JIT</span>
      </div>
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">SHANNON ENTROPY</span>
        <span id="nerd-entropy" class="nerd-stat-val amber">0.00 nats</span>
        <span class="nerd-stat-sub">H(P) = -&sum; p ln(p)</span>
      </div>
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">TOP-2 MARGIN</span>
        <span id="nerd-margin" class="nerd-stat-val purple">+0.0%</span>
        <span class="nerd-stat-sub">&Delta;P = p&sup1; - p&sup2;</span>
      </div>
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">COSINE SIMILARITY</span>
        <span id="nerd-cosine" class="nerd-stat-val">+0.000</span>
        <span class="nerd-stat-sub">s&top;a / &tau; (&tau;=${this.temperature.toFixed(2)})</span>
      </div>
      <div class="nerd-stat-card">
        <span class="nerd-stat-label">VLM BACKBONE</span>
        <span class="nerd-stat-val cyan">Qwen3.5-0.8B</span>
        <span class="nerd-stat-sub">WebGPU INT4 (Edge Inference)</span>
      </div>
    `;
  }

  /**
   * Records a model decision, computes information theory metrics, and appends to the log
   * @param {Object} data
   * @param {string} data.action - Action ID or label
   * @param {number} data.confidence - Top-1 probability (0..1)
   * @param {number} data.latencyMs - Measured execution latency
   * @param {Array<{action: any, probability: number, dotProduct?: number}>} [data.ranking]
   * @param {string} [data.details] - Short contextual description
   */
  record({ action, confidence, latencyMs, ranking = [], details = '' }) {
    this.decisionCount++;

    // 1. Rolling Latency tracking
    this.latencyHistory.push(latencyMs);
    if (this.latencyHistory.length > 30) this.latencyHistory.shift();

    if (latencyMs < this.minLatency) this.minLatency = latencyMs;
    if (latencyMs > this.maxLatency) this.maxLatency = latencyMs;

    const sumLatency = this.latencyHistory.reduce((a, b) => a + b, 0);
    this.avgLatency = sumLatency / this.latencyHistory.length;
    const throughput = this.avgLatency > 0 ? Math.round(1000 / this.avgLatency) : 5000;

    // 2. Information Theory: Shannon Entropy H(P) = -sum(p * ln(p))
    let entropy = 0;
    let top2Margin = 0;
    let topDot = 0;

    if (ranking && ranking.length > 0) {
      for (const r of ranking) {
        const p = Math.max(1e-12, r.probability);
        entropy -= p * Math.log(p);
      }
      if (ranking.length >= 2) {
        top2Margin = (ranking[0].probability - ranking[1].probability) * 100;
      } else {
        top2Margin = ranking[0].probability * 100;
      }
      topDot = ranking[0].dotProduct ?? (Math.log(ranking[0].probability) * this.temperature);
    } else {
      top2Margin = confidence * 100;
    }

    // 3. Update Nerd Stats DOM Elements
    const latencyEl = document.getElementById('nerd-latency');
    const throughputEl = document.getElementById('nerd-throughput');
    const entropyEl = document.getElementById('nerd-entropy');
    const marginEl = document.getElementById('nerd-margin');
    const cosineEl = document.getElementById('nerd-cosine');

    if (latencyEl) {
      latencyEl.textContent = `${this.minLatency.toFixed(2)} / ${this.avgLatency.toFixed(2)} ms`;
    }
    if (throughputEl) {
      throughputEl.textContent = `${throughput.toLocaleString()} /s`;
    }
    if (entropyEl) {
      entropyEl.textContent = `${entropy.toFixed(2)} nats`;
    }
    if (marginEl) {
      marginEl.textContent = `+${top2Margin.toFixed(1)}%`;
    }
    if (cosineEl) {
      cosineEl.textContent = topDot >= 0 ? `+${topDot.toFixed(3)}` : topDot.toFixed(3);
    }

    // 4. Append to Diagnostic Real-Time Stream with prominent %
    this.appendStreamEntry({
      seq: this.decisionCount,
      action: typeof action === 'object' ? (action.label || action.id) : String(action),
      confidencePct: (confidence * 100).toFixed(1),
      latencyMs: latencyMs.toFixed(2),
      top2Margin: top2Margin.toFixed(1),
      details,
    });
  }

  /**
   * Appends an entry into the diagnostic stream
   */
  appendStreamEntry({ seq, action, confidencePct, latencyMs, top2Margin, details }) {
    if (!this.streamEl) return;

    const row = document.createElement('div');
    row.className = 'log-entry';

    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');

    row.innerHTML = `
      <span class="log-time font-mono">${timeStr}</span>
      <span class="log-seq font-mono">#${String(seq).padStart(3, '0')}</span>
      <span class="log-action-badge winner font-mono">${action}</span>
      <span class="log-pct-badge font-mono">${confidencePct}%</span>
      <span class="log-meta font-mono">${latencyMs}ms | &Delta;+${top2Margin}%${details ? ' | ' + details : ''}</span>
    `;

    this.streamEl.appendChild(row);
    this.streamEl.scrollTop = this.streamEl.scrollHeight;

    while (this.streamEl.children.length > this.maxLogs) {
      this.streamEl.removeChild(this.streamEl.children[0]);
    }
  }

  /**
   * Logs a generic system or engine event
   */
  logSystem(msg, type = 'info') {
    if (!this.streamEl) return;
    const row = document.createElement('div');
    row.className = 'log-entry';
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];

    row.innerHTML = `
      <span class="log-time font-mono">${timeStr}</span>
      <span class="log-seq font-mono">[SYS]</span>
      <span class="log-meta font-mono" style="color: #94a3b8;">${msg}</span>
    `;
    this.streamEl.appendChild(row);
    this.streamEl.scrollTop = this.streamEl.scrollHeight;
  }
}
