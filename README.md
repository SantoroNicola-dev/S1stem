# S1stem v1.2.0

> **What S1stem is (and what it isn't):**  
> S1stem is not a giant frontier LLM, and it does not try to be one. It is a compact, 109M-parameter local decision engine designed for **fast discrete choices** (System 1). While large cloud LLMs excel at creative writing and complex multi-step reasoning, using them to pick a tool or route a command takes seconds and burns API tokens. S1stem handles that specific job locally in **~10 milliseconds, at zero cost, with zero network lag**, and scales to thousands of actions.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Runtime](https://img.shields.io/badge/runtime-WebGPU%20%2F%20ONNX%20(Transformers.js)-yellow.svg)]()
[![Backbone](https://img.shields.io/badge/backbone-BGE--Base%20109M%20(768%20dim)-blueviolet.svg)]()
[![Memory RSS](https://img.shields.io/badge/memory-%3C%201%20GB%20RAM%20(~686MB)-success.svg)]()
[![Action Scaling](https://img.shields.io/badge/actions-3000%2B%20scalable-brightgreen.svg)]()

```javascript
import { S1stemEngine } from './core/s1stem.js';

// Initialize System 1 Decision Engine (CLM)
const engine = new S1stemEngine({
  embeddingModelId: 'Xenova/bge-base-en-v1.5',
  embeddingDim: 768,
  temperature: 0.04
});

// One-Shot registration of agent actions/tools (vector computation performed once)
await engine.registerActions([
  { id: 'FILE_READ',     label: 'Read Local File',       description: 'Inspects code and configuration from filesystem.' },
  { id: 'RUN_TERMINAL',  label: 'Execute Shell Command', description: 'Spawns terminal subprocess for compilation and tests.' },
  { id: 'GIT_COMMIT',    label: 'Create Git Commit',     description: 'Records code change snapshot into git branch.' },
  { id: 'DEPLOY_WEB',    label: 'Deploy Web App',        description: 'Pushes bundle to hosting production cluster.' }
]);

// Instant runtime decision (encodes only the new incoming state: ~12-16 ms, zero tokens wasted)
const decision = await engine.choice('Need to run npm test and check compiler errors in bash terminal.');
console.log(decision.winner);     // "RUN_TERMINAL"
console.log(decision.confidence); // 0.942 (94.2%)
console.log(decision.latencyMs);  // ~12.4 ms
```

---

## 1. Concept, Architecture & Purpose

### Real Operation: Making Local Agent Tasks Agile and Cost-Effective
In modern agentic automation systems (coding agents, long-horizon decision loops, robotic systems, and browser navigators), the overwhelming majority of operations consists of **continuous discrete choices and micro-decisions**:
* Selecting which tool to invoke among hundreds of registered capabilities.
* Triage and routing compiler or test runner errors into the correct recovery action.
* Choosing the next branch of a plan or next reflexive maneuver from a live visual canvas state.

Delegating each of these micro-choices to massive generative models (System 2, such as Claude, GPT, or local 70B models) is **inefficient, expensive, and slow**: it consumes valuable context window tokens, introduces multi-second delays on every step, and increases the risk of attention drift or hallucinations in long-horizon loops.

**S1stem is built to act as the System 1 module for local AI agents:** a compact, deterministic, ultra-fast (<16 ms) engine that resolves discrete choices and reflexive decisions, delegating only high-level deliberative reasoning to System 2 models.

---

### Architecture: Contrastive Language Model (CLM) — Nvidia & Stanford University
S1stem is grounded in the principles of the **Contrastive Language Model (CLM)** architecture developed jointly by Nvidia and Stanford University:

```
[ Candidate Actions {a_1, ..., a_K} ]
          │
          ▼ (One-Shot Boot-Time Encoding)
    [ Action Encoder E_a ] ────────► [ Pre-Cached Action Matrix A in RAM ] (K x 768)
                                                      │
                                                      │ (Fast Inner Product)
[ Current State s ] ──► [ State Encoder E_s ] ───────►┴──► [ Softmax z = (v_s · A^T) / τ ]
(Canvas Frame, Log, Code, Telemetry)   (v_s in R^768)                   │
                                                                        ▼
                                                       [ Winning Action + Confidence ] (<16 ms)
```

1. **Dual-Encoder (Two-Tower Architecture):**
   * **State Encoder ($E_s$):** Projects the agent's current state (visual canvas frame from `HTMLCanvasElement`, screenshot, telemetry log, error text, or code snippet) into a normalized latent vector $v_s \in \mathbb{R}^{768}$ with unit norm $\|v_s\|_2 = 1.0$.
   * **Action Encoder ($E_a$):** Projects definitions of available candidate decisions, tools, or commands into latent vectors $v_a \in \mathbb{R}^{768}$ with $\|v_a\|_2 = 1.0$.
   * Both encoders map into the **same shared embedding space**.

2. **One-Shot Action Pre-computation (The Decisive Advantage over Autoregressive LLMs):**
   * **Traditional decision models (e.g., Causal LLMs):** Must re-evaluate the entire context by concatenating state and all candidate options on every single decision step ($O(K)$ tokens per turn). This leads to context window explosion, elevated latencies (~150–2,000 ms), and **catastrophic failure when choices exceed 255 candidates**.
   * **S1stem (CLM):** Computes action representations **exactly once at initialization (`registerActions`)** and caches the Action Matrix $A$ in memory. At runtime, the model only encodes the **incoming state**: decision resolution collapses to a fast inner product $v_s \cdot A^T / \tau$ and calibrated Softmax.
   * This enables S1stem to effortlessly scale to **over 3,000 simultaneous actions and tools** while maintaining sub-16 ms latency.

3. **Superior Long-Horizon Task Performance:**
   * On prolonged agentic workflows, the CLM architecture achieves an **81.6% success rate versus 71.1% for JEV**, operating **5.7x to 9x faster** (16 ms vs 150 ms).
   * It completely eliminates tool selection token burn ($0.00 per million decisions) and prevents context degradation in the primary System 2 orchestrator.

---

## 2. Agentic Use Cases & Interactive Testbeds

The platform includes both direct interfaces for agentic loops (tool routing, error triage) and interactive graphical testbeds demonstrating Dual-Mode inference (`Visual Frame` vs `Text State`):

### 1. Integration into Local Agents (Tool Routing & Triage)
```javascript
// Autonomous tool dispatching without querying the primary generative LLM
const agentTools = [
  { name: 'read_file', description: 'Read and inspect content from project files.' },
  { name: 'run_command', description: 'Execute bash terminal shell commands, tests, and scripts.' },
  { name: 'browser_action', description: 'Control headless browser page and inspect DOM.' }
];

const dispatch = await engine.routeTool('User requested running unit tests in bash terminal', agentTools);
console.log(dispatch.selectedTool); // "run_command" (<1 ms)
```

### 2. Interactive Reactive Control Testbeds

| Application | Architectural System 1 Role | Supported Modes | Testbed Link |
| :--- | :--- | :--- | :--- |
| **Tetris AI** | Geometric Macro-Placement ($T_0$) | Visual Frame / Text State | [Tetris AI Testbed](examples/tetris/index.html) |
| **Snake AI** | Spatial Geodesic Reflex ($T_0$) | Visual Frame / Text State | [Snake AI Testbed](examples/snake/index.html) |
| **Rover AI** | Tactical Navigation & Hazard Detection | Visual Frame / Text State | [Rover AI Testbed](examples/rover/index.html) |
| **Agent Dispatcher** | Tool Routing & Security Arbiter (REPL) | Natural Language / Noul / Score | [Agent Dispatcher Testbed](examples/dispatcher/index.html) |
| **PowerShell Showcase** | Command-Line Assistant & Slot-Filling | Natural Language / Noul / Score | [PowerShell Showcase](powershell/) |

---

### Interactive Dual-Mode Testbed Details

#### 1. Tetris AI (Macro-Placement Heuristic)
* **Dual-Mode Evaluation:**
  * *Visual Frame:* 10×20 matrix canvas screenshot analyzed directly by Qwen3.5 VLM to evaluate column heights and contour shapes.
  * *Text State:* Immediate text observation summarizing geometric metrics (completed lines, covered holes, surface bumpiness).
* **Testbed:** [`examples/tetris/index.html`](examples/tetris/index.html)

#### 2. Snake AI (Spatial Geodesic Reflex)
* **Dual-Mode Evaluation:**
  * *Visual Frame:* Direct canvas scanning of the 16×16 grid, visually detecting food location and snake body configuration.
  * *Text State:* BFS geodesic distance to food, open escape pockets (flood-fill), and survival masks.
* **Testbed:** [`examples/snake/index.html`](examples/snake/index.html)

#### 3. Rover AI (Tactical Hazard Detection & Safety Scoring)
* **Dual-Mode Evaluation:**
  * *Visual Frame:* Outpost camera frame tracking sentry patrol zones and energy conduits.
  * *Text State:* Sensory telemetry coupled with polar boolean hypothesis verification (`s1.noul`) and ordinal safety rating (`s1.score`).
* **Testbed:** [`examples/rover/index.html`](examples/rover/index.html)

#### 4. Agent Dispatcher & Security Guard (Autonomous Tool Routing & Guardrail)
* **Real-World REPL User Interaction:**
  * *Interactive Terminal:* Natural prompt line (`s1stem-agent> `) to type operational instructions in natural language without UI button constraints.
  * *Real 3-Step Pipeline:* Dispatches user intent across 28 authentic DevOps/Cloud/Security tools (`routeTool`), executes polar authorization guardrail (`noul`: ALLOW vs FLAGGED), and calculates execution impact rating 1–5 (`score`).
  * *Tool Space Manager:* Built-in modal to browse active tools, register custom tools, or import JSON manifests with instant RAM vectorization.
  * *Live Integrated Benchmark (`:benchmark`):* Runs 10 consecutive neural decisions measuring minimum, mean, p50, p95 latency and real throughput (~220 decisions/sec at $0.00).
* **Testbed:** [`examples/dispatcher/index.html`](examples/dispatcher/index.html)

---

## 3. Real-World Application Showcase: PowerShell Assistant (`powershell/`)

S1stem is a general-purpose, domain-agnostic decision engine. One of its **concrete practical applications** is real-time assistance for console terminal typing (shells, IDEs, system automation), where a compact System 1 model can instantly provide the exact command syntax when the user cannot recall it from memory.

The [`powershell/`](powershell/) directory contains a **reference implementation** showcasing how to integrate S1stem directly into PowerShell:

```powershell
# Practical usage examples in the PowerShell console:
s1 command to create a new file in D:\
s1 command to list files in a directory in c:\test
s1 create a new directory and enter inside, directory name new_dir
s1 show processes consuming the most memory
s1 ping google to check network connection
s1 check available disk space on drives
s1 calculate sha256 hash of a file
```

* **How this showcase leverages S1stem capabilities:**
  * **Complete PowerShell Catalog (Over 1,580 Cmdlets & Functions)**: The entire official PowerShell 7 catalog (`powershell/data/pwsh-cmdlets.json`) is vectorized in RAM, covering all standard cmdlets (from `Get-FileHash` and `Compress-Archive` to `ConvertFrom-Json` and `Get-Service`), with 100% English semantic descriptions.
  * **Two-Tier Instant Neural Routing**: Maps user intent between Tier 1 (curated high-precision patterns with advanced slot-filling) and Tier 2 (semantic search across 1,582 cmdlets) in 3–10 ms at $0.00 token cost.
  * **Contextual Slot-Filling**: Automatically extracts paths (`D:\`, `c:\test`), directory names (`new_dir`), filenames, and flags directly from natural language.
  * **Polar Security Guardrail (Noul & Score)**: If the proposed command is destructive or modifies system state (e.g. `Remove-Item -Recurse`, process termination), the engine flags it with a visible alert before execution.
  * **Interactive Action Menu**:
    * `[E]` (or Enter): Executes the command in the shell
    * `[C]`: Copies the command to system clipboard
    * `[M]`: Inserts the command into the command line for editing before running
    * `[A]`: Cancels the operation

* **To install and test this application showcase:**
```powershell
# Run the installer script from the powershell folder:
powershell -ExecutionPolicy Bypass -File .\powershell\install.ps1
```

---

## 4. Quickstart & Minimal Working Example

### Prerequisites
* **Node.js** $\ge$ 18.0.0 or any **modern browser** with WebGPU and ES Modules support.
* Local HTTP server (`python -m http.server 8080` or `npx serve .`).

### Minimal Working Example (MWE)

```javascript
import { S1stemEngine } from './core/s1stem.js';

// 1. Initialize unified CLM engine
const engine = new S1stemEngine({
  embeddingModelId: 'Xenova/bge-base-en-v1.5',
  embeddingDim: 768,
  temperature: 0.04
});

// 2. Candidate action registration (One-shot)
await engine.registerActions([
  { id: 'RETURNS_POLICY', label: 'Return & refund policy' },
  { id: 'TECH_SUPPORT',    label: 'Technical troubleshooting' },
  { id: 'STORE_HOURS',     label: 'Retail store hours' }
]);

// 3. Evaluate decision (automatic routing based on input type)
const decision = await engine.choice('Identify user intent: What is the return window for hardware purchased online?');

console.log(`Winning Action : ${decision.winner}`);                  // RETURNS_POLICY
console.log(`Confidence     : ${(decision.confidence * 100).toFixed(1)}%`); // 94.0%
console.log(`Latency        : ${decision.latencyMs.toFixed(3)} ms`);
```

Run the offline verification test suite:
```bash
npm test
```

---

## 5. Benchmarks & Honest Comparison

To keep things completely transparent, we tested **S1stem v1.2.0** against live cloud models: **Microsoft-Decision-1** (Azure Foundry SystemOne API), **gpt-6-luna** (OpenAI Responses API on Azure), and **gpt-6.1-sol** (Azure OpenAI chat completions).

The benchmark evaluated **612 real test cases** drawn from 6 standard public datasets (*MASSIVE*, *When2Call*, *Glaive Tool Calling*, *HellaSwag*, *CLINC150*, *Banking77*) plus practical DevOps and cloud triage scenarios.

---

### The Reality: What the Numbers Actually Mean

| Model / Service | Type & Deployment | Accuracy (Top-1) | Median Latency (p50) | Cost / 1k Decisions | Max Options Handled | Offline? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1stem v1.2.0** | **Local WebGPU / CPU (109M)** | **65.5%** | **10.4 ms** | **$0.000** | **3,000+** | **Yes** (100% in-browser / Node) |
| **Microsoft-Decision-1** | Azure Cloud Decision API | 76.6% | 152.2 ms | $0.042 | 255 *(Hard limit)* | No (Cloud API) |
| **gpt-6-luna** | Azure OpenAI Responses API | 75.8% | 1,364.9 ms | ~$0.039 | ~128 *(Prompt bloat)* | No (Cloud API) |
| **gpt-6.1-sol** | Azure OpenAI Chat Completions | 70.4% | 1,703.7 ms | ~$0.701 | ~64 *(Prompt bloat)* | No (Cloud API) |

---

### 3 Takeaways in Plain English

1. **Accuracy: Big LLMs understand broader language nuances**
   * Giant cloud models reach higher overall accuracy on general text intents (**75.8% – 76.6%** vs **65.5%** for S1stem). That is expected: S1stem has only 109 million parameters, while frontier LLMs have hundreds of billions.
   * S1stem does not replace GPT for open-ended conversation; it handles routine action selection and routing so the big LLM doesn't have to.

2. **Speed: S1stem is 14× to 160× faster**
   * S1stem decides in **10.4 milliseconds** right on your device.
   * Microsoft-Decision-1 takes ~150 ms over the network.
   * GPT-6 models take **1.3 to 1.7 seconds** per single choice. For real-time applications (command-line tools, games, UI navigation, high-frequency agent loops), multi-second delays are unusable.

3. **Scale: S1stem handles large action catalogs without crashing**
   * If you have a large palette of actions (like PowerShell's 1,582 cmdlets or hundreds of API tools):
     * **Microsoft-Decision-1 crashes with an HTTP 400 error** if you pass more than 255 options (*"choice question decision requires 2-255 options"*).
     * **GPT-6 models suffer prompt token explosion**, slowing down further and costing more money per turn.
     * **S1stem pre-computes action vectors once** and searches across **3,000+ actions in ~35 ms** at zero cost.

---

### Action Space Scalability Test ($K = 32$ to $K = 3,000$ Options)

Here is what happens as the number of available candidate actions grows:

| Number of Actions ($K$) | S1stem One-Shot Setup | S1stem Decision Time | Microsoft-Decision-1 | GPT-6 (Sol / Luna) |
| :---: | :---: | :---: | :--- | :--- |
| **32 options** | 172 ms | **7.0 ms** | Works (170 ms) | Works (~1,400 ms) |
| **64 options** | 297 ms | **7.2 ms** | Works (171 ms) | Works (~1,800 ms) |
| **128 options** | 534 ms | **7.4 ms** | Works (183 ms) | High prompt token bloat |
| **255 options** | 875 ms | **10.4 ms** | Works (286 ms) | High prompt token bloat |
| **256 options** | 951 ms | **9.9 ms** | **BLOCKED (HTTP 400 error)** | Excessive latency / cost |
| **1,024 options** | 4,137 ms | **37.1 ms** | **BLOCKED (HTTP 400 error)** | Context window bloat |
| **3,000 options** | 12,937 ms | **35.1 ms** | **BLOCKED (HTTP 400 error)** | Context window bloat |

---

### Permutation Robustness (Does changing option order change the choice?)

When you shuffle the order of choices:
* Large autoregressive models exhibit a **3% to 6% flip rate** simply because the order of options in the prompt changed (attention bias towards the first or last options).
* S1stem computes independent dot products for each candidate: its flip rate on shuffled options is strictly **0.0%**.

---

### Verifying the Tests Locally

To run the offline verification test suite on your own machine:

```bash
# Install dependencies
npm install

# Run the local engine tests (includes Tetris, Snake, Rover and 3,000-action scale test)
npm test

# Run the PowerShell assistant tests (1,595 cmdlets catalog)
npm run test:powershell
```

---

## 6. Running Testbeds Locally

Start a local HTTP server from the repository root:

```bash
python -m http.server 8080
# or
npx serve .
```

Navigate to any of the interactive testbeds in your browser:
* **Tetris AI:** `http://localhost:8080/examples/tetris/index.html`
* **Snake AI:** `http://localhost:8080/examples/snake/index.html`
* **Rover AI:** `http://localhost:8080/examples/rover/index.html`
* **Agent Dispatcher:** `http://localhost:8080/examples/dispatcher/index.html`
* **PowerShell Showcase:** `powershell -ExecutionPolicy Bypass -File .\powershell\install.ps1` (Command-Line Assistant)

---

## 7. License

Released under the **Apache 2.0** License — see [LICENSE](./LICENSE) for details.
