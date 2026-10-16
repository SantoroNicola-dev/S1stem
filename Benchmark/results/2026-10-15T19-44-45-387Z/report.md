# Empirical Decision Benchmark: S1stem vs Microsoft-Decision-1 vs GPT-6 Luna (Decisions) vs GPT-6.1 Sol

**Execution Date**: 2026-10-15T20:42:44.585Z  
**Hardware & Platform**: Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz (win32, Node v24.13.0)  
**Evaluation Protocol**: Stratified Sampling (Seed: 42, 50 samples/source)  
**Evaluated Held-Out Cases**: 312 items  

## 1. Headline Comparative Matrix (Held-Out Test Set)

| Model / Architecture | Runtime & Deployment | Top-1 Accuracy (95% Wilson CI) | Median Latency (p50) | Tail Latency (p95) | Calibration (ECE / Brier) | Invariant Flip Rate | Cost / 1k Ops |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1stem v1.2.0 (CLM)** | Local WebGPU / ONNX (In-Process CPU/GPU) | **68.6%** [63.2% - 73.5%] | **525.18 ms** | **1247.52 ms** | 0.0548 / 0.1684 | **0%** | **$0.000** |
| **Microsoft-Decision-1** | Azure Foundry SystemOne Decision API | 75.6% [70.6% - 80.1%] | 233.69 ms | 532.35 ms | 0.1617 / 0.1818 | 2.9% | $0.042 |
| **gpt-6-luna (Decisions)** | Azure Foundry OpenAI Responses API | 74.7% [69.6% - 79.2%] | 1535.3 ms | 2589.86 ms | N/A | 11.4% | ~$0.0388 |
| **gpt-6.1-sol (Frontier)** | Azure Foundry OpenAI Chat Completions API | 73.4% [68.2% - 78%] | 2042.35 ms | 6043.09 ms | N/A *(Generative)* | 5.7% | ~$0.745 |

## 2. Statistical Significance (Paired McNemar Test vs S1stem)

* **S1stem vs Microsoft-Decision-1**:
  * Contingency: both correct: 194, S1stem only: 20, MD1 only: 42, both wrong: 56
  * McNemar $\chi^2$: 7.113, $p$-value: **0.0077** (Statistically Significant (p < 0.05))

* **S1stem vs gpt-6-luna (Decisions API)**:
  * Contingency: both correct: 185, S1stem only: 29, Luna only: 48, both wrong: 50
  * McNemar $\chi^2$: 4.208, $p$-value: **0.0402** (Statistically Significant (p < 0.05))

* **S1stem vs gpt-6.1-sol (Chat Completions)**:
  * Contingency: both correct: 186, S1stem only: 28, GPT-6 only: 43, both wrong: 55
  * McNemar $\chi^2$: 2.761, $p$-value: **0.0966** (Not Statistically Significant)

## 3. Action Space Scalability Limit ($K = 32$ to $K = 3,000$ Candidates)

| Candidate Count ($K$) | S1stem Caching | S1stem Latency | Microsoft-Decision-1 Status | gpt-6-luna Status | gpt-6.1-sol Status |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **K = 32** | 1080.9 ms | **115.5 ms** | Supported | Supported | Supported |
| **K = 64** | 1234.6 ms | **22.99 ms** | Supported | Supported | Supported |
| **K = 128** | 1468.1 ms | **101.43 ms** | Supported | Supported | Skipped |
| **K = 255** | 3072 ms | **319.47 ms** | Supported | Skipped | Skipped |
| **K = 256** | 7457.1 ms | **58.35 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |
| **K = 1024** | 24525.9 ms | **92.47 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |
| **K = 3000** | 77828.9 ms | **34.66 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |

## 4. Contaminated Dev Set Disclosure


> [!WARNING]
> The dev set contains historical operational testbed items where environmental tokens partially overlap with contextual polarity rules.
> S1stem Dev Accuracy: **100%** vs MD1: **100%** vs Luna: **100%** vs GPT-6: **100%** (4 items).

