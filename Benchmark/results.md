# Empirical Decision Benchmark: S1stem vs Microsoft-Decision-1 vs GPT-6 Luna (Decisions) vs GPT-6.1 Sol

**Execution Date**: 2026-10-14T19:50:53.423Z  
**Hardware & Platform**: Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz (win32, Node v24.13.0)  
**Evaluation Protocol**: Stratified Sampling (Seed: 42, 100 samples/source)  
**Evaluated Held-Out Cases**: 612 items  

## 1. Headline Comparative Matrix (Held-Out Test Set)

| Model / Architecture | Runtime & Deployment | Top-1 Accuracy (95% Wilson CI) | Median Latency (p50) | Tail Latency (p95) | Calibration (ECE / Brier) | Invariant Flip Rate | Cost / 1k Ops |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1stem v1.2.0 (CLM)** | Local WebGPU / ONNX (In-Process CPU/GPU) | **65.5%** [61.7% - 69.2%] | **10.4 ms** | **27.3 ms** | 0.0692 / 0.1583 | **0%** | **$0.000** |
| **Microsoft-Decision-1** | Azure Foundry SystemOne Decision API | 76.6% [73.1% - 79.8%] | 152.25 ms | 419.97 ms | 0.1608 / 0.1861 | N/A% | $0.042 |
| **gpt-6-luna (Decisions)** | Azure Foundry OpenAI Responses API | 75.8% [72.3% - 79%] | 1364.91 ms | 2442.22 ms | N/A | 5.7% | ~$0.0395 |
| **gpt-6.1-sol (Frontier)** | Azure Foundry OpenAI Chat Completions API | 70.4% [66.7% - 73.9%] | 1703.68 ms | 7049.41 ms | N/A *(Generative)* | 2.9% | ~$0.701 |

## 2. Statistical Significance (Paired McNemar Test vs S1stem)

* **S1stem vs Microsoft-Decision-1**:
  * Contingency: both correct: 386, S1stem only: 15, MD1 only: 83, both wrong: 128
  * McNemar $\chi^2$: 45.806, $p$-value: **0** (Statistically Significant (p < 0.05))

* **S1stem vs gpt-6-luna (Decisions API)**:
  * Contingency: both correct: 380, S1stem only: 21, Luna only: 84, both wrong: 127
  * McNemar $\chi^2$: 36.61, $p$-value: **0** (Statistically Significant (p < 0.05))

* **S1stem vs gpt-6.1-sol (Chat Completions)**:
  * Contingency: both correct: 350, S1stem only: 51, GPT-6 only: 81, both wrong: 130
  * McNemar $\chi^2$: 6.371, $p$-value: **0.0116** (Statistically Significant (p < 0.05))

## 3. Action Space Scalability Limit ($K = 32$ to $K = 3,000$ Candidates)

| Candidate Count ($K$) | S1stem Caching | S1stem Latency | Microsoft-Decision-1 Status | gpt-6-luna Status | gpt-6.1-sol Status |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **K = 32** | 172 ms | **7.04 ms** | Supported | Supported | Supported |
| **K = 64** | 297.1 ms | **7.27 ms** | Supported | Supported | Supported |
| **K = 128** | 534.7 ms | **7.41 ms** | Supported | Supported | Skipped |
| **K = 255** | 875.8 ms | **10.45 ms** | Supported | Skipped | Skipped |
| **K = 256** | 951.9 ms | **9.93 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |
| **K = 1024** | 4137.7 ms | **37.09 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |
| **K = 3000** | 12937.4 ms | **35.06 ms** | BLOCKED **(BLOCKED HTTP 400)** | Skipped | Skipped |

## 4. Contaminated Dev Set Disclosure


> [!WARNING]
> The dev set contains historical operational testbed items where environmental tokens partially overlap with contextual polarity rules.
> S1stem Dev Accuracy: **100%** vs MD1: **100%** vs Luna: **100%** vs GPT-6: **100%** (4 items).

