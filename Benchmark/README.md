# Benchmark — S1stem vs Competitor Decision Models (Azure Foundry)

Subproject for head-to-head empirical benchmarking between:
1. **S1stem v1.2.0** — Local Contrastive Language Model (CLM) running edge in-process on CPU / WebGPU ($0.00 cost, zero network round-trip).
2. **Microsoft-Decision-1** — Microsoft's dedicated fast decision model hosted on Azure Foundry via SystemOne Decision endpoint.
3. **gpt-6-luna** — OpenAI's dedicated bounded decision model on the Decisions API (`POST /v1/decisions`).
4. **gpt-6.1-sol** — Azure Foundry frontier generative large language model via chat completions.

> **Note**: Private cloud credentials in `Benchmark/config.json` are gitignored to prevent accidental leaks. The benchmark scripts and public dataset loaders can be safely versioned.

---

## 1. Configuration (`config.json`)

1. Open `Benchmark/config.json` (or copy from `Benchmark/config.example.json`):
```json
{
  "azure_foundry": {
    "endpoint": "https://<your-foundry-resource>.services.ai.azure.com",
    "api_key": "YOUR_AZURE_FOUNDRY_API_KEY",
    "gpt_6_1_sol": {
      "deployment_name": "gpt-6.1-sol",
      "path": "/openai/v1/chat/completions"
    },
    "microsoft_decision_1": {
      "deployment_name": "decision-1",
      "path": "/providers/microsoft/v1/systemone"
    }
  },
  "openai": {
    "endpoint": "https://api.openai.com",
    "api_key": "YOUR_OPENAI_API_KEY",
    "gpt_6_luna": {
      "model": "gpt-6-luna",
      "path": "/v1/decisions"
    }
  },
  "s1stem": {
    "temperature": 0.04
  },
  "benchmark_settings": {
    "default_samples": 50,
    "timeout_ms": 30000
  }
}
```
2. Replace `"YOUR_AZURE_FOUNDRY_API_KEY"` and/or `"YOUR_OPENAI_API_KEY"` with your active keys (or pass `AZURE_FOUNDRY_API_KEY` / `OPENAI_API_KEY` as environment variables).

---

## 2. Pre-Flight Connectivity Check

Run a single diagnostic call to verify credentials and connectivity before executing the full benchmark:

```bash
node Benchmark/check_connection.js
```

This tests:
- S1stem local engine initialization and latency.
- Microsoft-Decision-1 endpoint reachability on `/providers/microsoft/v1/systemone`.
- gpt-6-luna Decisions API reachability on `POST /v1/decisions`.
- gpt-6.1-sol endpoint reachability on `/openai/v1/chat/completions`.


---

## 3. Running the Head-to-Head Benchmark

To run the head-to-head benchmark with the default 50 samples:

```bash
node Benchmark/run_benchmark.js
```

### Custom Sample Count

To customize the number of test decisions (e.g. 20 for a quick smoke test or 100 for an extended evaluation):

```bash
node Benchmark/run_benchmark.js --samples 20
node Benchmark/run_benchmark.js --samples 100
```

---

## 4. Benchmark Outputs

- **Console Matrix**: Real-time table displaying Accuracy, p50 Latency, p95 Latency, Throughput, and Cost per 1k operations.
- **`Benchmark/results.json`**: Structured JSON output formatted for automated analysis and CI/CD pipelines.
