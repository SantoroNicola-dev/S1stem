# Contributing to S1stem

Thank you for your interest in contributing to **S1stem**! We welcome community contributions to advance ultra-fast System 1 contrastive AI decision engines.

---

## 🛠️ Development Philosophy

1. **Zero-Compile & Zero-Dependency First:**
   * The core library must remain runnable as pure Vanilla JavaScript (ES Modules) without requiring bundlers, transpilers, or external runtime libraries.
   * Any new feature must preserve the `<0.20 ms` latency envelope on consumer CPUs.
2. **Strict Typeless / Portable Standard:**
   * Keep standard IEEE 754 Float32 compliance for direct interoperability across Web, C#, and Python.
3. **Reproducibility & Safety:**
   * When adding new game or agent testbeds, prioritize deterministic, measurable reward heuristics and clear spatial/state metrics.

---

## 🚀 Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/S1stem.git
   cd S1stem
   ```

2. Run tests to verify the core module:
   ```bash
   npm test
   ```

3. Launch the local showcase server:
   ```bash
   python -m http.server 8080
   # or
   npx serve .
   ```

4. Open `http://localhost:8080` in your web browser.

---

## 📝 Pull Request Guidelines

* Create a dedicated feature branch (`git checkout -b feature/my-cool-feature`).
* Ensure code style is consistent, readable, and well-commented.
* If introducing new heuristics or decision spaces, add documentation explaining state formulations and candidate mappings.
* Submit your PR with a clear description of changes, performance impact, and screenshots/recordings if UI components are modified.

---

## 📜 License Notice

By contributing to S1stem, you agree that your contributions will be licensed under the **Apache License 2.0**.
