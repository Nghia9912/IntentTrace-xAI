# IntentTrace-xAI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IntentTrace-xAI** is an open-source evaluation framework designed to mathematically verify whether LLM-generated code aligns with the user's initial natural language intent. It mitigates hallucination risks by combining deterministic static analysis with deep neural attribution.

## System Architecture

The framework operates on a multi-pipeline architecture, transitioning from low-cost, high-reliability deterministic checks to deep neural analysis:

1. **Static Analysis Pipeline (Base Layer):** Utilizes Python's native ast module to construct a Control Flow Graph (CFG). It runs a Breadth-First Search (BFS) to detect unreachable/dead code, a primary indicator of LLM hallucinations. Operates with O(V+E) complexity.
2. **Semantic Similarity Pipeline:** Maps both the natural language intent and the generated code's documentation into a shared high-dimensional vector space using sentence-transformers (all-MiniLM-L6-v2) to compute Cosine Similarity.
3. **Zero-Shot xAI Attribution Pipeline:** The core engine. It utilizes microsoft/codebert-base and PyTorch's Captum library. By applying **Integrated Gradients (IG)** on the Cosine Similarity objective function, it computes an attribution matrix, mapping control flow execution branches back to specific tokens in the original prompt.
4. **Aggregation Hub:** Normalizes the tensor outputs and outputs a final confidence metric.

## Installation

Ensure you have a modern Python environment (>=3.9). Clone the repository and install the pinned dependencies:

    git clone https://github.com/Nghia9912/IntentTrace-xAI.git
    cd IntentTrace-xAI
    python -m venv venv

    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate

    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

## Quick Start (Static Analysis)

The current release (v0.1.0) includes the fully functional Static Analysis module.

    from core.static_analyzer import DeadCodeDetector

    # LLM generated code with hallucinated dead code
    code = """
    def process(x):
        return x * 2
        print("LLM Hallucination") # Unreachable
    """

    detector = DeadCodeDetector(code)
    dead_nodes = detector.analyze()

    for node in dead_nodes:
        print(f"Dead code detected at line: {node['line']}, type: {node['type']}")

## Roadmap

- [x] **Sprint 1:** Deterministic Static Analysis & Triplet Ground Truth Dataset.
- [ ] **Sprint 2:** Semantic Space Integration (sentence-transformers).
- [ ] **Sprint 3:** xAI Engine & Gradient Mathematics (Captum & CodeBERT).
- [ ] **Sprint 4:** Metric Aggregation, CI/CD Integration, and CLI Deployment.

## References

1. Feng et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages.
2. Kokhlikyan et al. (2020). Captum: A unified and generic model interpretability library for PyTorch.
3. Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks.