# IntentTrace-xAI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IntentTrace-xAI** is a neuro-symbolic evaluation framework designed to mathematically verify whether LLM-generated code aligns with the user's natural language intent. By integrating deterministic static analysis with deep neural attribution, it provides a transparent and explainable metric for code integrity.

## Motivation

As LLMs become central to software development, the risk of "Code Hallucinations"—where generated code appears syntactically correct but diverges from the intended logic—grows significantly. IntentTrace-xAI addresses this by offering an automated, explainable auditing layer that bridges the gap between natural language requirements and source code execution.

## System Architecture

The framework operates on a multi-pipeline architecture, transitioning from low-cost deterministic checks to deep neural analysis:

1.  **Static Analysis Pipeline:** Uses Python's `ast` module to construct a Control Flow Graph (CFG) and detect unreachable code with $\mathcal{O}(V+E)$ complexity.
2.  **Semantic Similarity Pipeline:** Maps intent and code docstrings into a high-dimensional vector space using `all-MiniLM-L6-v2` for Cosine Similarity ($S_{cos}$) calculation.
3.  **Zero-Shot xAI Attribution Pipeline:** The core engine utilizing `microsoft/codebert-base` and PyTorch `Captum`. It applies **Integrated Gradients (IG)** to compute an attribution matrix $\Phi_{IG}$, mapping logic branches back to specific tokens in the original prompt.
4.  **Aggregation Hub:** Normalizes tensor outputs and produces the final confidence metric $D_{intent\_final}$.

## Detailed Multi-pipeline Architecture

IntentTrace-xAI employs a **Neuro-Symbolic** approach, stacking layers for maximum reliability:

* **Layer 1: Static Analysis (Symbolic Filter):** Builds a CFG using the `ast` module to detect dead code. Hallucinations trigger a severe 0.1x penalty.
* **Layer 2: Semantic Similarity (Global Context):** Uses `all-MiniLM-L6-v2` to compute $S_{cos}$ between intent and code structure.
* **Layer 3: xAI Engine (Neural Attribution):** Leverages `CodeBERT` and `Integrated Gradients` to map execution branches back to specific prompt tokens.
* **Layer 4: Aggregation Hub:** Fuses all signals into $D_{intent\_final} = \alpha \cdot \hat{\Phi}_{IG} + (1-\alpha) \cdot S_{cos}$.

## Experimental Results (v0.1.0)

The system was evaluated against a Ground Truth dataset of 20 triplets (Prompt, $C_{true}$, $C_{false}$). The following AUC-ROC scores demonstrate the framework's discriminative power at various $\alpha$ (aggregation weights) levels:

| Alpha ($\alpha$) | AUC-ROC Score | Status |
| :--- | :--- | :--- |
| 0.3 | **0.9550** | PASSED |
| 0.5 | **0.8825** | PASSED |
| 0.7 | **0.7875** | PASSED |

## Installation

    git clone https://github.com/Nghia9912/IntentTrace-xAI.git
    cd IntentTrace-xAI
    python -m venv venv

    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate

    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

## Quick Start

    from core.evaluator import IntentEvaluator

    evaluator = IntentEvaluator()
    prompt = "Write a function to calculate the factorial of a positive integer n."
    code = """
    def factorial(n):
        if n <= 1: return 1
        return n * factorial(n - 1)
    """

    result = evaluator.evaluate(prompt, code, alpha=0.3)
    print(f"Confidence Score: {result['d_intent_final']:.4f}")

## Roadmap

- [x] **Sprint 1:** Deterministic Static Analysis & Triplet Ground Truth Dataset.
- [x] **Sprint 2:** Semantic Space Integration (sentence-transformers).
- [x] **Sprint 3:** xAI Engine & Gradient Mathematics (Captum & CodeBERT).
- [x] **Sprint 4:** Metric Aggregation, AUC-ROC Benchmark, and CLI Deployment.
- [x] **Sprint 5:** CI/CD Integration and Automated Code Coverage Reports.

## References

1. Feng et al. (2020). *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*.
2. Kokhlikyan et al. (2020). *Captum: A unified and generic model interpretability library for PyTorch*.
3. Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks*.