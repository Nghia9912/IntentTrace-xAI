import pytest
from core.evaluator import IntentEvaluator

@pytest.fixture(scope="module")
def evaluator():
    return IntentEvaluator()

def test_evaluation_pipeline_integration(evaluator):
    """Verifies the end-to-end execution of all 3 pipelines combined."""
    prompt = "Write a function to add two numbers."
    code = "def add(a, b): return a + b"
    
    result = evaluator.evaluate(prompt, code, alpha=0.5)
    
    assert "d_intent_final" in result
    assert 0.0 <= result["d_intent_final"] <= 1.0
    assert result["has_dead_code"] is False

def test_dead_code_penalty(evaluator):
    """Ensures the presence of dead code properly penalizes the final metric."""
    prompt = "Write a function to add two numbers."
    code = "def add(a, b):\n    return a + b\n    print('dead')"
    
    result = evaluator.evaluate(prompt, code, alpha=0.5)
    
    assert result["has_dead_code"] is True
    assert result["d_intent_final"] < 0.2 # The severe penalty must be applied