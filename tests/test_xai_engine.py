import pytest
from models.hf_wrapper import CodeBERTWrapper
from core.xai_engine import XAIEngine

@pytest.fixture(scope="module")
def xai_setup():
    """Initializes the CodeBERT model and xAI engine for the test suite."""
    wrapper = CodeBERTWrapper()
    engine = XAIEngine(wrapper)
    return engine

def test_attribution_vector_dimensions(xai_setup):
    """
    Verifies that the attribution vector length exactly matches the tokenized prompt length.
    """
    prompt = "Write a Python function to sort an array."
    code = "def sort_array(arr): return sorted(arr)"
    
    result = xai_setup.compute_attribution(prompt, code)
    
    tokens = result["tokens"]
    attributions = result["attributions"]
    
    assert len(tokens) == len(attributions), "Vector dimension mismatch."
    assert tokens[0] == "<s>", "Missing RoBERTa start token."
    assert tokens[-1] == "</s>", "Missing RoBERTa end token."
    assert any(attr != 0.0 for attr in attributions), "Attribution gradients collapsed to zero."