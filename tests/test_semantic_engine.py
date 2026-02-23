import pytest
from core.semantic_engine import SemanticEngine
import os
import json

@pytest.fixture(scope="module")
def engine():
    """Loads the model into memory exactly once for the entire test module."""
    return SemanticEngine()

def test_high_semantic_similarity(engine):
    """Verifies high similarity detection when the code matches the intent."""
    prompt = "Write a function to calculate the factorial of a number."
    code = '''
    def factorial(n):
        """Calculates the factorial of a given integer."""
        if n == 0: return 1
        return n * factorial(n-1)
    '''
    score = engine.compute_similarity(prompt, code)
    assert score > 0.6, f"Expected high semantic alignment, got {score}"

def test_low_semantic_similarity(engine):
    """Verifies low similarity detection when the code drastically deviates from the intent."""
    prompt = "Write a function to calculate the factorial of a number."
    code = '''
    def connect_to_database(url):
        # Establish connection to SQL database
        pass
    '''
    score = engine.compute_similarity(prompt, code)
    assert score < 0.4, f"Expected low semantic alignment, got {score}"

def test_fallback_raw_code_similarity(engine):
    """Verifies the fallback mechanism when the code lacks any comments or docstrings."""
    prompt = "Reverse a string"
    code = "def rev(s): return s[::-1]"
    score = engine.compute_similarity(prompt, code)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_ground_truth_semantic_scores(engine):
    """
    Evaluates and outputs the S_cos metric across the 20-triplet dataset.
    Validates the pipeline's stability over the entire ground truth.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'ground_truth.json')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("\n--- Semantic Similarity Scores (S_cos) ---")
    for item in data:
        score_true = engine.compute_similarity(item['prompt'], item['c_true'])
        score_false = engine.compute_similarity(item['prompt'], item['c_false'])
        print(f"[{item['id']}] C_true: {score_true:.4f} | C_false: {score_false:.4f}")
        
    assert len(data) == 20, "Ground truth dataset must contain exactly 20 triplets."