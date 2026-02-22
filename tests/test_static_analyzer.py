import os
import json
import ast
import pytest
from core.static_analyzer import DeadCodeDetector

def test_syntax_error_handling():
    """Validates fail-fast behavior when encountering syntactically invalid code."""
    invalid_code = "def foo():\n    return 1 +"
    with pytest.raises(SyntaxError):
        DeadCodeDetector(invalid_code)

def test_unreachable_code_after_return():
    """Verifies the detection of dead code located after a return statement."""
    code = """
def process(x):
    return x * 2
    print("LLM Hallucination") # Unreachable
    x = 10
"""
    detector = DeadCodeDetector(code)
    results = detector.analyze()
    
    assert len(results) == 2, "Must detect exactly 2 lines of dead code."
    assert results[0]['line'] == 4
    assert results[1]['line'] == 5

def test_clean_code():
    """Ensures valid code does not trigger false positive warnings."""
    code = """
def is_even(n):
    if n % 2 == 0:
        return True
    return False
"""
    detector = DeadCodeDetector(code)
    results = detector.analyze()
    assert len(results) == 0, "Clean code must not contain dead code."

def test_ground_truth_syntax_integrity():
    """
    Reads the JSON file and verifies that all 40 code snippets (20 c_true, 20 c_false) 
    can be successfully compiled into an AST without throwing a SyntaxError.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'ground_truth.json')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        try:
            DeadCodeDetector(item['c_true'])
        except SyntaxError as e:
            pytest.fail(f"Syntax error in C_true of {item['id']}: {e}")
            
        try:
            DeadCodeDetector(item['c_false'])
        except SyntaxError as e:
            pytest.fail(f"Syntax error in C_false of {item['id']}: {e}")