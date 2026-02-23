import ast
import re
from typing import List
from sentence_transformers import SentenceTransformer, util

class SemanticEngine:
    """
    Computes semantic similarity between natural language intents and code blocks
    using a pre-trained sentence-transformer model (optimized for CPU/Latency).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Force CPU execution to reserve GPU VRAM for the xAI Engine in Sprint 3
        self.model = SentenceTransformer(model_name, device='cpu')

    def extract_semantics_from_code(self, source_code: str) -> str:
        """Extracts semantic signals (docstrings and inline comments) from source code."""
        semantics: List[str] = []
        
        # 1. Extract Docstrings via AST
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        semantics.append(docstring.strip())
        except SyntaxError:
            pass # Ignore AST errors here; handled by the Static Analyzer pipeline
            
        # 2. Extract Inline Comments via Regex
        comments = re.findall(r'#.*', source_code)
        semantics.extend([c.strip('# ').strip() for c in comments])
        
        return " ".join(semantics) if semantics else ""

    def compute_similarity(self, prompt: str, source_code: str) -> float:
        """
        Calculates Cosine Similarity between the prompt and the code's semantic signals.
        Returns a scalar bounded between [-1.0, 1.0].
        """
        code_semantics = self.extract_semantics_from_code(source_code)
        
        # Fallback: Directly compare raw source code if no metadata exists
        target_text = code_semantics if code_semantics else source_code
        
        # Encode into Tensor Embeddings
        embedding_prompt = self.model.encode(prompt, convert_to_tensor=True)
        embedding_code = self.model.encode(target_text, convert_to_tensor=True)
        
        # Compute Cosine Similarity
        cosine_score = util.cos_sim(embedding_prompt, embedding_code)
        
        return float(cosine_score[0][0])