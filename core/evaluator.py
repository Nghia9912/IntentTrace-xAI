from typing import Dict, List
from core.static_analyzer import DeadCodeDetector
from core.semantic_engine import SemanticEngine
from core.xai_engine import XAIEngine
from models.hf_wrapper import CodeBERTWrapper

class IntentEvaluator:
    """
    The central aggregation hub that unifies Static Analysis, Semantic Embeddings, 
    and xAI Attribution into a single confidence metric.
    """
    def __init__(self):
        # Initialize the pipelines
        self.semantic_engine = SemanticEngine()
        self.codebert_wrapper = CodeBERTWrapper()
        self.xai_engine = XAIEngine(self.codebert_wrapper)

    def normalize_attributions(self, attributions: List[float]) -> float:
        """
        Normalizes the attribution vector into a single scalar concentration coefficient.
        """
        sum_positive = sum(max(attr, 0.0) for attr in attributions)
        sum_absolute = sum(abs(attr) for attr in attributions)
        
        # Guard against zero-division if the model detects zero correlation
        if sum_absolute == 0.0:
            return 0.0
        return sum_positive / sum_absolute

    def evaluate(self, prompt: str, code: str, alpha: float = 0.5) -> Dict:
        """
        Executes the full multi-pipeline evaluation framework.
        """
        # 1. Static Analysis Pipeline (O(V+E) complexity)
        static_detector = DeadCodeDetector(code)
        dead_nodes = static_detector.analyze()
        has_dead_code = len(dead_nodes) > 0

        # 2. Semantic Similarity Pipeline
        s_cos = self.semantic_engine.compute_similarity(prompt, code)

        # 3. Zero-Shot xAI Attribution Pipeline
        xai_result = self.xai_engine.compute_attribution(prompt, code)
        phi_hat_ig = self.normalize_attributions(xai_result["attributions"])

        # 4. Aggregation Hub
        d_intent_final = (alpha * phi_hat_ig) + ((1.0 - alpha) * s_cos)
        
        # Hard penalty logic: Dead code explicitly indicates severe hallucination
        if has_dead_code:
            d_intent_final *= 0.1

        return {
            "has_dead_code": has_dead_code,
            "dead_code_count": len(dead_nodes),
            "s_cos": s_cos,
            "phi_hat_ig": phi_hat_ig,
            "d_intent_final": d_intent_final,
            "alpha": alpha
        }