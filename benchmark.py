import os
import json
from sklearn.metrics import roc_auc_score
from core.evaluator import IntentEvaluator

def run_benchmark():
    print("Initializing IntentTrace-xAI Benchmark Protocol...")
    evaluator = IntentEvaluator()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'ground_truth.json')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    alphas = [0.3, 0.5, 0.7]
    results = {alpha: {"y_true": [], "y_scores": []} for alpha in alphas}

    print(f"Scanning {len(data)} triplets. This requires heavy tensor computation...\n")

    for idx, item in enumerate(data):
        prompt = item['prompt']
        for alpha in alphas:
            res_true = evaluator.evaluate(prompt, item['c_true'], alpha=alpha)
            results[alpha]["y_true"].append(1)
            results[alpha]["y_scores"].append(res_true["d_intent_final"])
            
            res_false = evaluator.evaluate(prompt, item['c_false'], alpha=alpha)
            results[alpha]["y_true"].append(0)
            results[alpha]["y_scores"].append(res_false["d_intent_final"])
        print(f"Processed [{idx+1}/{len(data)}]: {item['id']}")

    print("\n=== AUC-ROC Report v0.1.0 ===")
    for alpha in alphas:
        auc = roc_auc_score(results[alpha]["y_true"], results[alpha]["y_scores"])
        status = "PASSED" if auc >= 0.70 else "FAILED"
        print(f"Alpha = {alpha:.1f} | AUC-ROC = {auc:.4f} [{status}]")

if __name__ == "__main__":
    run_benchmark()