import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import os
from core.evaluator import IntentEvaluator

# Script to generate the ROC curve for the project's performance evaluation.
def plot_roc_curves(output_file="roc_curve_v010.png"):
    evaluator = IntentEvaluator()
    
    # Load the triplet ground truth dataset
    with open('data/ground_truth.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    alphas = [0.3, 0.5, 0.7]
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'cornflowerblue', 'green']
    
    for alpha, color in zip(alphas, colors):
        y_true = []
        y_scores = []
        
        for item in data:
            # Score positive cases
            res_t = evaluator.evaluate(item['prompt'], item['c_true'], alpha=alpha)
            y_true.append(1)
            y_scores.append(res_t['d_intent_final'])
            
            # Score negative cases
            res_f = evaluator.evaluate(item['prompt'], item['c_false'], alpha=alpha)
            y_true.append(0)
            y_scores.append(res_f['d_intent_final'])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'Alpha {alpha} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - IntentTrace-xAI v0.1.0')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_file)
    print(f"Success: ROC Curve saved to {output_file}")

if __name__ == "__main__":
    plot_roc_curves()