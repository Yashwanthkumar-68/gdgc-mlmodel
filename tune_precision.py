"""
Multiclass Probability Threshold Optimizer
Tunes decision thresholds / class weights to maximize Macro F1
and balance minority class precision and recall.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

class MulticlassThresholdTuner:
    """
    Optimizes class probability multipliers using coordinate ascent / Nelder-Mead.
    Guarantees no evaluation leakage by optimizing strictly on out-of-fold predictions.
    """
    def __init__(self, n_classes=4):
        self.n_classes = n_classes
        self.weights_ = np.ones(n_classes)

    def fit(self, oof_probs, y_true):
        def objective(weights):
            scaled_probs = oof_probs * weights
            preds = np.argmax(scaled_probs, axis=1)
            macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            return -macro_f1

        initial_weights = np.ones(self.n_classes)
        res = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-4}
        )
        self.weights_ = res.x / np.sum(res.x) * self.n_classes
        return self

    def predict(self, probs):
        scaled_probs = probs * self.weights_
        return np.argmax(scaled_probs, axis=1)

if __name__ == "__main__":
    print("MulticlassThresholdTuner module loaded successfully.")
