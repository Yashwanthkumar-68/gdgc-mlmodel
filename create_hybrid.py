"""
Hybrid Multi-Model Ensemble Module
Blends diverse tree architectures (CatBoost, XGBoost, ExtraTrees, LightGBM)
and applies protocol domain alignment for physical network consistency.
"""

import os
import numpy as np
import pandas as pd
from feature_engineering import NetworkFeatureExtractor

TARGET_MAP = {'Benign': 0, 'Availability': 1, 'Reconnaissance': 2, 'Spoofing': 3}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

def create_hybrid_ensemble(cb_weight=0.35, xgb_weight=0.35, et_weight=0.15, rf_weight=0.15):
    """
    Creates a robust hybrid ensemble combining symmetric oblivious trees,
    histogram gradient boosting, and randomized bagging.
    """
    models_dir = "models"
    cb_probs = np.load(os.path.join(models_dir, "catboost_fe_test_probs.npy"))
    xgb_probs = np.load(os.path.join(models_dir, "xgboost_fe_test_probs.npy"))
    et_probs = np.load(os.path.join(models_dir, "extra_trees_fe_test_probs.npy"))
    rf_probs = np.load(os.path.join(models_dir, "random_forest_fe_test_probs.npy"))

    blend = (cb_weight * cb_probs +
             xgb_weight * xgb_probs +
             et_weight * et_probs +
             rf_weight * rf_probs)

    preds_int = np.argmax(blend, axis=1)
    preds = [INV_TARGET_MAP[i] for i in preds_int]
    preds_series = pd.Series(preds)

    # Protocol Domain Invariants
    test_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\test.csv"
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        # Port 5353 mDNS multicast spoofing
        p5353_mask = (test['id.resp_p'] == 5353) & (preds_series == 'Reconnaissance')
        preds_series.loc[p5353_mask] = 'Spoofing'

        # Port 22 SSH single-packet SYN scan probes
        p22_mask = (test['id.resp_p'] == 22)
        preds_series.loc[p22_mask] = 'Reconnaissance'

    return preds_series

if __name__ == "__main__":
    print("Creating hybrid ensemble...")
    preds = create_hybrid_ensemble()
    print("Hybrid class counts:")
    print(preds.value_counts())
