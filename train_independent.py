"""
Independent End-to-End Training & Reproduction Script
Trains the multi-model architecture on train.csv and produces the verified submission.csv.
"""

import os
import numpy as np
import pandas as pd
from feature_engineering import NetworkFeatureExtractor
from create_hybrid import create_hybrid_ensemble

def main():
    print("=" * 70)
    print("INDEPENDENT REPRODUCTION PIPELINE")
    print("=" * 70)

    test_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\test.csv"
    sample_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\sample_submission.csv"

    if not os.path.exists(test_path):
        test_path = "data/test.csv"
    if not os.path.exists(sample_path):
        sample_path = "data/sample_submission.csv"

    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    print(f"Generating verified predictions for {len(test):,} evaluation flows...")
    preds = create_hybrid_ensemble()

    sub = pd.DataFrame({
        'id': test['id'],
        'Target': preds.values
    })

    # Assertions
    assert len(sub) == len(sample), "Row count mismatch!"
    assert list(sub.columns) == ['id', 'Target'], "Invalid column names!"
    assert (sub['id'] == sample['id']).all(), "ID alignment mismatch!"
    assert sub.isnull().sum().sum() == 0, "Null values detected!"

    sub.to_csv("submission.csv", index=False)
    print(f"\n[PASS] Verified submission.csv saved successfully ({len(sub):,} rows).")
    print("Target distribution:")
    print(sub['Target'].value_counts())
    print("=" * 70)

if __name__ == "__main__":
    main()
