"""
Train and Evaluate Pipeline
Executes 5-fold Stratified / Group-aware cross-validation,
evaluates Macro F1, precision, recall, and generates classification reports.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from feature_engineering import NetworkFeatureExtractor

TARGET_MAP = {'Benign': 0, 'Availability': 1, 'Reconnaissance': 2, 'Spoofing': 3}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

def run_train_and_evaluate(train_path="data/train.csv"):
    if not os.path.exists(train_path):
        train_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\train.csv"

    print("=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION PIPELINE")
    print("=" * 70)

    train = pd.read_csv(train_path)
    y = train['Target'].map(TARGET_MAP).values

    extractor = NetworkFeatureExtractor()
    X = extractor.fit_transform(train)
    print(f"Extracted {X.shape[1]} domain features for {len(X):,} flows.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(X), 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_probs[val_idx] = clf.predict_proba(X_va)
        fold_f1 = f1_score(y_val, np.argmax(oof_probs[val_idx], axis=1), average='macro')
        print(f"  Fold {fold + 1}/5 | Macro F1: {fold_f1:.5f}")

    overall_f1 = f1_score(y, np.argmax(oof_probs, axis=1), average='macro')
    print("\n" + "=" * 70)
    print(f"OVERALL 5-FOLD OUT-OF-FOLD MACRO F1: {overall_f1:.5f}")
    print("=" * 70)
    print(classification_report(y, np.argmax(oof_probs, axis=1), target_names=list(TARGET_MAP.keys()), digits=5))
    return overall_f1

if __name__ == "__main__":
    run_train_and_evaluate()
