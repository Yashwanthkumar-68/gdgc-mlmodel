"""
Exploratory Data Analysis (EDA) Module
Analyzes:
- Class imbalance across the 4 threat classes
- Train vs Test distribution shifts (Port 21 0-duration trap, Port 5353 multicast)
- Protocol and service distribution shifts
"""

import os
import pandas as pd
import numpy as np

def run_eda(train_path="data/train.csv", test_path="data/test.csv"):
    if not os.path.exists(train_path):
        train_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\train.csv"
    if not os.path.exists(test_path):
        test_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\test.csv"

    print("=" * 70)
    print("NETWORK INTRUSION DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"\n[1] DATASET DIMENSIONS:")
    print(f"  Training set : {train.shape[0]:,} rows x {train.shape[1]} columns")
    print(f"  Evaluation set: {test.shape[0]:,} rows x {test.shape[1]} columns")

    print(f"\n[2] TARGET CLASS DISTRIBUTION (Macro F1 = Equal 25% weight):")
    counts = train['Target'].value_counts()
    percentages = train['Target'].value_counts(normalize=True) * 100
    for cls in counts.index:
        print(f"  {cls:<15s}: {counts[cls]:>6,} rows ({percentages[cls]:>5.2f}%)")

    print(f"\n[3] CORE SERVICE PORT DISTRIBUTION (Train vs Test):")
    ports = [21, 53, 80, 443, 1883, 5353]
    for p in ports:
        tr_cnt = (train['id.resp_p'] == p).sum()
        te_cnt = (test['id.resp_p'] == p).sum()
        tr_pct = tr_cnt / len(train) * 100
        te_pct = te_cnt / len(test) * 100
        print(f"  Port {p:<5d} : Train={tr_cnt:>6,} ({tr_pct:>5.2f}%) | Test={te_cnt:>6,} ({te_pct:>5.2f}%)")

    print(f"\n[4] THE ZERO-DURATION DoS TRAP (Port 21 Analysis):")
    tr_p21_0bwd = ((train['id.resp_p'] == 21) & (train['bwd_pkts_tot'] == 0)).sum()
    tr_p21_1bwd = ((train['id.resp_p'] == 21) & (train['bwd_pkts_tot'] == 1)).sum()
    te_p21_0bwd = ((test['id.resp_p'] == 21) & (test['bwd_pkts_tot'] == 0)).sum()
    te_p21_1bwd = ((test['id.resp_p'] == 21) & (test['bwd_pkts_tot'] == 1)).sum()
    print(f"  Train Port 21: bwd_pkts==1 (RST reply)={tr_p21_1bwd:,} | bwd_pkts==0={tr_p21_0bwd:,}")
    print(f"  Test Port 21 : bwd_pkts==1 (RST reply)={te_p21_1bwd:,} | bwd_pkts==0={te_p21_0bwd:,}")
    print("=" * 70)

if __name__ == "__main__":
    run_eda()
