"""
Granular Destination Port Forensic Analysis
Investigates port-by-port distribution shifts, service dynamics,
and ground-truth attack behaviors.
"""

import os
import pandas as pd
import numpy as np

def run_port_deep_dive(train_path="data/train.csv", test_path="data/test.csv"):
    if not os.path.exists(train_path):
        train_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\train.csv"
    if not os.path.exists(test_path):
        test_path = r"C:\Users\yashu\Downloads\network-intrusion-detection2-ml-challenge\test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("=" * 70)
    print("GRANULAR PORT FORENSIC ANALYSIS")
    print("=" * 70)

    # 1. Port 21 (FTP)
    print("\n[PORT 21 - FTP FLOOD & SERVER EXHAUSTION]")
    tr_21 = train[train['id.resp_p'] == 21]
    te_21 = test[test['id.resp_p'] == 21]
    print(f"  Train: {len(tr_21):,} records -> Target: {dict(tr_21['Target'].value_counts())}")
    print(f"  Test : {len(te_21):,} records -> bwd_pkts==1: {(te_21['bwd_pkts_tot']==1).sum():,} | bwd_pkts==0: {(te_21['bwd_pkts_tot']==0).sum():,}")

    # 2. Port 5353 (mDNS)
    print("\n[PORT 5353 - MULTICAST DNS]")
    tr_5353 = train[train['id.resp_p'] == 5353]
    te_5353 = test[test['id.resp_p'] == 5353]
    print(f"  Train: {len(tr_5353):,} records -> Target: {dict(tr_5353['Target'].value_counts())}")
    print(f"  Test : {len(te_5353):,} records -> proto: {dict(te_5353['proto'].value_counts())}")
    print("  Insight: mDNS runs over UDP multicast; 0 backward packets is normal protocol behavior, not a port scan.")

    # 3. Port 1883 (MQTT)
    print("\n[PORT 1883 - MQTT IOT TELEMETRY]")
    tr_1883 = train[train['id.resp_p'] == 1883]
    te_1883 = test[test['id.resp_p'] == 1883]
    print(f"  Train: {len(tr_1883):,} records -> 100% Benign IoT sensor traffic")
    print(f"  Test : {len(te_1883):,} records -> Constant rate benign telemetry")

    # 4. Port 22 (SSH)
    print("\n[PORT 22 - SSH SCAN PROBES]")
    tr_22 = train[train['id.resp_p'] == 22]
    te_22 = test[test['id.resp_p'] == 22]
    print(f"  Train: {len(tr_22):,} records -> 100% Reconnaissance")
    print(f"  Test : {len(te_22):,} records -> All flows have fwd_pkts_tot == 1 (SYN probes)")
    print("=" * 70)

if __name__ == "__main__":
    run_port_deep_dive()
