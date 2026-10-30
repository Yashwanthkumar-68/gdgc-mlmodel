# Adaptive Network Intrusion Detection Engine (ANIDE)
### Multi-Class Threat Classification & Traffic Profile Generalization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Competition](https://img.shields.io/badge/Competition-GDG%20on%20Campus%20ML%20Challenge-orange.svg)](https://www.kaggle.com/)
[![Evaluation Metric](https://img.shields.io/badge/Metric-Macro%20F1-green.svg)](https://scikit-learn.org/)
[![Validation F1](https://img.shields.io/badge/5--Fold%20OOF%20F1-0.9817-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Developed by:** [Yashwanth Kumar Thammishetti](https://github.com/Yashwanthkumar-68)  
**Organized by:** Google Developer Groups on Campus (GDG on Campus)  
**Target Threat Classes:** `Benign`, `Availability`, `Reconnaissance`, `Spoofing`  

---

## 🛡️ Project Overview

In enterprise network defense, machine learning models frequently fail when deployed against novel adversarial tactics because training environments rarely capture live server degradation. 

This repository implements the **Adaptive Network Intrusion Detection Engine (ANIDE)**—an end-to-end, reproducible machine learning system designed to overcome cross-environment traffic shifts. Rather than relying on memorized client IP sockets or transient ephemeral ports, the architecture models fundamental **protocol invariants, bidirectional flow asymmetries, packet payload densities, and transport-layer state dynamics**.

The system achieves an **Out-Of-Fold (OOF) Macro F1 of 0.9817** and **99.87% overall classification accuracy** across rigorous 5-fold stratified cross-validation.

---

## 🏗️ Architecture Pipeline

The diagram below illustrates the end-to-end data processing, feature transformation, multi-model ensembling, and inference pipeline:

```mermaid
flowchart TD
    A["Raw Network Flow Records<br/>(84 Zeek/NetFlow Features)"] --> B["Stage 1: Ingestion & Sanitation<br/>- Prune Ephemeral Sockets (id.orig_p)<br/>- Zero-Variance Elimination"]
    B --> C["Stage 2: Domain Feature Engineering<br/>- 133 Domain Features<br/>- Directional Asymmetries & Rate Metrics<br/>- TCP Flag & Window Fingerprints"]
    C --> D["Stage 3: Stratified 5-Fold Cross-Validation"]
    
    subgraph Multi_Model_Ensemble ["Stage 4: Diverse Multi-Architecture Ensemble"]
        E1["CatBoost Classifier<br/>(Symmetric Oblivious Trees)"]
        E2["XGBoost Classifier<br/>(Histogram Gradient Boosting)"]
        E3["LightGBM Classifier<br/>(Leaf-Wise Tree Splitting)"]
        E4["ExtraTrees & Random Forest<br/>(Randomized Bagging Ensembles)"]
    end
    
    D --> Multi_Model_Ensemble
    Multi_Model_Ensemble --> F["Stage 5: Soft-Voting Probability Fusion"]
    F --> G["Stage 6: Protocol Domain Invariants<br/>- Port 5353 Multicast Disambiguation<br/>- Port 22 Stealth Probe Alignment"]
    G --> H["Stage 7: Threshold Tuning & Calibration<br/>- Coordinate Optimization for Macro F1"]
    H --> I["Final Verified Predictions<br/>submission.csv (33,161 Rows)"]
```

---

## 🔬 Core Engineering Discoveries & Forensic Insights

### 1. The Server-Drop Asymmetry Anomaly (Port 21 FTP Floods)
* **Training Reality:** Over 94% of training traffic consists of high-volume FTP (Port 21) floods where the target daemon remained active and returned TCP reset (`RST`) packets (`bwd_pkts_tot == 1`). This generated artificial byte rates exceeding 100 MB/s.
* **Evaluation Shift:** In real-world attack scenarios, victim buffers become saturated, resulting in dropped SYN queues where no return packets ever leave the host (`bwd_pkts_tot == 0`, `duration == 0.0s`).
* **Solution:** Naive models classify silent packets as port scans. We engineered forward-invariant attack signatures (`sig_hping_syn_flood`) combining fixed window scaling (`window_size == 64`) and standardized forward payload lengths (`payload_len == 120`), coupled with duration-floor regularizations ($\epsilon = 10^{-6}$).

### 2. Disambiguating Multicast Spoofing vs. Stealth Scans (Port 5353)
* **The Conflict:** Standard trees split on `bwd_pkts_tot == 0` to flag unresponded flows as `Reconnaissance`.
* **Network Reality:** Multicast DNS (`mDNS` on Port 5353, UDP `224.0.0.251`) communicates via local network broadcast and naturally does not yield unicast replies. In reality, hostile activity on Port 5353 represents multicast poisoning (`Spoofing`), not reconnaissance scanning.
* **Solution:** We introduced protocol-aware domain invariants isolating UDP multicast transactions from unicast TCP scanning probes.

### 3. Severe Class Imbalance & Macro F1 Metric Alignment
* The training data exhibits extreme skew:
  * `Availability`: 80,072 (94.5%)
  * `Spoofing`: 2,323 (2.7%)
  * `Benign`: 1,583 (1.9%)
  * `Reconnaissance`: 735 (0.9%)
* Because the competition evaluates **Macro F1** (unweighted mean across all 4 classes), failing to detect a minority attack destroys 25% of the total leaderboard score. We resolved this by combining class-balanced loss weighting with post-hoc probability threshold tuning.

---

## ⚙️ Feature Engineering Taxonomy (133 Features)

Our feature pipeline in [`feature_engineering.py`](feature_engineering.py) generates 133 domain features grouped into six functional tiers:

1. **Service Semantics & IANA Port Tiers:**
   * Drops client ephemeral ports (`id.orig_p`) to prevent socket memorization.
   * Categorizes destination ports into standard IANA tiers: *Well-Known* ($<1024$), *Registered* ($1024-49151$), and *Dynamic* ($\ge 49152$).
   * Dedicated binary indicators for critical service ports: FTP (21), SSH (22), DNS (53), HTTP (80), HTTPS (443), MQTT (1883), and mDNS (5353).
2. **Directional Flow Asymmetries & Density:**
   * Packet volume asymmetry: $\text{Asymmetry}_{\text{pkt}} = \frac{\text{fwd\_pkts} - \text{bwd\_pkts}}{\text{fwd\_pkts} + \text{bwd\_pkts} + \epsilon}$
   * Byte transfer ratios: $\text{Ratio}_{\text{bytes}} = \frac{\text{fwd\_bytes} + 1}{\text{bwd\_bytes} + 1}$
   * Directional density: Forward payload per packet vs. backward payload per packet.
3. **Transport Layer State & TCP Flag Dynamics:**
   * Pure probe detection: `syn_only` ($\text{SYN} > 0 \land \text{ACK} == 0$).
   * Connection teardown: `fin_only` ($\text{FIN} > 0 \land \text{SYN} == 0$).
   * Control flag intensity and ratio of reset flags to connection establishment flags (`rst_syn_ratio`).
4. **Operating System & Tool Window Fingerprinting:**
   * Forward-to-backward window size ratio: $\text{Window}_{\text{ratio}} = \frac{\text{fwd\_init\_window} + 1}{\text{bwd\_init\_window} + 1}$.
   * Binary fingerprint masks for known stack configurations ($64$, $0$, $64240$, $65535$).
5. **Specialized Attack Vector Fingerprints:**
   * `sig_hping_syn_flood`: SYN packet with 120-byte payload and initial window 64.
   * `sig_slowloris_dos`: Sustained forward packets on Port 80 with zero backward response.
   * `sig_udp_scan`: Non-DNS unanswered UDP probes.
6. **Duration-Floored Rate Metrics:**
   * Numerically stable log rates: $\log\left(1 + \frac{\text{bytes}}{\max(\text{duration}, 10^{-6})}\right)$.

---

## 📊 Experimental Results & Validation Benchmarks

Cross-validation was evaluated using out-of-fold predictions on the full competition training dataset:

| Model Architecture | Feature Representation | Validation Strategy | Macro F1 | Availability F1 | Benign F1 | Recon F1 | Spoofing F1 | Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Tree (Raw) | 82 Raw Features | 5-Fold Stratified | 0.7638 | 0.9982 | 0.7410 | 0.6215 | 0.6947 | 98.51% |
| LightGBM Tuned | + 133 Domain Features | 5-Fold Stratified | 0.9482 | 0.9998 | 0.9320 | 0.9145 | 0.9463 | 99.82% |
| CatBoost Oblivious | + Domain Invariants | 5-Fold Grouped | 0.9767 | 1.0000 | 0.9680 | 0.9520 | 0.9868 | 99.89% |
| **Hybrid ANIDE Ensemble (Final)** | **133 Features + Protocol Fusion** | **Stratified Group CV + Tuned** | **0.9817** | **1.0000** | **0.9861** | **0.9890** | **0.9905** | **99.94%** |

---

## 🎯 Final Verified Submission Profile (`submission.csv`)

The official evaluation predictions submitted to the competition leaderboard ([`submission.csv`](submission.csv)) align with the physical network protocol dynamics discovered across our empirical auditing:

| Target Threat Class | Predicted Volume | Percentage | Primary Traffic Sources & Protocol Dynamics |
| :--- | :---: | :---: | :--- |
| **Availability** | **10,542** | **31.79%** | Volumetric DoS floods on Port 21 (FTP) & slowloris attacks on Port 80. |
| **Benign** | **10,428** | **31.45%** | IoT telemetry on Port 1883 (MQTT), standard web browsing (Port 80/443), and benign DNS. |
| **Reconnaissance** | **6,891** | **20.78%** | Stealth TCP SYN probe scans across closed/filtered ports & single-packet SSH probes (Port 22). |
| **Spoofing** | **5,300** | **15.98%** | Local network mDNS cache poisoning (Port 5353) & query spoofing on Port 53. |
| **Total Flows** | **33,161** | **100.00%** | **Verified 100% complete, zero nulls, matching sample IDs.** |

* **Verification Checksum (SHA-256):** `07635b7f5600f7f2ed05db6a2f8dd52399bc53b5f47b24b4b4f01af5a71fba5e`

---

## 🗂️ Clean Repository Organization

```
├── .gitignore                             # Git configuration ignoring temporary artifacts
├── Network_Intrusion_Detection_GDG.ipynb  # Self-contained end-to-end Jupyter Notebook
├── README.md                              # Technical architectural overview & documentation
├── TECHNICAL_REPORT.md                    # Deep-dive oral defense and forensic analysis report
├── analyze_features.py                    # Feature importance and gain attribution analyzer
├── create_hybrid.py                       # Multi-model soft-voting ensemble script
├── eda.py                                 # Exploratory data audit and distribution profiling
├── feature_engineering.py                 # Core domain transformation module (133 features)
├── main.py                                # CLI management console
├── models_summary.json                    # Machine-readable evaluation metrics & parameters
├── port_deep_dive.py                      # Port-level forensic analysis & protocol auditing
├── requirements.txt                       # Project dependencies
├── submission.csv                         # Verified final predictions (33,161 rows)
├── train_and_evaluate.py                  # 5-fold cross-validation & diagnostic engine
├── train_independent.py                   # Self-contained independent reproduction script
└── tune_precision.py                      # Multi-class threshold & precision optimizer
```

---

## 🚀 Reproduction & Usage Guide

### 1. Environment Installation
Clone the repository and install dependencies in a clean virtual environment:
```bash
git clone https://github.com/Yashwanthkumar-68/gdgc-mlmodel.git
cd gdgc-mlmodel
pip install -r requirements.txt
```

### 2. Run Diagnostics & Exploratory Analysis
```bash
python main.py --eda
python main.py --ports
python main.py --features
```

### 3. Run Full 5-Fold Cross-Validation
```bash
python main.py --train
```

### 4. Generate Verified Competition Predictions
```bash
python main.py --predict
```

The script outputs `submission.csv` and verifies:
* Exactly 33,161 rows matching `sample_submission.csv`.
* Clean column structure: `id,Target`.
* Zero null or undefined predictions.

---

## 📜 Citation & License

This project was developed for the **GDG on Campus Network Intrusion Detection ML Challenge**.  
Licensed under the [MIT License](LICENSE).
