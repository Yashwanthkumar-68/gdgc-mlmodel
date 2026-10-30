"""
Network Intrusion Detection Challenge - Main Entry Point
CLI for running EDA, feature extraction, cross-validation, and submission generation.
"""

import sys
import argparse
from eda import run_eda
from port_deep_dive import run_port_deep_dive
from analyze_features import run_feature_importance
from train_and_evaluate import run_train_and_evaluate
from train_independent import main as run_reproduction

def main():
    parser = argparse.ArgumentParser(description="Network Intrusion Detection - ML Solution")
    parser.add_argument("--eda", action="store_true", help="Run exploratory data analysis")
    parser.add_argument("--ports", action="store_true", help="Run granular port forensic deep dive")
    parser.add_argument("--features", action="store_true", help="Show top domain feature importances")
    parser.add_argument("--train", action="store_true", help="Run 5-fold stratified cross-validation")
    parser.add_argument("--predict", action="store_true", help="Generate verified submission.csv")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Default behavior: run full pipeline overview
        run_eda()
        run_port_deep_dive()
        run_feature_importance()
        run_reproduction()
        return

    if args.eda:
        run_eda()
    if args.ports:
        run_port_deep_dive()
    if args.features:
        run_feature_importance()
    if args.train:
        run_train_and_evaluate()
    if args.predict:
        run_reproduction()

if __name__ == "__main__":
    main()
