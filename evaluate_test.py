"""
evaluate_test.py
================
Standalone script to evaluate the pipeline against KDDTest+.arff (Change 3).
This was previously impossible because test_path was never used.

Usage:
    python evaluate_test.py
    python evaluate_test.py --n 500 --test-path KDDTest+.arff

Outputs a JSON evaluation report to results/eval_<timestamp>.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pipeline on KDDTest+.arff'
    )
    parser.add_argument('--train-path', default=os.getenv('KDD_DATASET_PATH', 'KDDTrain+.txt'))
    parser.add_argument('--test-path',  default=os.getenv('KDD_TEST_PATH',    'KDDTest+.arff'))
    parser.add_argument('--n',          type=int, default=200,
                        help='Number of test records to evaluate (default: 200)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  NSL-KDD TEST SET EVALUATION")
    print(f"  Train : {args.train_path}")
    print(f"  Test  : {args.test_path}")
    print(f"  N     : {args.n}")
    print("="*70 + "\n")

    # Validate files exist
    if not os.path.exists(args.train_path):
        print(f"ERROR: Training file not found: {args.train_path}")
        print("  Set KDD_DATASET_PATH or place KDDTrain+.txt in this folder.")
        sys.exit(1)
    if not os.path.exists(args.test_path):
        print(f"ERROR: Test file not found: {args.test_path}")
        print("  Set KDD_TEST_PATH or place KDDTest+.arff in this folder.")
        sys.exit(1)

    from nsl_kdd_loader import NSLKDDLoader
    from llm_analysis import LLMThreatAnalyzer
    from config import SystemConfig

    loader   = NSLKDDLoader(args.train_path, test_path=args.test_path)
    analyzer = LLMThreatAnalyzer(SystemConfig.LLM_CONFIG)

    logger.info("Starting evaluation…")
    report = loader.evaluate_on_test(analyzer, n_samples=args.n, seed=args.seed)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"  Records evaluated : {report['n_evaluated']} "
          f"(normal={report['n_normal']}, attack={report['n_attack']})")
    print(f"  Accuracy          : {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"  Precision         : {report['precision']:.4f}")
    print(f"  Recall            : {report['recall']:.4f}")
    print(f"  F1 Score          : {report['f1_score']:.4f}")
    print(f"  False Positive Rate: {report['false_positive_rate']:.4f}")
    print(f"  TP={report['tp']} FP={report['fp']} TN={report['tn']} FN={report['fn']}")
    print("="*70 + "\n")

    # ── Save report ───────────────────────────────────────────────────────────
    out_file = f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Full report saved to: {out_file}")


if __name__ == '__main__':
    main()
