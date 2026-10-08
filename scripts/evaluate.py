"""Official MOT Evaluation Script using standard MOTChallenge metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.mot_evaluator import MOTEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking results against ground truth.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth gt.txt file.")
    parser.add_argument("--results", type=str, required=True, help="Path to predicted results.txt file.")
    parser.add_argument("--seq_name", type=str, default="Sequence", help="Name of sequence.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for evaluation.")
    args = parser.parse_args()

    evaluator = MOTEvaluator(iou_threshold=args.iou_thresh)
    res = evaluator.evaluate_sequence(
        gt_df_or_path=args.gt,
        pred_df_or_path=args.results,
        seq_name=args.seq_name
    )

    print("\n" + "=" * 60)
    print(f"MOT EVALUATION RESULTS: {args.seq_name}")
    print("=" * 60)
    print(res["rendered_table"])

    print("\nKey Tracking Metrics Summary:")
    print(f"  MOTA:      {res['metrics']['MOTA']:.2f}%")
    print(f"  IDF1:      {res['metrics']['IDF1']:.2f}%")
    print(f"  IDSW:      {res['metrics']['IDSW']}")
    print(f"  Precision: {res['metrics']['Precision']:.2f}%")
    print(f"  Recall:    {res['metrics']['Recall']:.2f}%")
    print("=" * 60 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(res["metrics"], f, indent=2)
        print(f"Metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
