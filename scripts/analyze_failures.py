"""CLI Script for Automated Tracking Failure Case and ID Switch Root Cause Analysis.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.failure_analyzer import FailureAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze tracking failure cases and categorize ID switches.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth gt.txt.")
    parser.add_argument("--results", type=str, required=True, help="Path to tracking predictions results.txt.")
    parser.add_argument("--output_report", type=str, default="results/failure_analysis_report.md", help="Output markdown report path.")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--occlusion_thresh", type=float, default=0.15, help="Overlap threshold for occlusion flag.")
    parser.add_argument("--velocity_thresh", type=float, default=25.0, help="Velocity threshold (px/frame) for fast motion.")
    args = parser.parse_args()

    analyzer = FailureAnalyzer(
        iou_match_thresh=args.iou_thresh,
        occlusion_iou_thresh=args.occlusion_thresh,
        fast_motion_velocity_thresh=args.velocity_thresh
    )

    results = analyzer.analyze(
        gt_df_or_path=args.gt,
        pred_df_or_path=args.results,
        output_report_path=args.output_report
    )

    print("\n" + "=" * 65)
    print("FAILURE CASE AND ID SWITCH ROOT CAUSE ANALYSIS")
    print("=" * 65)
    print(f"Total Identified ID Switches: {results['total_switches']}\n")
    print("Breakdown by Root Cause:")
    print(results["cause_breakdown"].to_string(index=False))
    print("=" * 65)
    print(f"Detailed diagnostics and incident log written to: {args.output_report}\n")


if __name__ == "__main__":
    main()
