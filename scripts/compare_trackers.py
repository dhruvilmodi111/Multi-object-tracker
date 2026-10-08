"""Compare tracking performance before and after tuning.
Evaluates MOTA, IDF1, ID Switches (IDSW), Precision, and Recall side-by-side.
Generates comparative Markdown tables and visual bar charts.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.mot_evaluator import MOTEvaluator
from scripts.run_tracking import run_tracker


def compare_trackers(
    seq_dir: str,
    baseline_cfg: str,
    tuned_cfg: str,
    gt_file: str,
    output_dir: str = "results",
    use_detections: str = None,
    device: str = "auto"
) -> pd.DataFrame:
    """Run and compare baseline vs tuned tracker on a sequence."""
    os.makedirs(output_dir, exist_ok=True)
    baseline_res_file = os.path.join(output_dir, "baseline_results.txt")
    tuned_res_file = os.path.join(output_dir, "tuned_results.txt")

    # 1. Run Baseline Tracker
    print("\n>>> STEP 1: Running Baseline Tracker...")
    run_tracker(
        seq_dir=seq_dir,
        config_path=baseline_cfg,
        output_results_path=baseline_res_file,
        use_detections_file=use_detections,
        device=device
    )

    # 2. Run Tuned Tracker
    print("\n>>> STEP 2: Running Tuned Occlusion-Aware Tracker...")
    run_tracker(
        seq_dir=seq_dir,
        config_path=tuned_cfg,
        output_results_path=tuned_res_file,
        use_detections_file=use_detections,
        device=device
    )

    # 3. Evaluate Both Trackers
    evaluator = MOTEvaluator(iou_threshold=0.5)
    base_metrics = evaluator.evaluate_sequence(gt_file, baseline_res_file, seq_name="Baseline")["metrics"]
    tuned_metrics = evaluator.evaluate_sequence(gt_file, tuned_res_file, seq_name="Tuned")["metrics"]

    # Build Comparative DataFrame
    metrics_keys = ["MOTA", "IDF1", "IDSW", "Precision", "Recall", "MT", "ML"]
    rows = []
    for k in metrics_keys:
        b_val = base_metrics[k]
        t_val = tuned_metrics[k]
        if k == "IDSW":
            diff = t_val - b_val
            pct_change = f"{-100.0 * (b_val - t_val) / max(1, b_val):+.1f}%" if b_val > 0 else "0.0%"
            b_str = f"{int(b_val)}"
            t_str = f"{int(t_val)}"
        elif k in ["MT", "ML"]:
            b_str = f"{int(b_val)}"
            t_str = f"{int(t_val)}"
            pct_change = f"{t_val - b_val:+d}"
        else:
            diff = t_val - b_val
            pct_change = f"{diff:+.2f}%"
            b_str = f"{b_val:.2f}%"
            t_str = f"{t_val:.2f}%"

        rows.append({
            "Metric": k,
            "Baseline (Standard SORT)": b_str,
            "Tuned (OcclusionDeepSORT)": t_str,
            "Delta / Improvement": pct_change
        })

    df_comp = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("TRACKING PERFORMANCE COMPARISON: BEFORE VS AFTER TUNING")
    print("=" * 70)
    print(df_comp.to_markdown(index=False))
    print("=" * 70)

    # Save Markdown report
    summary_md_path = os.path.join(output_dir, "comparison_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("# Tracking Performance Comparison (Before vs After Tuning)\n\n")
        f.write(f"Sequence: `{Path(seq_dir).name}`\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n\n### Key Takeaways:\n")
        f.write(f"- **IDF1 Improvement:** {base_metrics['IDF1']:.2f}% -> {tuned_metrics['IDF1']:.2f}%\n")
        f.write(f"- **ID Switches (IDSW):** Reduced from {base_metrics['IDSW']} down to {tuned_metrics['IDSW']}\n")
        f.write("- **Occlusion Re-ID:** Lost-track appearance buffering prevents spurious ID recreation.\n")

    print(f"\nSaved comparison summary markdown to: {summary_md_path}")

    # Generate visual bar plot
    try:
        plot_path = os.path.join(output_dir, "comparison_metrics.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Chart 1: IDF1 & MOTA
        categories = ["IDF1 (%)", "MOTA (%)", "Precision (%)", "Recall (%)"]
        b_vals = [base_metrics["IDF1"], base_metrics["MOTA"], base_metrics["Precision"], base_metrics["Recall"]]
        t_vals = [tuned_metrics["IDF1"], tuned_metrics["MOTA"], tuned_metrics["Precision"], tuned_metrics["Recall"]]
        x = range(len(categories))
        width = 0.35

        axes[0].bar([i - width/2 for i in x], b_vals, width=width, label="Baseline", color="#e74c3c")
        axes[0].bar([i + width/2 for i in x], t_vals, width=width, label="Tuned", color="#2ecc71")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, rotation=15)
        axes[0].set_ylim(0, 105)
        axes[0].set_title("Tracking Accuracy & Consistency")
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Chart 2: ID Switches (Lower is better)
        switches = [base_metrics["IDSW"], tuned_metrics["IDSW"]]
        bars = axes[1].bar(["Baseline", "Tuned"], switches, color=["#e74c3c", "#2ecc71"], width=0.5)
        axes[1].set_title("ID Switches (Lower is Better)")
        axes[1].set_ylabel("Total Switches")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, str(int(b.get_height())), ha="center", fontweight="bold")

        # Chart 3: Track Consistency (MT vs ML)
        mt_ml = ["Mostly Tracked (MT)", "Mostly Lost (ML)"]
        b_mt = [base_metrics["MT"], base_metrics["ML"]]
        t_mt = [tuned_metrics["MT"], tuned_metrics["ML"]]
        x_m = range(len(mt_ml))
        axes[2].bar([i - width/2 for i in x_m], b_mt, width=width, label="Baseline", color="#e74c3c")
        axes[2].bar([i + width/2 for i in x_m], t_mt, width=width, label="Tuned", color="#2ecc71")
        axes[2].set_xticks(x_m)
        axes[2].set_xticklabels(mt_ml)
        axes[2].set_title("Trajectory Continuity")
        axes[2].legend()
        axes[2].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Generated comparison visualization chart: {plot_path}")
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")

    return df_comp


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Tuned Tracker.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to sequence directory.")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth gt.txt file.")
    parser.add_argument("--baseline_config", type=str, default="configs/baseline_config.yaml", help="Baseline config.")
    parser.add_argument("--tuned_config", type=str, default="configs/tuned_config.yaml", help="Tuned config.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output results.")
    parser.add_argument("--use_detections", type=str, default=None, help="Optional detections file.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    args = parser.parse_args()

    compare_trackers(
        seq_dir=args.sequence,
        baseline_cfg=args.baseline_config,
        tuned_cfg=args.tuned_config,
        gt_file=args.gt,
        output_dir=args.output_dir,
        use_detections=args.use_detections,
        device=args.device
    )


if __name__ == "__main__":
    main()
