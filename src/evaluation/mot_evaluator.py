"""Official CLEAR MOT and IDF1 Evaluation using `motmetrics`.
Computes standard benchmarks: MOTA, IDF1, IDSW, Precision, Recall, MT, ML, FP, FN.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility shim for motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm

from src.utils.io_utils import load_mot_ground_truth


class MOTEvaluator:
    """Evaluates tracking predictions against ground truth using standard MOT challenge metrics."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate_sequence(
        self,
        gt_df_or_path: Any,
        pred_df_or_path: Any,
        seq_name: str = "Sequence"
    ) -> Dict[str, Any]:
        """Evaluate a single sequence.

        Args:
            gt_df_or_path: Path to gt.txt or loaded pandas DataFrame
            pred_df_or_path: Path to results.txt or loaded pandas DataFrame
            seq_name: Name of sequence for summary table

        Returns:
            Dictionary of metrics and formatted summary table
        """
        if isinstance(gt_df_or_path, (str, os.PathLike)):
            gt_df = load_mot_ground_truth(str(gt_df_or_path))
        else:
            gt_df = gt_df_or_path.copy()

        if isinstance(pred_df_or_path, (str, os.PathLike)):
            if not os.path.exists(str(pred_df_or_path)):
                raise FileNotFoundError(f"Prediction file not found: {pred_df_or_path}")
            pred_df = pd.read_csv(
                str(pred_df_or_path),
                header=None,
                names=[
                    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                    "conf", "x", "y", "z"
                ]
            )
        else:
            pred_df = pred_df_or_path.copy()

        # MOT challenge evaluation filter: class 1 (pedestrian) and conf > 0 in GT
        if "class" in gt_df.columns:
            # Keep pedestrian (1) and static person (7) or ignore non-pedestrians
            gt_df = gt_df[gt_df["class"].isin([1, 2, 7])]
        if "visibility" in gt_df.columns:
            # Evaluation typically evaluates targets with visibility > 0 or all valid
            pass

        accumulator = mm.MOTAccumulator(auto_id=True)

        frames = sorted(list(set(gt_df["frame"].unique()).union(set(pred_df["frame"].unique()))))

        for frame_id in frames:
            frame_gt = gt_df[gt_df["frame"] == frame_id]
            frame_pred = pred_df[pred_df["frame"] == frame_id]

            gt_ids = frame_gt["id"].values.tolist()
            pred_ids = frame_pred["id"].values.tolist()

            if len(frame_gt) > 0:
                gt_boxes = frame_gt[["bb_left", "bb_top", "bb_width", "bb_height"]].values.astype(float)
            else:
                gt_boxes = np.empty((0, 4), dtype=float)

            if len(frame_pred) > 0:
                pred_boxes = frame_pred[["bb_left", "bb_top", "bb_width", "bb_height"]].values.astype(float)
            else:
                pred_boxes = np.empty((0, 4), dtype=float)

            # Compute IoU distance matrix for motmetrics (max_iou=0.5 -> threshold = 1 - 0.5 = 0.5)
            # motmetrics distances.iou_matrix accepts tlwh format!
            dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=self.iou_threshold)

            accumulator.update(
                gt_ids,
                pred_ids,
                dist_matrix
            )

        mh = mm.metrics.create()
        summary = mh.compute(
            accumulator,
            metrics=[
                "mota", "motp", "idf1", "idp", "idr",
                "num_switches", "num_false_positives", "num_misses",
                "mostly_tracked", "mostly_lost", "num_objects", "precision", "recall"
            ],
            name=seq_name
        )

        # Convert to dictionary with clear float types
        metrics_dict = {
            "sequence": seq_name,
            "MOTA": float(summary["mota"].iloc[0] * 100.0),
            "IDF1": float(summary["idf1"].iloc[0] * 100.0),
            "IDSW": int(summary["num_switches"].iloc[0]),
            "Precision": float(summary["precision"].iloc[0] * 100.0) if not pd.isna(summary["precision"].iloc[0]) else 0.0,
            "Recall": float(summary["recall"].iloc[0] * 100.0) if not pd.isna(summary["recall"].iloc[0]) else 0.0,
            "IDP": float(summary["idp"].iloc[0] * 100.0) if not pd.isna(summary["idp"].iloc[0]) else 0.0,
            "IDR": float(summary["idr"].iloc[0] * 100.0) if not pd.isna(summary["idr"].iloc[0]) else 0.0,
            "FP": int(summary["num_false_positives"].iloc[0]),
            "FN": int(summary["num_misses"].iloc[0]),
            "MT": int(summary["mostly_tracked"].iloc[0]),
            "ML": int(summary["mostly_lost"].iloc[0]),
            "Num_Objects": int(summary["num_objects"].iloc[0])
        }

        # Formatted string table
        table_str = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        return {
            "metrics": metrics_dict,
            "summary_df": summary,
            "rendered_table": table_str
        }
