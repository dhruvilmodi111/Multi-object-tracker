"""Automated Failure Analysis and ID-Switch Root-Cause Categorization Engine.
Categorizes ID switches into:
- Visual Occlusion / Crossing Paths
- Missed Detection (False Negative gap)
- Fast Motion / Maneuver
- Similar Appearance / Ambiguous Matching
"""

from typing import Dict, List, Any, Tuple, Optional
import os
import numpy as np
import pandas as pd

from src.utils.io_utils import load_mot_ground_truth
from src.utils.box_utils import compute_iou_matrix, compute_inter_min_matrix


class FailureAnalyzer:
    """Analyzes tracking results vs ground truth and categorizes ID switches."""

    def __init__(
        self,
        iou_match_thresh: float = 0.5,
        occlusion_iou_thresh: float = 0.15,
        fast_motion_velocity_thresh: float = 25.0,  # pixels per frame
    ):
        self.iou_match_thresh = iou_match_thresh
        self.occlusion_iou_thresh = occlusion_iou_thresh
        self.fast_motion_velocity_thresh = fast_motion_velocity_thresh

    def analyze(
        self,
        gt_df_or_path: Any,
        pred_df_or_path: Any,
        output_report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive failure case and ID switch root cause analysis.

        Returns:
            Dict containing switch incidents list, cause breakdown summary, and markdown report.
        """
        if isinstance(gt_df_or_path, (str, os.PathLike)):
            gt_df = load_mot_ground_truth(str(gt_df_or_path))
        else:
            gt_df = gt_df_or_path.copy()

        if isinstance(pred_df_or_path, (str, os.PathLike)):
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

        # Frame-by-frame mapping of GT ID -> Pred ID
        frames = sorted(list(set(gt_df["frame"].unique()).intersection(set(pred_df["frame"].unique()))))

        gt_pred_mapping: Dict[int, Dict[int, int]] = {}  # frame -> {gt_id: pred_id}
        gt_boxes_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        pred_boxes_by_frame: Dict[int, Dict[int, np.ndarray]] = {}

        for f in frames:
            f_gt = gt_df[gt_df["frame"] == f]
            f_pred = pred_df[pred_df["frame"] == f]

            gt_ids = f_gt["id"].values
            pred_ids = f_pred["id"].values

            gt_boxes = f_gt[["bb_left", "bb_top", "bb_width", "bb_height"]].values.astype(np.float32)
            pred_boxes = f_pred[["bb_left", "bb_top", "bb_width", "bb_height"]].values.astype(np.float32)

            # Convert tlwh to xyxy
            gt_xyxy = np.copy(gt_boxes)
            gt_xyxy[:, 2] += gt_xyxy[:, 0]
            gt_xyxy[:, 3] += gt_xyxy[:, 1]

            pred_xyxy = np.copy(pred_boxes)
            if len(pred_xyxy) > 0:
                pred_xyxy[:, 2] += pred_xyxy[:, 0]
                pred_xyxy[:, 3] += pred_xyxy[:, 1]

            gt_boxes_by_frame[f] = {gid: box for gid, box in zip(gt_ids, gt_xyxy)}
            pred_boxes_by_frame[f] = {pid: box for pid, box in zip(pred_ids, pred_xyxy)}

            mapping = {}
            if len(gt_xyxy) > 0 and len(pred_xyxy) > 0:
                iou_mat = compute_iou_matrix(gt_xyxy, pred_xyxy)
                for g_idx, gid in enumerate(gt_ids):
                    best_p_idx = int(np.argmax(iou_mat[g_idx]))
                    best_iou = float(iou_mat[g_idx, best_p_idx])
                    if best_iou >= self.iou_match_thresh:
                        mapping[gid] = pred_ids[best_p_idx]
            gt_pred_mapping[f] = mapping

        # Trace trajectory per GT ID and detect ID switch incidents
        all_gt_ids = sorted(list(gt_df["id"].unique()))
        incidents: List[Dict[str, Any]] = []

        for gid in all_gt_ids:
            history: List[Tuple[int, Optional[int]]] = []
            for f in frames:
                mapped_pid = gt_pred_mapping[f].get(gid, None)
                history.append((f, mapped_pid))

            last_valid_pid = None
            last_valid_frame = None

            for f, cur_pid in history:
                if cur_pid is None:
                    continue

                if last_valid_pid is not None and cur_pid != last_valid_pid:
                    # ID switch detected!
                    switch_frame = f
                    gap_frames = switch_frame - last_valid_frame - 1

                    # Diagnose primary cause
                    cause, details = self._diagnose_cause(
                        gid=gid,
                        switch_frame=switch_frame,
                        last_frame=last_valid_frame,
                        gap_frames=gap_frames,
                        gt_boxes_by_frame=gt_boxes_by_frame,
                        frames=frames
                    )

                    incidents.append({
                        "gt_id": int(gid),
                        "frame": int(switch_frame),
                        "old_pred_id": int(last_valid_pid),
                        "new_pred_id": int(cur_pid),
                        "gap_frames": int(gap_frames),
                        "primary_cause": cause,
                        "details": details
                    })

                last_valid_pid = cur_pid
                last_valid_frame = f

        # Summary statistics
        cause_counts = {
            "Occlusion / Crossing": 0,
            "Missed Detection Gap": 0,
            "Fast Motion / Velocity": 0,
            "Similar Appearance / Ambiguity": 0
        }
        for inc in incidents:
            cause = inc["primary_cause"]
            if cause in cause_counts:
                cause_counts[cause] += 1
            else:
                cause_counts[cause] = 1

        total_switches = len(incidents)
        breakdown = []
        for cause, count in cause_counts.items():
            pct = (count / total_switches * 100.0) if total_switches > 0 else 0.0
            breakdown.append({
                "Cause": cause,
                "Count": count,
                "Percentage": f"{pct:.1f}%"
            })
        breakdown_df = pd.DataFrame(breakdown)

        report_md = self._format_markdown_report(incidents, breakdown_df, total_switches)

        if output_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
            with open(output_report_path, "w") as f:
                f.write(report_md)

        return {
            "total_switches": total_switches,
            "cause_breakdown": breakdown_df,
            "incidents": incidents,
            "markdown_report": report_md
        }

    def _diagnose_cause(
        self,
        gid: int,
        switch_frame: int,
        last_frame: int,
        gap_frames: int,
        gt_boxes_by_frame: Dict[int, Dict[int, np.ndarray]],
        frames: List[int]
    ) -> Tuple[str, str]:
        """Classify the root cause based on spatial overlap, velocity, and detection gap."""
        # 1. Check for Occlusion / Crossing in the window [last_frame - 1, switch_frame]
        window_frames = [f for f in frames if last_frame - 2 <= f <= switch_frame]
        max_overlap_with_other = 0.0
        overlapping_other_id = None

        for wf in window_frames:
            if wf not in gt_boxes_by_frame or gid not in gt_boxes_by_frame[wf]:
                continue
            box_g = gt_boxes_by_frame[wf][gid]
            for other_gid, box_o in gt_boxes_by_frame[wf].items():
                if other_gid == gid:
                    continue
                # Compute IoU and Inter-over-min
                iou = float(compute_iou_matrix(box_g[None, :], box_o[None, :])[0, 0])
                inter_min = float(compute_inter_min_matrix(box_g[None, :], box_o[None, :])[0, 0])
                score = max(iou, inter_min)
                if score > max_overlap_with_other:
                    max_overlap_with_other = score
                    overlapping_other_id = other_gid

        if max_overlap_with_other >= self.occlusion_iou_thresh:
            return (
                "Occlusion / Crossing",
                f"Overlap {max_overlap_with_other:.2f} with GT ID {overlapping_other_id} during transition"
            )

        # 2. Check for Missed Detection Gap
        if gap_frames >= 2:
            return (
                "Missed Detection Gap",
                f"Detection missing for {gap_frames} consecutive frames prior to switch"
            )

        # 3. Check for Fast Motion / Maneuver
        if (switch_frame in gt_boxes_by_frame and gid in gt_boxes_by_frame[switch_frame] and
            last_frame in gt_boxes_by_frame and gid in gt_boxes_by_frame[last_frame]):
            b1 = gt_boxes_by_frame[last_frame][gid]
            b2 = gt_boxes_by_frame[switch_frame][gid]
            c1 = np.array([(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])
            c2 = np.array([(b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0])
            dt = max(1, switch_frame - last_frame)
            velocity = float(np.linalg.norm(c2 - c1) / dt)

            if velocity >= self.fast_motion_velocity_thresh:
                return (
                    "Fast Motion / Velocity",
                    f"Target moved at {velocity:.1f} px/frame (exceeding {self.fast_motion_velocity_thresh} px/frame)"
                )

        # 4. Similar Appearance / Ambiguous Matching (occurs when no occlusion or high motion detected)
        return (
            "Similar Appearance / Ambiguity",
            "Continuous tracking displaced to nearby trajectory without heavy occlusion or speed jump"
        )

    def _format_markdown_report(
        self,
        incidents: List[Dict[str, Any]],
        breakdown_df: pd.DataFrame,
        total_switches: int
    ) -> str:
        lines = []
        lines.append("# Automated Tracking Failure & ID Switch Root Cause Analysis")
        lines.append(f"\n**Total Identified ID Switches:** {total_switches}\n")
        lines.append("## Root Cause Breakdown\n")
        lines.append(breakdown_df.to_markdown(index=False))
        lines.append("\n## Incident Log (Sample / Significant Events)\n")

        if len(incidents) == 0:
            lines.append("No ID switches detected! Tracker maintained 100% persistent identity continuity.")
        else:
            lines.append("| Frame | GT ID | Old Pred ID | New Pred ID | Primary Cause | Context / Diagnostics |")
            lines.append("|---|---|---|---|---|---|")
            for inc in incidents[:30]:
                lines.append(
                    f"| {inc['frame']} | {inc['gt_id']} | {inc['old_pred_id']} | "
                    f"{inc['new_pred_id']} | **{inc['primary_cause']}** | {inc['details']} |"
                )
            if len(incidents) > 30:
                lines.append(f"\n*(Showing 30 of {len(incidents)} incidents)*")

        lines.append("\n## Tracker Recommendations:")
        lines.append("- For **Occlusion / Crossing**: Increase `max_lost_age` and enable occlusion feature freeze to protect appearance memory.")
        lines.append("- For **Missed Detection Gap**: Lower detector confidence threshold or use Kalman coasting propagation.")
        lines.append("- For **Fast Motion**: Increase Kalman process noise velocity covariance and relax Mahalanobis gate threshold.")
        lines.append("- For **Similar Appearance**: Increase Re-ID gallery size and adjust appearance weighting $\\alpha$.\n")

        return "\n".join(lines)
