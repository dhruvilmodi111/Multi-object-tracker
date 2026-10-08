"""Visualizer for Multi-Object Tracking with Trajectory Trails, Track States, and Occlusion Indicators.
"""

from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import os
import cv2
import numpy as np

if TYPE_CHECKING:
    from src.tracker.track import Track


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate consistent, vibrant BGR color for a given track ID."""
    # Use golden ratio HSV distribution for distinct, aesthetically pleasant colors
    golden_ratio_conjugate = 0.618033988749895
    h = ((track_id * golden_ratio_conjugate) % 1.0) * 180.0
    s = 200.0 + (track_id * 17) % 55
    v = 220.0 + (track_id * 23) % 35
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


class TrackVisualizer:
    """Renders tracking boxes, trajectories, IDs, states, and HUD metrics onto frames."""

    def __init__(
        self,
        line_thickness: int = 2,
        font_scale: float = 0.55,
        tail_length: int = 30,
        show_trajectories: bool = True,
        show_hud: bool = True
    ):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.tail_length = tail_length
        self.show_trajectories = show_trajectories
        self.show_hud = show_hud
        # Keep global trajectory memory across frames: id -> list of (cx, cy)
        self.trajectory_history: Dict[int, List[Tuple[int, int]]] = {}

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Any],
        frame_idx: int = 0,
        fps: float = 30.0,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Render tracking annotations on a copy of the frame."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Draw trajectory tails first (so boxes render cleanly on top)
        if self.show_trajectories:
            for t in tracks:
                tid = t.track_id
                box = t.to_xyxy()
                cx = int(round((box[0] + box[2]) / 2.0))
                cy = int(round(box[3]))  # Ground/foot contact point

                if tid not in self.trajectory_history:
                    self.trajectory_history[tid] = []
                self.trajectory_history[tid].append((cx, cy))
                if len(self.trajectory_history[tid]) > self.tail_length:
                    self.trajectory_history[tid].pop(0)

                points = self.trajectory_history[tid]
                color = get_color_for_id(tid)

                for i in range(1, len(points)):
                    alpha = float(i) / float(len(points))
                    pt1 = points[i - 1]
                    pt2 = points[i]
                    thickness = max(1, int(round(self.line_thickness * alpha)))
                    cv2.line(vis_frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw bounding boxes and status labels
        active_count = 0
        occluded_count = 0

        for t in tracks:
            tid = t.track_id
            box = t.to_xyxy()
            x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
            x2, y2 = min(w - 1, int(round(box[2]))), min(h - 1, int(round(box[3])))

            color = get_color_for_id(tid)
            is_occluded = getattr(t, "is_occluded", False)
            state_str = "ACTIVE" if t.is_active else ("LOST" if t.is_lost else "TENTATIVE")

            if t.is_active:
                active_count += 1
            if is_occluded:
                occluded_count += 1

            # Box styling: double line or alert color if occluded
            if is_occluded:
                # Orange/Red highlight border
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 140, 255), self.line_thickness + 1, cv2.LINE_AA)
            else:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.line_thickness, cv2.LINE_AA)

            # Label text
            tag = f"ID:{tid}"
            if is_occluded:
                tag += " [OCC]"
            if getattr(t, "reid_recovered_count", 0) > 0:
                tag += f" [ReID:{t.reid_recovered_count}]"

            (text_w, text_h), baseline = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            # Label background header
            lbl_y1 = max(0, y1 - text_h - 6)
            lbl_y2 = y1
            lbl_x2 = min(w, x1 + text_w + 8)

            bg_color = (0, 100, 220) if is_occluded else color
            cv2.rectangle(vis_frame, (x1, lbl_y1), (lbl_x2, lbl_y2), bg_color, -1)
            cv2.putText(
                vis_frame,
                tag,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw HUD dashboard in top-left corner
        if self.show_hud:
            hud_bg_w, hud_bg_h = 320, 85
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
            cv2.rectangle(vis_frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (100, 100, 100), 1)

            cv2.putText(
                vis_frame,
                f"Frame: {frame_idx:04d} | Tracks: {len(tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            occ_text = f"Occlusions: {occluded_count}"
            occ_color = (0, 165, 255) if occluded_count > 0 else (180, 180, 180)
            cv2.putText(
                vis_frame,
                occ_text,
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                occ_color,
                1,
                cv2.LINE_AA
            )
            sub_info = "Status: Re-ID & Occlusion Buffer Active"
            if additional_info and "tracker_name" in additional_info:
                sub_info = f"Tracker: {additional_info['tracker_name']}"
            cv2.putText(
                vis_frame,
                sub_info,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 120),
                1,
                cv2.LINE_AA
            )

        return vis_frame


class VideoExporter:
    """Exports processed tracking frames into an MP4 video."""

    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Optional[Tuple[int, int]] = None):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer: Optional[cv2.VideoWriter] = None
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def write_frame(self, frame: np.ndarray) -> None:
        if self.writer is None:
            h, w = frame.shape[:2]
            if self.frame_size is None:
                self.frame_size = (w, h)
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
