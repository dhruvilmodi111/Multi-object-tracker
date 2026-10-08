"""Standalone Visualization Script to generate annotated tracking video from MOT results.
"""

from typing import Optional, List, Dict, Any
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils import load_seqinfo, get_sequence_frame_paths
from src.utils.visualization import TrackVisualizer, VideoExporter
from src.tracker.track import Track, TrackState
from src.utils.box_utils import xyxy_to_z


def visualize_results_on_sequence(
    seq_dir: str,
    results_path: str,
    output_video_path: str,
    fps_override: Optional[float] = None
) -> str:
    """Render bounding boxes, IDs, trajectories, and HUD from results.txt onto sequence frames."""
    seq_info = load_seqinfo(seq_dir)
    frame_paths = get_sequence_frame_paths(seq_dir)
    fps = fps_override or seq_info.get("frameRate", 30.0)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Load results: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    df_res = pd.read_csv(
        results_path,
        header=None,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    )

    first_frame = cv2.imread(frame_paths[0])
    h, w = first_frame.shape[:2]

    visualizer = TrackVisualizer(show_trajectories=True, show_hud=True)
    exporter = VideoExporter(output_video_path, fps=fps, frame_size=(w, h))

    print(f"Generating visualization video: {output_video_path}...")

    # Group predictions by frame
    grouped = {int(f): g for f, g in df_res.groupby("frame")}

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Rendering"), start=1):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        mock_tracks = []
        if frame_idx in grouped:
            rows = grouped[frame_idx]
            for _, row in rows.iterrows():
                tid = int(row["id"])
                x1 = float(row["bb_left"])
                y1 = float(row["bb_top"])
                x2 = x1 + float(row["bb_width"])
                y2 = y1 + float(row["bb_height"])
                xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

                # Construct lightweight mock track object for visualization
                mean = np.zeros(8, dtype=np.float32)
                mean[:4] = xyxy_to_z(xyxy)
                t = Track(mean=mean, covariance=np.eye(8, dtype=np.float32), track_id=tid)
                t.state = TrackState.ACTIVE
                mock_tracks.append(t)

        vis_frame = visualizer.draw_tracks(
            frame=frame,
            tracks=mock_tracks,
            frame_idx=frame_idx,
            fps=fps,
            additional_info={"tracker_name": Path(results_path).stem}
        )
        exporter.write_frame(vis_frame)

    exporter.release()
    print(f"Visualization video saved successfully: {output_video_path}")
    return output_video_path


def main():
    parser = argparse.ArgumentParser(description="Render tracking visualization video.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to sequence directory.")
    parser.add_argument("--results", type=str, required=True, help="Path to predictions results.txt.")
    parser.add_argument("--output_video", type=str, default="visualizations/tracking_demo.mp4", help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=None, help="Frame rate.")
    args = parser.parse_args()

    visualize_results_on_sequence(
        seq_dir=args.sequence,
        results_path=args.results,
        output_video_path=args.output_video,
        fps_override=args.fps
    )


if __name__ == "__main__":
    main()
