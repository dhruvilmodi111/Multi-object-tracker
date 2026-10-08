"""Run Multi-Object Tracking on a MOT sequence or video file.
"""

from typing import Dict, Any, List, Optional
import argparse
import os
import sys
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.yolo_detector import YOLOPersonDetector, Detection
from src.reid.feature_extractor import ReIDFeatureExtractor
from src.tracker.occlusion_tracker import OcclusionTracker
from src.utils.io_utils import (
    load_seqinfo,
    get_sequence_frame_paths,
    write_mot_results,
    load_mot_ground_truth
)
from src.utils.visualization import TrackVisualizer, VideoExporter


def run_tracker(
    seq_dir: str,
    config_path: str,
    output_results_path: str,
    save_video_path: Optional[str] = None,
    use_detections_file: Optional[str] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """Execute tracking pipeline on sequence and output results."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tracker_name = cfg.get("tracker_name", "OcclusionTracker")
    det_cfg = cfg.get("detector", {})
    reid_cfg = cfg.get("reid", {})
    trk_cfg = cfg.get("tracker", {})

    print(f"\n==========================================")
    print(f"Running Tracker: {tracker_name}")
    print(f"Sequence: {seq_dir}")
    print(f"Configuration: {config_path}")
    print(f"==========================================")

    # Load sequence frames & metadata
    seq_info = load_seqinfo(seq_dir)
    frame_paths = get_sequence_frame_paths(seq_dir)

    if not frame_paths:
        raise FileNotFoundError(f"No valid image frames found in {seq_dir}")

    fps = seq_info.get("frameRate", 30.0)

    # Initialize Re-ID feature extractor
    reid_extractor = ReIDFeatureExtractor(
        model_name=reid_cfg.get("model_name", "mobilenet_v3_small"),
        embedding_dim=reid_cfg.get("embedding_dim", 512),
        device=device
    )

    # Initialize detector if not using precomputed / GT detections
    detector = None
    gt_dets_by_frame = {}
    if use_detections_file:
        print(f"Using external/ground-truth detections from: {use_detections_file}")
        df_dets = load_mot_ground_truth(use_detections_file)
        # Filter for visible objects (visibility > 0)
        if "visibility" in df_dets.columns:
            df_dets = df_dets[df_dets["visibility"] > 0.0]
        for f_idx, group in df_dets.groupby("frame"):
            gt_dets_by_frame[int(f_idx)] = group[["bb_left", "bb_top", "bb_width", "bb_height"]].values
    else:
        print(f"Initializing YOLO Person Detector: {det_cfg.get('model_name', 'yolov8n.pt')}")
        detector = YOLOPersonDetector(
            model_name=det_cfg.get("model_name", "yolov8n.pt"),
            conf_thresh=det_cfg.get("conf_thresh", 0.35),
            iou_thresh=det_cfg.get("iou_thresh", 0.50),
            device=device
        )

    # Initialize Occlusion Tracker
    tracker = OcclusionTracker(
        max_lost_age=trk_cfg.get("max_lost_age", 30),
        n_init=trk_cfg.get("n_init", 3),
        appearance_weight=trk_cfg.get("appearance_weight", 0.70),
        stage1_match_thresh=trk_cfg.get("stage1_match_thresh", 0.45),
        stage2_iou_thresh=trk_cfg.get("stage2_iou_thresh", 0.50),
        reid_lost_thresh=trk_cfg.get("reid_lost_thresh", 0.38),
        occlusion_iou_thresh=trk_cfg.get("occlusion_iou_thresh", 0.20),
        gallery_size=trk_cfg.get("gallery_size", 10),
        feature_ema_alpha=trk_cfg.get("feature_ema_alpha", 0.90),
        enable_lost_reid=trk_cfg.get("enable_lost_reid", True),
        enable_occlusion_freeze=trk_cfg.get("enable_occlusion_freeze", True)
    )

    visualizer = None
    video_exporter = None
    if save_video_path:
        first_frame = cv2.imread(frame_paths[0])
        h, w = first_frame.shape[:2]
        visualizer = TrackVisualizer(show_trajectories=True, show_hud=True)
        video_exporter = VideoExporter(save_video_path, fps=fps, frame_size=(w, h))

    recorded_tracks = []
    total_time = 0.0

    print(f"Processing {len(frame_paths)} frames...")

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=tracker_name), start=1):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        t0 = time.perf_counter()

        # Step 1: Detect persons
        detections: List[Detection] = []
        if use_detections_file:
            if frame_idx in gt_dets_by_frame:
                boxes_tlwh = gt_dets_by_frame[frame_idx]
                for b in boxes_tlwh:
                    xyxy = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32)
                    detections.append(Detection(bbox=xyxy, conf=1.0, class_id=0))
        else:
            detections = detector.detect(frame)

        # Step 2: Extract Re-ID Appearance Embeddings
        if len(detections) > 0:
            boxes = np.array([d.bbox for d in detections], dtype=np.float32)
            features = reid_extractor.extract_from_boxes(frame, boxes)
            for d, feat in zip(detections, features):
                d.feature = feat

        # Step 3: Update Tracker
        active_tracks = tracker.step(detections)

        total_time += (time.perf_counter() - t0)

        # Step 4: Record output in MOT format
        for track in active_tracks:
            tlwh = track.to_tlwh()
            recorded_tracks.append({
                "frame": frame_idx,
                "track_id": track.track_id,
                "bb_left": tlwh[0],
                "bb_top": tlwh[1],
                "bb_width": tlwh[2],
                "bb_height": tlwh[3],
                "conf": 1.0
            })

        # Step 5: Visualize and write to video
        if video_exporter and visualizer:
            vis_frame = visualizer.draw_tracks(
                frame=frame,
                tracks=active_tracks,
                frame_idx=frame_idx,
                fps=fps,
                additional_info={"tracker_name": tracker_name}
            )
            video_exporter.write_frame(vis_frame)

    if video_exporter:
        video_exporter.release()
        print(f"Visualization video successfully saved to: {save_video_path}")

    # Write MOT results file
    write_mot_results(output_results_path, recorded_tracks)
    print(f"Tracking results saved to: {output_results_path}")

    fps_achieved = len(frame_paths) / max(1e-4, total_time)
    print(f"Tracking finished. Processed {len(frame_paths)} frames in {total_time:.2f}s ({fps_achieved:.1f} FPS)")

    return {
        "tracker_name": tracker_name,
        "results_path": output_results_path,
        "video_path": save_video_path,
        "num_frames": len(frame_paths),
        "total_time": total_time,
        "fps": fps_achieved
    }


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Object Tracking.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to sequence directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--output", type=str, default="results/results.txt", help="Path to output results file.")
    parser.add_argument("--save_video", type=str, default=None, help="Optional output MP4 video path.")
    parser.add_argument("--use_detections", type=str, default=None, help="Optional precomputed/GT detections file.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto').")
    args = parser.parse_args()

    run_tracker(
        seq_dir=args.sequence,
        config_path=args.config,
        output_results_path=args.output,
        save_video_path=args.save_video,
        use_detections_file=args.use_detections,
        device=args.device
    )


if __name__ == "__main__":
    main()
