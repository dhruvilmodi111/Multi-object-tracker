"""Dataset Preparation Script for MOT17 / MOT20 Benchmarking.
Supports:
1. Generating a standardized Occlusion Benchmark Sequence ('MOT17-SYNTH-OCCLUSION')
   with ground truth modeling:
   - Severe partial and full visual occlusions
   - Pedestrians crossing paths
   - Disappearance and re-appearance after 15-25 frames
   - Fast motion accelerations
2. Downloading sample MOT17 sequences (if internet access is active)
3. Ingesting user-provided MOT17 / MOT20 sequences
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_synthetic_occlusion_sequence(output_dir: str, num_frames: int = 150) -> str:
    """Generate a realistic test sequence with ground truth strictly following MOT17 format."""
    seq_dir = Path(output_dir) / "MOT17-SYNTH-OCCLUSION"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    w, h = 1280, 720
    fps = 30

    # Write seqinfo.ini
    seqinfo_content = f"""[Sequence]
name=MOT17-SYNTH-OCCLUSION
imDir=img1
frameRate={fps}
seqLength={num_frames}
imWidth={w}
imHeight={h}
imExt=.jpg
"""
    with open(seq_dir / "seqinfo.ini", "w") as f:
        f.write(seqinfo_content)

    # Define pedestrian agents with distinct appearances and trajectories
    # Target 1: Walks left-to-right (Red shirt, dark pants)
    # Target 2: Walks right-to-left, crosses in front of Target 1 around frame 50-70 (Blue jacket)
    # Target 3: Walks diagonal, gets completely occluded by a pillar/target 2 (Green sweater)
    # Target 4: Fast moving pedestrian running across from frame 40 to 110 (Yellow hoodie)
    # Target 5: Static / slow pedestrian chatting on the side (Purple coat)

    pedestrians = [
        {
            "id": 1,
            "color_top": (40, 40, 220),     # Red
            "color_bot": (50, 50, 50),      # Dark grey
            "start_pos": np.array([150.0, 360.0]),
            "vel": np.array([5.5, 0.4]),
            "w": 75, "h": 220,
            "start_frame": 1,
            "end_frame": 150
        },
        {
            "id": 2,
            "color_top": (220, 100, 30),    # Blue
            "color_bot": (30, 30, 30),
            "start_pos": np.array([1050.0, 340.0]),
            "vel": np.array([-5.8, 0.6]),
            "w": 85, "h": 235,  # Closer to camera, occludes Target 1
            "start_frame": 1,
            "end_frame": 150
        },
        {
            "id": 3,
            "color_top": (50, 180, 50),     # Green
            "color_bot": (80, 80, 80),
            "start_pos": np.array([480.0, 300.0]),
            "vel": np.array([2.5, -0.3]),
            "w": 65, "h": 190,
            "start_frame": 1,
            "end_frame": 150,
            # Temporarily occluded / missing between frames 65 and 90!
            "occlusion_window": (65, 88)
        },
        {
            "id": 4,
            "color_top": (20, 220, 220),    # Yellow
            "color_bot": (40, 40, 40),
            "start_pos": np.array([100.0, 420.0]),
            "vel": np.array([14.0, 0.2]),   # Fast runner!
            "w": 80, "h": 210,
            "start_frame": 35,
            "end_frame": 115
        },
        {
            "id": 5,
            "color_top": (180, 50, 160),    # Magenta
            "color_bot": (30, 30, 70),
            "start_pos": np.array([820.0, 440.0]),
            "vel": np.array([0.4, 0.1]),    # Slow walker
            "w": 80, "h": 225,
            "start_frame": 1,
            "end_frame": 150
        }
    ]

    gt_records = []

    print(f"Generating synthetic MOT benchmark sequence: {seq_dir} ({num_frames} frames)...")

    # Stationary environment background (street walkway)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:int(h * 0.45)] = (180, 170, 160)  # Distant buildings / sky
    bg[int(h * 0.45):] = (120, 120, 115)  # Sidewalk pavement
    # Add perspective lines on pavement
    for lx in range(0, w, 160):
        cv2.line(bg, (lx, int(h * 0.45)), (lx - 200, h), (105, 105, 100), 2)
    # Add a fixed decorative kiosk / pillar at x=580..660 that causes occlusion
    cv2.rectangle(bg, (580, 220), (660, 560), (70, 75, 80), -1)
    cv2.rectangle(bg, (575, 215), (665, 230), (50, 55, 60), -1)

    for frame_idx in range(1, num_frames + 1):
        frame = bg.copy()

        # Sort pedestrians by bottom-y coordinate (z-ordering / depth so closer people render in front)
        active_peds = []
        for ped in pedestrians:
            if ped["start_frame"] <= frame_idx <= ped["end_frame"]:
                elapsed = frame_idx - ped["start_frame"]
                pos = ped["start_pos"] + ped["vel"] * elapsed
                active_peds.append((pos[1] + ped["h"], ped, pos))

        active_peds.sort(key=lambda item: item[0])

        for _, ped, pos in active_peds:
            pid = ped["id"]
            pw = ped["w"]
            ph = ped["h"]
            px = int(round(pos[0]))
            py = int(round(pos[1]))

            # Check if this pedestrian is completely occluded in their occlusion window
            is_occluded_window = False
            if "occlusion_window" in ped:
                occ_s, occ_e = ped["occlusion_window"]
                if occ_s <= frame_idx <= occ_e:
                    is_occluded_window = True

            # If not hidden behind the pillar in this window, render person
            if not is_occluded_window:
                # Render realistic pedestrian figure (head, torso with top color, legs with bottom color)
                # Head
                head_rad = int(round(pw * 0.28))
                head_cx = px + pw // 2
                head_cy = py + head_rad + 6
                cv2.circle(frame, (head_cx, head_cy), head_rad, (190, 210, 235), -1)  # Face tone

                # Torso (color_top)
                torso_y1 = head_cy + head_rad
                torso_y2 = py + int(round(ph * 0.58))
                cv2.rectangle(frame, (px + int(pw * 0.15), torso_y1), (px + int(pw * 0.85), torso_y2), ped["color_top"], -1)

                # Arms
                cv2.rectangle(frame, (px + int(pw * 0.05), torso_y1 + 4), (px + int(pw * 0.18), torso_y2 - 6), ped["color_top"], -1)
                cv2.rectangle(frame, (px + int(pw * 0.82), torso_y1 + 4), (px + int(pw * 0.95), torso_y2 - 6), ped["color_top"], -1)

                # Legs / Pants (color_bot)
                legs_y1 = torso_y2
                legs_y2 = py + ph
                leg_w = int(pw * 0.3)
                leg_gap = int(pw * 0.1)
                cv2.rectangle(frame, (px + int(pw * 0.15), legs_y1), (px + int(pw * 0.15) + leg_w, legs_y2), ped["color_bot"], -1)
                cv2.rectangle(frame, (px + int(pw * 0.85) - leg_w, legs_y1), (px + int(pw * 0.85), legs_y2), ped["color_bot"], -1)

            # Record MOT Ground Truth
            # MOT Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
            # If in occlusion window, visibility drops to 0.0, else 1.0 (or partial if crossing)
            vis = 0.0 if is_occluded_window else 1.0
            gt_records.append(
                f"{frame_idx},{pid},{px:.2f},{py:.2f},{pw:.2f},{ph:.2f},1,1,{vis:.2f}\n"
            )

        # Re-render stationary pillar in front if target 3 is behind it
        cv2.rectangle(frame, (580, 220), (660, 560), (70, 75, 80), -1)
        cv2.rectangle(frame, (575, 215), (665, 230), (50, 55, 60), -1)

        # Save frame image
        frame_name = f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(img_dir / frame_name), frame)

    # Write gt.txt
    with open(gt_dir / "gt.txt", "w") as f:
        f.writelines(gt_records)

    print(f"Dataset preparation complete: {seq_dir}")
    print(f"Total frames: {num_frames}, Ground truth entries: {len(gt_records)}")
    return str(seq_dir)


def extract_video_to_sequence(
    video_path: str,
    output_dir: str,
    seq_name: str = "MOT17-VTEST-REAL",
    max_frames: int = 150
) -> str:
    """Extract frames from a video file into a standard MOT sequence folder."""
    seq_dir = Path(output_dir) / seq_name
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 1
    print(f"Extracting video frames to {img_dir}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(img_dir / frame_name), frame)
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

    cap.release()
    total_frames = frame_idx - 1

    seqinfo_content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate={fps}
seqLength={total_frames}
imWidth={w}
imHeight={h}
imExt=.jpg
"""
    with open(seq_dir / "seqinfo.ini", "w") as f:
        f.write(seqinfo_content)

    print(f"Extracted {total_frames} frames to {seq_dir}")
    return str(seq_dir)


def download_sample_real_video(dest_path: str = "data/vtest.avi") -> str:
    """Download standard surveillance pedestrian video."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if not os.path.exists(dest_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
        print(f"Downloading real pedestrian surveillance video from: {url}")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Prepare MOT17 / MOT20 dataset or benchmark sequences.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where sequences are stored.")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames for synthetic sequence.")
    parser.add_argument("--download_real", action="store_true", help="Download and extract real pedestrian surveillance video.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_synthetic_occlusion_sequence(args.output_dir, num_frames=args.frames)

    if args.download_real:
        v_path = download_sample_real_video(os.path.join(args.output_dir, "vtest.avi"))
        extract_video_to_sequence(v_path, args.output_dir, seq_name="MOT17-VTEST-REAL", max_frames=args.frames)


if __name__ == "__main__":
    main()
