"""Input/Output utilities for MOT benchmark formats, video loading, and text files.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import configparser
import os
import glob
import cv2
import numpy as np
import pandas as pd


def load_seqinfo(seq_dir: str) -> Dict[str, Any]:
    """Parse seqinfo.ini inside a MOT sequence directory."""
    ini_path = Path(seq_dir) / "seqinfo.ini"
    info = {
        "name": Path(seq_dir).name,
        "imDir": "img1",
        "frameRate": 30,
        "seqLength": 0,
        "imWidth": 1920,
        "imHeight": 1080,
        "imExt": ".jpg"
    }
    if ini_path.exists():
        cp = configparser.ConfigParser()
        cp.read(str(ini_path))
        if "Sequence" in cp:
            sec = cp["Sequence"]
            info["name"] = sec.get("name", info["name"])
            info["imDir"] = sec.get("imDir", info["imDir"])
            info["frameRate"] = float(sec.get("frameRate", info["frameRate"]))
            info["seqLength"] = int(sec.get("seqLength", info["seqLength"]))
            info["imWidth"] = int(sec.get("imWidth", info["imWidth"]))
            info["imHeight"] = int(sec.get("imHeight", info["imHeight"]))
            info["imExt"] = sec.get("imExt", info["imExt"])
    return info


def get_sequence_frame_paths(seq_dir: str) -> List[str]:
    """Retrieve sorted list of frame file paths from a sequence directory."""
    seq_path = Path(seq_dir)
    img_dir = seq_path / "img1"
    if not img_dir.exists():
        img_dir = seq_path

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(str(img_dir / ext)))
        files.extend(glob.glob(str(img_dir / ext.upper())))

    # Natural sorting by filename / integer if possible
    def sort_key(p: str) -> Tuple[int, str]:
        stem = Path(p).stem
        digits = "".join([c for c in stem if c.isdigit()])
        if digits:
            return (int(digits), stem)
        return (0, stem)

    files = sorted(list(set(files)), key=sort_key)
    return files


def load_mot_ground_truth(gt_file: str) -> pd.DataFrame:
    """Load standard MOT ground truth CSV (comma or space separated).
    Columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    # Read with flexible separator
    try:
        df = pd.read_csv(
            gt_file,
            header=None,
            sep=r"[\s,]+",
            engine="python",
            names=[
                "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                "conf", "class", "visibility"
            ],
            dtype={
                "frame": int,
                "id": int,
                "bb_left": float,
                "bb_top": float,
                "bb_width": float,
                "bb_height": float,
                "conf": float,
                "class": int,
                "visibility": float
            }
        )
    except Exception:
        # Fallback to standard comma read
        df = pd.read_csv(
            gt_file,
            header=None,
            names=[
                "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                "conf", "class", "visibility"
            ]
        )
    return df


def write_mot_results(results_path: str, tracks: List[Dict[str, Any]]) -> None:
    """Write tracker predictions to MOT format file.
    Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1
    """
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "w") as f:
        for t in sorted(tracks, key=lambda x: (x["frame"], x["track_id"])):
            f.write(
                f"{int(t['frame'])},{int(t['track_id'])},"
                f"{t['bb_left']:.2f},{t['bb_top']:.2f},"
                f"{t['bb_width']:.2f},{t['bb_height']:.2f},"
                f"{t.get('conf', 1.0):.4f},-1,-1,-1\n"
            )
