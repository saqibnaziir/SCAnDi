"""
Sperm_V2 Dataset — Exploratory Data Analysis
============================================
Analyzes the Sperm_V2 dataset structure, image specs, and statistics.
Run before segmentation to understand the data.

Usage:
  python eda_sperm_v2.py --data /path/to/Sperm_V2
"""

import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

IMG_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def is_tile_file(path: Path) -> bool:
    """True if path is a tile image (excludes whole-slide, LSM). Tiles: _mNNN in filename."""
    if path.suffix.lower() not in IMG_EXTENSIONS:
        return False
    name = path.name
    if "Tiled" in name or " Tiled" in name:
        return False
    return bool(re.search(r"_m\d+", name))


def get_tile_files(folder: Path) -> list:
    return sorted([f for f in folder.iterdir() if f.is_file() and is_tile_file(f)],
                  key=lambda p: p.name)


def pinkness(img_bgr: np.ndarray) -> np.ndarray:
    r, g, b = img_bgr[:, :, 2].astype(np.float32), img_bgr[:, :, 1].astype(np.float32), img_bgr[:, :, 0].astype(np.float32)
    return r - 0.5 * (g + b)


def run_eda(data_root: Path, output_dir: Path | None = None, sample_per_folder: int = 5):
    data_root = Path(data_root)
    if not data_root.exists():
        log.error(f"Data root not found: {data_root}")
        return
    folders = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not folders:
        log.error(f"No subfolders in {data_root}")
        return
    log.info(f"Sperm_V2 EDA — {data_root}")
    report = {"data_root": str(data_root), "folders": {}, "totals": {"folders": 0, "tile_images": 0}}
    all_shapes = set()
    all_pinkness = []
    for folder in folders:
        tile_files = get_tile_files(folder)
        folder_report = {"tile_images": len(tile_files), "shapes": {}, "pinkness_mean": []}
        if not tile_files:
            report["folders"][folder.name] = folder_report
            continue
        shapes_count = {}
        pink_list = []
        for path in tqdm(tile_files, desc=folder.name, leave=False):
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            shapes_count[(h, w)] = shapes_count.get((h, w), 0) + 1
            all_shapes.add((h, w))
            p = pinkness(img)[::8, ::8]
            pm = float(np.nanmean(p))
            pink_list.append(pm)
            all_pinkness.append(pm)
        folder_report["shapes"] = {str(k): v for k, v in shapes_count.items()}
        if pink_list:
            folder_report["pinkness_mean"] = round(float(np.mean(pink_list)), 2)
        report["folders"][folder.name] = folder_report
        report["totals"]["tile_images"] += len(tile_files)
        report["totals"]["folders"] += 1
    report["totals"]["unique_shapes"] = [list(s) for s in all_shapes]
    if all_pinkness:
        report["totals"]["pinkness_overall"] = round(float(np.mean(all_pinkness)), 2)
    out_dir = Path(output_dir) if output_dir else data_root.parent / "outputs" / "eda_sperm_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sperm_v2_eda_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {out_dir / 'sperm_v2_eda_report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sperm_V2 dataset EDA")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to Sperm_V2 root (Image-01, Image-02, ...)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()
    run_eda(args.data, args.output, args.sample)
