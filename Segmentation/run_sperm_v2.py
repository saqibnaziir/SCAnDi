"""
Run segmentation on Sperm_V2 dataset.
=====================================
Processes only tile images (filenames with _mNNN). Skips whole-slide overviews and .lsm files.

Usage:
  python run_sperm_v2.py --data /path/to/Sperm_V2 --output /path/to/output
  python run_sperm_v2.py --data /path/to/Sperm_V2 --output /path/to/output --use-cellpose
"""

import argparse
from pathlib import Path

from maps import run_batch, SPERM_V2_FOLDER_CELL_TYPE, _is_sperm_v2_tile


def main():
    parser = argparse.ArgumentParser(description="Sperm_V2 segmentation — tiles only")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to Sperm_V2 root (Image-01, Image-02, ...)")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process only this folder (e.g. Image-01)")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-overlap", type=int, default=64)
    parser.add_argument("--save-patches", action="store_true")
    parser.add_argument("--use-cellpose", action="store_true")
    parser.add_argument("--cellpose-model", default="cyto2", choices=["cyto", "cyto2", "nuclei"])
    parser.add_argument("--cellpose-diameter", type=float, default=25.0)
    parser.add_argument("--cellpose-flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellpose-cellprob-threshold", type=float, default=-1.0)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    f2c = SPERM_V2_FOLDER_CELL_TYPE
    if args.folder:
        if args.folder not in f2c:
            parser.error(f"Unknown folder '{args.folder}'. Choices: {list(f2c)}")
        f2c = {args.folder: f2c[args.folder]}

    run_batch(
        args.data, args.output,
        patch_size=args.patch_size, patch_overlap=args.patch_overlap,
        save_patches=args.save_patches, use_cellpose=args.use_cellpose,
        cellpose_diameter=args.cellpose_diameter, cellpose_model=args.cellpose_model,
        cellpose_flow_threshold=args.cellpose_flow_threshold,
        cellpose_cellprob_threshold=args.cellpose_cellprob_threshold, use_gpu=args.use_gpu,
        folder_to_cell_type=f2c, tile_filter=_is_sperm_v2_tile,
    )


if __name__ == "__main__":
    main()
