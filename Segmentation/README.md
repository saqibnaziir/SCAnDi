# SCAnDI — Sperm Cell Segmentation

Cell segmentation pipeline for **sperm cells** from red-stained microscopy images (Christmas Tree / nuclear fast red). Sperm cell only.

---

## Overview

This module produces binary masks, instance label maps, and per-cell metadata for sperm cell microscopy slides. It supports:

- **Classical pipeline**: CLAHE enhancement → colour-based foreground detection (HSV + pinkness score) → morphological cleanup → DoG-pinkness watershed → red-stain post-filter
- **Cellpose pipeline**: Deep-learning segmentation on raw images (no pre-masking), followed by red-stain post-filter

For dense images (>20% foreground), images are split into overlapping 512×512 patches, segmented per-patch, and stitched with IoU-based deduplication.

---

## Requirements

- Python 3.9+
- OpenCV, NumPy, scikit-image, SciPy, Matplotlib

Optional for deep learning:

- Cellpose (`pip install cellpose`)

---

## Installation

```bash
cd Segmentation
pip install -r requirements.txt
```

If using Cellpose:

```bash
pip install cellpose
```

---

## Supported Datasets

| Dataset       | Structure                    | Folders          |
|---------------|------------------------------|------------------|
| SCAnDI        | `data_root/S001`, `S011`     | S001, S011       |
| Sperm_V2      | `data_root/Image-01` … `Image-06` | Image-01 … Image-06 |

**Sperm_V2**: Tile images only. Files with `_mNNN` (e.g. `*_m000.tif`) are processed; whole-slide overviews (`*Tiled*`) and `.lsm` files are skipped.

---

## Data Paths

Place your data so that the structure matches:

```
<DATA_ROOT>/
├── Image-01/        # Sperm_V2
│   ├── *.tif        # tiles: *_m000.tif, *_m001.tif, ...
│   └── *Tiled*.tif  # skipped (whole-slide)
├── Image-02/
...
└── Image-06/
```

Or for SCAnDI:

```
<DATA_ROOT>/
├── S001/
│   └── *.tif
└── S011/
    └── *.tif
```

---

## Usage

### 1. EDA (Sperm_V2, optional)

```bash
python eda_sperm_v2.py --data /path/to/Sperm_V2 [--output /path/to/eda_report]
```

Reports tile counts, image shapes, and pinkness stats per folder.

### 2. Segmentation — Sperm_V2 (tiles only)

```bash
python run_sperm_v2.py --data /path/to/Sperm_V2 --output /path/to/output
```

With Cellpose:

```bash
python run_sperm_v2.py --data /path/to/Sperm_V2 --output /path/to/output --use-cellpose [--use-gpu]
```

Single folder:

```bash
python run_sperm_v2.py --data /path/to/Sperm_V2 --output /path/to/output --folder Image-01
```

### 3. Segmentation — SCAnDI (maps.py directly)

```bash
python maps.py --data /path/to/SCAnDI_data --output /path/to/output
```

With Cellpose:

```bash
python maps.py --data /path/to/SCAnDI_data --output /path/to/output --use-cellpose
```

Single image:

```bash
python maps.py --single /path/to/image.tif sperm --output /path/to/output
```

---

## Output Structure

```
<output>/
├── GT_masks/
│   └── <folder>/
│       ├── <stem>_binary.png
│       ├── <stem>_instance.npy
│       ├── <stem>_instance_vis.png
│       └── <stem>_metadata.json
├── figures/
│   └── <folder>/
│       └── <stem>_seg.png
└── batch_summary.json
```

- `*_binary.png`: Binary foreground mask  
- `*_instance.npy`: Instance label map (NumPy int32)  
- `*_instance_vis.png`: Colour overlay  
- `*_metadata.json`: Per-cell properties (area, centroid, bbox, etc.)

---

## Tuning (Cellpose)

- **`--cellpose-diameter`**: Cell diameter in pixels (e.g. 25). Measure several cells and average.
- **`--cellpose-flow-threshold`**: Higher (0.6–0.8) → more cells.
- **`--cellpose-cellprob-threshold`**: Lower (e.g. -2.0) → more permissive.

---

## Files

| File             | Description                                 |
|------------------|---------------------------------------------|
| `maps.py`        | Core pipeline: classical + Cellpose, patch stitch |
| `run_sperm_v2.py`| Batch runner for Sperm_V2 (tiles only)      |
| `eda_sperm_v2.py`| EDA for Sperm_V2                            |
| `requirements.txt` | Python dependencies                        |
