"""
SCAnDI Project — Cell Segmentation Pipeline
============================================
Generates pseudo-GT segmentation maps for sperm cells from Christmas Tree
or nuclear fast red stained microscopy images. Sperm cell only.

Pipeline overview
-----------------
Classical mode (default):
  1. CLAHE contrast enhancement
  2. Colour-based foreground detection  (HSV + pinkness score)
  3. Morphological cleanup              (remove noise & fill holes)
  4. DoG-pinkness watershed             (Difference of Gaussians seeds)
  5. Size + shape filtering
  6. Instance map export

Cellpose mode (--use-cellpose):
  1. Raw image → Cellpose (NO pre-masking — model sees the full slide)
  2. Post-filter by red staining to remove false positives
  3. Instance map export

Patch-based mode: Images split into overlapping 512×512 patches,
segmented per-patch, then stitched with IoU-based deduplication.
"""

import os, json, logging, re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage import measure
from skimage.color import label2rgb
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cell-type parameter profiles (sperm only)
# ──────────────────────────────────────────────────────────────────────────────
PROFILES = {
    'sperm': {
        'use_patches':        True,
        'fg_density_thresh':  0.20,
        'pinkness_thresh':    22,
        'val_min':            35,
        'val_max':            240,
        'sat_min':            18,
        'open_k':             3,
        'open_iter':          2,
        'close_k':            5,
        'close_iter':         2,
        'min_area':           60,
        'max_area':           3000,
        'dog_sigma_small':    2.0,
        'dog_sigma_large':    6.0,
        'dog_weight':         0.5,
        'ws_min_dist':        14,
        'ws_thresh_abs':      4.0,
        'ws_compactness':     0.005,
        'ws_min_area':        60,
        'ws_max_area':        3000,
        'clahe_clip':         2.0,
        'clahe_tile':         8,
        'min_mean_pinkness':  18,
    },
}

FOLDER_CELL_TYPE = {'S001': 'sperm', 'S011': 'sperm'}

SPERM_V2_FOLDER_CELL_TYPE = {
    'Image-01': 'sperm', 'Image-02': 'sperm', 'Image-03': 'sperm',
    'Image-04': 'sperm', 'Image-05': 'sperm', 'Image-06': 'sperm',
}


def _is_sperm_v2_tile(path: Path) -> bool:
    """True if path is a tile to segment (excludes .lsm, *Tiled*, whole-slide)."""
    if path.suffix.lower() not in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
        return False
    name = path.name
    if 'Tiled' in name or ' Tiled' in name:
        return False
    return bool(re.search(r'_m\d+', name))


def _pinkness(img_bgr: np.ndarray) -> np.ndarray:
    r = img_bgr[:, :, 2].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    b = img_bgr[:, :, 0].astype(np.float32)
    return r - 0.5 * (g + b)


def _apply_clahe(img_bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return cv2.cvtColor(cv2.merge([h, s, clahe.apply(v)]), cv2.COLOR_HSV2BGR)


def _fast_foreground_coverage(img_bgr: np.ndarray, pinkness_thresh: float = 15) -> float:
    sample = _pinkness(img_bgr)[::4, ::4]
    return float((sample > pinkness_thresh).sum()) / sample.size


def _iou_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def _filter_regions_by_red_staining(img_bgr: np.ndarray, instance_map: np.ndarray,
                                    p: dict, apply_filter: bool = True) -> np.ndarray:
    if not apply_filter or instance_map.max() == 0:
        return instance_map
    min_pink = p.get('min_mean_pinkness', 18)
    pink = _pinkness(_apply_clahe(img_bgr, p['clahe_clip'], p['clahe_tile']))
    out = np.zeros_like(instance_map, dtype=np.int32)
    new_label = 1
    for r in measure.regionprops(instance_map):
        if r.label == 0:
            continue
        region_mask = instance_map == r.label
        if float(pink[region_mask].mean()) >= min_pink:
            out[region_mask] = new_label
            new_label += 1
    return out


def _build_foreground_mask(img_bgr: np.ndarray, p: dict) -> np.ndarray:
    img_e = _apply_clahe(img_bgr, p['clahe_clip'], p['clahe_tile'])
    pink = _pinkness(img_e)
    _, s, v = cv2.split(cv2.cvtColor(img_e, cv2.COLOR_BGR2HSV))
    mask = (
        (pink > p['pinkness_thresh']) &
        (v > p['val_min']) & (v < p['val_max']) &
        (s > p['sat_min'])
    ).astype(np.uint8) * 255
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['open_k'], p['open_k']))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['close_k'], p['close_k']))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o, iterations=p['open_iter'])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c, iterations=p['close_iter'])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o, iterations=1)
    return mask


def _dog_watershed(fg_mask: np.ndarray, img_bgr: np.ndarray, p: dict) -> np.ndarray:
    img_e = _apply_clahe(img_bgr, p['clahe_clip'], p['clahe_tile'])
    pink = _pinkness(img_e).clip(0, 200)
    ps = gaussian_filter(pink, sigma=p['dog_sigma_small'])
    pl = gaussian_filter(pink, sigma=p['dog_sigma_large'])
    dog = np.maximum(ps - p['dog_weight'] * pl, 0)
    mask_bool = fg_mask.astype(bool)
    coords = peak_local_max(dog, min_distance=p['ws_min_dist'],
                            threshold_abs=p['ws_thresh_abs'],
                            labels=mask_bool, exclude_border=False)
    if len(coords) == 0:
        return np.zeros(fg_mask.shape, dtype=np.int32)
    seeds = np.zeros(pink.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers = ndimage.label(seeds)[0]
    ws = watershed(-ps, markers, mask=mask_bool, compactness=p['ws_compactness'])
    out = np.zeros_like(ws, dtype=np.int32)
    new_lbl = 1
    for lbl in np.unique(ws):
        if lbl == 0:
            continue
        m = ws == lbl
        if p['ws_min_area'] <= m.sum() <= p['ws_max_area']:
            out[m] = new_lbl
            new_lbl += 1
    return out


def segment_image(img_bgr: np.ndarray, cell_type: str) -> dict:
    assert cell_type in PROFILES
    p = PROFILES[cell_type]
    mask = _build_foreground_mask(img_bgr, p)
    inst = _dog_watershed(mask, img_bgr, p)
    fg_frac = mask.astype(bool).mean()
    inst = _filter_regions_by_red_staining(img_bgr, inst, p, apply_filter=(fg_frac < 0.85))
    binary = (inst > 0).astype(np.uint8) * 255
    return {
        'binary_mask': binary, 'instance_map': inst, 'cell_count': int(inst.max()),
        'region_props': measure.regionprops(inst), 'cell_type': cell_type, 'profile': p,
    }


def segment_image_cellpose(img_bgr: np.ndarray, cell_type: str,
                           diameter: float = 25.0, model_type: str = 'cyto2',
                           use_gpu: bool = False, flow_threshold: float = 0.4,
                           cellprob_threshold: float = -1.0,
                           apply_pinkness_filter: bool = True) -> dict:
    try:
        from cellpose import models as cp_models
    except ImportError:
        raise ImportError("Cellpose not installed. Run: pip install cellpose")
    assert cell_type in PROFILES
    p = PROFILES[cell_type]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    log.info(f"    Cellpose ({model_type}, diameter={diameter}px)")
    try:
        model = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)
    except Exception:
        model = cp_models.Cellpose(gpu=use_gpu, model_type=model_type)
    eval_result = model.eval(
        img_rgb, diameter=diameter, channels=[2, 1],
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
        normalize=True,
    )
    masks = eval_result[0]
    if isinstance(masks, list):
        masks = masks[0]
    instance_map = masks.astype(np.int32)
    fg_frac = _fast_foreground_coverage(img_bgr, p['pinkness_thresh'])
    instance_map = _filter_regions_by_red_staining(
        img_bgr, instance_map, p,
        apply_filter=(apply_pinkness_filter and fg_frac < 0.85),
    )
    binary = (instance_map > 0).astype(np.uint8) * 255
    return {
        'binary_mask': binary, 'instance_map': instance_map,
        'cell_count': int(instance_map.max()), 'region_props': measure.regionprops(instance_map),
        'cell_type': cell_type, 'profile': p, 'cellpose_diam': float(diameter),
    }


def patch_and_segment(img_bgr: np.ndarray, cell_type: str,
                      patch_size: int = 512, patch_overlap: int = 64,
                      save_patch_figures: bool = False, use_cellpose: bool = False,
                      cellpose_diameter: float = 25.0, cellpose_model: str = 'cyto2',
                      cellpose_flow_threshold: float = 0.4,
                      cellpose_cellprob_threshold: float = -1.0,
                      use_gpu: bool = False) -> dict:
    assert cell_type in PROFILES
    p = PROFILES[cell_type]
    stride = patch_size - patch_overlap
    H, W = img_bgr.shape[:2]
    patches_info = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            patch = img_bgr[y:min(y+patch_size, H), x:min(x+patch_size, W)]
            ph, pw = patch.shape[:2]
            if ph < patch_size or pw < patch_size:
                patch = np.pad(patch, ((0, patch_size-ph), (0, patch_size-pw), (0, 0)), mode='edge')
            patches_info.append((y, x, patch))
    log.info(f"    {len(patches_info)} patches ({patch_size}px, stride={stride}px)")
    all_regions = []
    patch_results_list = [] if save_patch_figures else None
    for y0, x0, patch in patches_info:
        if use_cellpose:
            res = segment_image_cellpose(patch, cell_type, diameter=cellpose_diameter,
                                         model_type=cellpose_model, use_gpu=use_gpu,
                                         flow_threshold=cellpose_flow_threshold,
                                         cellprob_threshold=cellpose_cellprob_threshold)
        else:
            res = segment_image(patch, cell_type)
        if save_patch_figures:
            patch_results_list.append((y0, x0, patch, res))
        inst = res['instance_map']
        y_ext = min(patch_size, H - y0)
        x_ext = min(patch_size, W - x0)
        for r in measure.regionprops(inst):
            if r.label == 0:
                continue
            lm = (inst == r.label).astype(np.uint8)[:y_ext, :x_ext]
            if lm.sum() == 0:
                continue
            all_regions.append((lm, int(r.area), y0, x0, y0+lm.shape[0], x0+lm.shape[1]))
    all_regions.sort(key=lambda x: x[1], reverse=True)
    stitched = np.zeros((H, W), dtype=np.int32)
    next_label = 1
    for lm, area, gr0, gc0, gr1, gc1 in all_regions:
        cand = np.zeros((H, W), dtype=np.uint8)
        cand[gr0:gr1, gc0:gc1] = lm
        overlap = np.logical_and(cand > 0, stitched > 0)
        if not overlap.any():
            stitched[gr0:gr1, gc0:gc1][(lm > 0) & (stitched[gr0:gr1, gc0:gc1] == 0)] = next_label
            next_label += 1
            continue
        dup = False
        for lbl in np.unique(stitched[overlap]):
            if lbl > 0 and _iou_mask(cand, (stitched == lbl).astype(np.uint8)) > 0.3:
                dup = True
                break
        if not dup:
            stitched[gr0:gr1, gc0:gc1][(lm > 0) & (stitched[gr0:gr1, gc0:gc1] == 0)] = next_label
            next_label += 1
    binary = (stitched > 0).astype(np.uint8) * 255
    out = {
        'binary_mask': binary, 'instance_map': stitched, 'cell_count': next_label - 1,
        'region_props': measure.regionprops(stitched), 'cell_type': cell_type, 'profile': p,
    }
    if patch_results_list is not None:
        out['patch_results'] = patch_results_list
    return out


def make_segmentation_figure(img_bgr: np.ndarray, result: dict, image_name: str = '') -> plt.Figure:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    binary = result['binary_mask']
    inst_map = result['instance_map']
    count = result['cell_count']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SCAnDI Segmentation — {image_name}\n'
                 f'Cell type: {result["cell_type"].capitalize()}  |  Detected: {count} cells',
                 fontsize=13, fontweight='bold')
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[1, 0].imshow(label2rgb(inst_map, image=img_rgb, bg_label=0, alpha=0.5, bg_color=(0, 0, 0)))
    axes[1, 0].set_title('Instance Segmentation Map')
    contour_img = img_rgb.copy()
    for lbl in np.unique(inst_map):
        if lbl == 0:
            continue
        cell_mask = (inst_map == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, cnts, -1, (255, 50, 50), 1)
        M = cv2.moments(cell_mask)
        if M['m00'] > 0:
            cv2.circle(contour_img, (int(M['m10']/M['m00']), int(M['m01']/M['m00'])), 2, (255, 230, 0), -1)
    axes[1, 1].imshow(contour_img)
    axes[1, 1].set_title(f'Contour Overlay — {count} cells')
    axes[1, 1].legend(handles=[
        Patch(facecolor='#ff3232', label='Cell boundary'),
        Patch(facecolor='#ffe600', label='Cell centroid'),
    ], loc='lower right', fontsize=8)
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    return fig


def save_gt_masks(result: dict, out_dir: Path, stem: str, save_patches: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_map = result['instance_map']
    binary = result['binary_mask']
    cv2.imwrite(str(out_dir / f'{stem}_binary.png'), binary)
    np.save(str(out_dir / f'{stem}_instance.npy'), inst_map)
    vis = (label2rgb(inst_map, image=np.zeros((*inst_map.shape, 3), dtype=np.uint8),
                     bg_label=0, bg_color=(0, 0, 0)) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f'{stem}_instance_vis.png'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    meta = {
        'image': stem, 'cell_type': result['cell_type'], 'cell_count': result['cell_count'],
        'cells': [{
            'id': int(r.label), 'area': float(r.area),
            'centroid_y': float(r.centroid[0]), 'centroid_x': float(r.centroid[1]),
            'bbox': list(r.bbox), 'eccentricity': float(r.eccentricity),
            'solidity': float(r.solidity),
            'major_axis': float(r.major_axis_length), 'minor_axis': float(r.minor_axis_length),
        } for r in result['region_props']],
    }
    with open(out_dir / f'{stem}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    if save_patches and 'patch_results' in result:
        patch_dir = out_dir / f'{stem}_patches'
        patch_dir.mkdir(parents=True, exist_ok=True)
        for y0, x0, pi, pr in result['patch_results']:
            fig = make_segmentation_figure(pi, pr, f'patch_{y0}_{x0}')
            fig.savefig(str(patch_dir / f'patch_{y0}_{x0}_seg.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
    log.info(f"  Saved → {out_dir}/{stem}_*")
    return meta


def run_batch(data_root: str | Path, output_root: str | Path,
              extensions: tuple = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
              patch_size: int = 512, patch_overlap: int = 64, save_patches: bool = False,
              use_cellpose: bool = False, cellpose_diameter: float = 25.0,
              cellpose_model: str = 'cyto2', cellpose_flow_threshold: float = 0.4,
              cellpose_cellprob_threshold: float = -1.0, use_gpu: bool = False,
              folder_to_cell_type: dict | None = None,
              tile_filter: Callable[[Path], bool] | None = None):
    f2c = folder_to_cell_type if folder_to_cell_type is not None else FOLDER_CELL_TYPE
    data_root = Path(data_root)
    output_root = Path(output_root)
    gt_root = output_root / 'GT_masks'
    fig_root = output_root / 'figures'
    summary = {}
    total_imgs = total_cells = 0
    for folder_name, cell_type in f2c.items():
        folder = data_root / folder_name
        if not folder.exists():
            log.warning(f"Folder not found: {folder} — skipping")
            continue
        all_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        files = sorted(f for f in all_files if tile_filter(f)) if tile_filter else sorted(all_files)
        log.info(f"\n{'='*60}\n{folder_name} ({cell_type}) — {len(files)} images")
        folder_summary = {'cell_type': cell_type, 'images': {}}
        for img_path in files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                log.warning(f"  Cannot read {img_path.name}")
                continue
            stem = img_path.stem
            p = PROFILES[cell_type]
            fg_frac = _fast_foreground_coverage(img_bgr, p['pinkness_thresh'])
            log.info(f"  [{stem}] shape={img_bgr.shape[:2]}  fg={fg_frac:.1%}")
            if fg_frac > p['fg_density_thresh']:
                log.info("    Dense → patch pipeline")
                result = patch_and_segment(img_bgr, cell_type, patch_size=patch_size,
                    patch_overlap=patch_overlap, save_patch_figures=save_patches,
                    use_cellpose=use_cellpose, cellpose_diameter=cellpose_diameter,
                    cellpose_model=cellpose_model, cellpose_flow_threshold=cellpose_flow_threshold,
                    cellpose_cellprob_threshold=cellpose_cellprob_threshold, use_gpu=use_gpu)
            else:
                log.info("    Sparse → full-image pipeline")
                if use_cellpose:
                    result = segment_image_cellpose(img_bgr, cell_type,
                        diameter=cellpose_diameter, model_type=cellpose_model, use_gpu=use_gpu,
                        flow_threshold=cellpose_flow_threshold,
                        cellprob_threshold=cellpose_cellprob_threshold)
                else:
                    result = segment_image(img_bgr, cell_type)
            log.info(f"    → {result['cell_count']} cells detected")
            meta = save_gt_masks(result, gt_root / folder_name, stem, save_patches)
            fig = make_segmentation_figure(img_bgr, result, stem)
            fp = fig_root / folder_name
            fp.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fp / f'{stem}_seg.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)
            folder_summary['images'][stem] = {'cell_count': result['cell_count'],
                'shape': list(img_bgr.shape[:2]), 'fg_coverage': round(fg_frac, 3)}
            total_imgs += 1
            total_cells += result['cell_count']
        summary[folder_name] = folder_summary
    summary['_totals'] = {'images_processed': total_imgs, 'total_cells_detected': total_cells}
    sp = output_root / 'batch_summary.json'
    with open(sp, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"\n{'='*60}")
    log.info(f"Done: {total_imgs} images, {total_cells} cells. Summary → {sp}")
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SCAnDI sperm cell segmentation')
    parser.add_argument('--data', help='Path to data root (batch mode)')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--patch-overlap', type=int, default=64)
    parser.add_argument('--save-patches', action='store_true')
    parser.add_argument('--use-cellpose', action='store_true')
    parser.add_argument('--cellpose-model', default='cyto2', choices=['cyto', 'cyto2', 'nuclei'])
    parser.add_argument('--cellpose-diameter', type=float, default=25.0)
    parser.add_argument('--cellpose-flow-threshold', type=float, default=0.4)
    parser.add_argument('--cellpose-cellprob-threshold', type=float, default=-1.0)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--single', nargs=2, metavar=('IMAGE', 'CELL_TYPE'),
                        help='Segment one image: --single path/img.tif sperm')
    args = parser.parse_args()
    if args.single:
        img_path, ctype = args.single
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f"Cannot read {img_path}"
        p = PROFILES[ctype]
        fg_frac = _fast_foreground_coverage(img_bgr, p['pinkness_thresh'])
        if fg_frac > p['fg_density_thresh']:
            result = patch_and_segment(img_bgr, ctype, use_cellpose=args.use_cellpose,
                cellpose_diameter=args.cellpose_diameter, cellpose_model=args.cellpose_model,
                cellpose_flow_threshold=args.cellpose_flow_threshold,
                cellpose_cellprob_threshold=args.cellpose_cellprob_threshold, use_gpu=args.use_gpu)
        else:
            result = segment_image_cellpose(img_bgr, ctype, diameter=args.cellpose_diameter,
                model_type=args.cellpose_model, use_gpu=args.use_gpu,
                flow_threshold=args.cellpose_flow_threshold,
                cellprob_threshold=args.cellpose_cellprob_threshold) if args.use_cellpose else segment_image(img_bgr, ctype)
        out = Path(args.output)
        save_gt_masks(result, out, Path(img_path).stem)
        fig = make_segmentation_figure(img_bgr, result, Path(img_path).stem)
        fig.savefig(str(out / f'{Path(img_path).stem}_seg.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"Cells detected: {result['cell_count']}. Results → {out}")
    else:
        if not args.data:
            parser.error('--data is required for batch mode')
        run_batch(args.data, args.output, patch_size=args.patch_size, patch_overlap=args.patch_overlap,
                  save_patches=args.save_patches, use_cellpose=args.use_cellpose,
                  cellpose_diameter=args.cellpose_diameter, cellpose_model=args.cellpose_model,
                  cellpose_flow_threshold=args.cellpose_flow_threshold,
                  cellpose_cellprob_threshold=args.cellpose_cellprob_threshold, use_gpu=args.use_gpu)
