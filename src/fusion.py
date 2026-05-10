import cv2
import numpy as np

from src.rgbd import normalize_depth_for_visualization
from src.metrics import compute_scale_and_shift

EPS = 1e-8


def fuse_depth_maps(
    depth_rgbd_raw: np.ndarray | None,
    depth_dpt_raw: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    alpha: float = 0.8,
):
    """
    Simple metric-space fusion:
    - align DPT to RGB-D scale in disparity space,
    - on valid RGB-D pixels: weighted average,
    - on missing RGB-D pixels: use DPT only.
    """
    dpt_raw = np.asarray(depth_dpt_raw, dtype=np.float32)

    if depth_rgbd_raw is None:
        dpt_metric = 1.0 / np.clip(dpt_raw, EPS, None)
        dpt_metric = np.clip(dpt_metric, min_depth, max_depth)
        dpt_metric[~np.isfinite(dpt_metric)] = np.nan
        return dpt_metric, normalize_depth_for_visualization(dpt_metric)

    rgbd = np.asarray(depth_rgbd_raw, dtype=np.float32)
    if rgbd.shape != dpt_raw.shape:
        rgbd = cv2.resize(
            rgbd,
            (dpt_raw.shape[1], dpt_raw.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    valid = np.isfinite(rgbd) & (rgbd > min_depth) & (rgbd < max_depth)

    if np.any(valid):
        rgbd_disp = np.zeros_like(rgbd, dtype=np.float32)
        rgbd_disp[valid] = 1.0 / np.clip(rgbd[valid], min_depth, None)

        scale, shift = compute_scale_and_shift(dpt_raw, rgbd_disp, valid)
        dpt_disp = scale * dpt_raw + shift
        dpt_disp = np.maximum(dpt_disp, 1.0 / max_depth)

        dpt_metric = 1.0 / dpt_disp
    else:
        dpt_metric = 1.0 / np.clip(dpt_raw, EPS, None)

    dpt_metric = np.clip(dpt_metric, min_depth, max_depth)
    dpt_metric[~np.isfinite(dpt_metric)] = np.nan

    fused = dpt_metric.copy()
    fused[valid] = alpha * rgbd[valid] + (1.0 - alpha) * dpt_metric[valid]

    fused = np.clip(fused, min_depth, max_depth)
    fused[~np.isfinite(fused)] = np.nan

    fused_vis = normalize_depth_for_visualization(fused)
    return fused.astype(np.float32), fused_vis
