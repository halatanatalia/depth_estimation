import h5py
import numpy as np
from PIL import Image

EPS = 1e-8


def normalize_depth_for_visualization(depth: np.ndarray) -> np.ndarray:
    """Normalize depth map to uint8 for visualization."""

    depth = np.asarray(depth, dtype=np.float32)
    vis = np.zeros(depth.shape, dtype=np.uint8)

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return vis

    d_min = float(np.nanmin(depth[valid]))
    d_max = float(np.nanmax(depth[valid]))
    if d_max - d_min <= EPS:
        vis[valid] = 255
        return vis

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - d_min) / (d_max - d_min + EPS)
    vis[valid] = np.clip(norm[valid] * 255.0, 0, 255).astype(np.uint8)
    return vis


def load_rgbd_data(data_path: str, index: int):
    """Load RGB image, raw RGB-D map and dense ground-truth depth from NYU Depth V2.

    Returns:
        rgb_image:       PIL RGB image
        depth_rgbd_raw:  raw Kinect depth (with holes) in meters
        depth_gt:        dense / inpainted depth map in meters
        depth_rgbd_vis:  uint8 visualization of raw RGB-D
    """
    with h5py.File(data_path, "r") as f:
        # HDF5 layout in nyu_depth_v2_labeled.mat:
        # images[index]     -> (3, 640, 480)
        # rawDepths[index]  -> (640, 480)
        # depths[index]     -> (640, 480)
        rgb = np.asarray(f["images"][index], dtype=np.uint8).transpose(2, 1, 0)

        if "rawDepths" in f:
            depth_rgbd_raw = np.asarray(f["rawDepths"][index], dtype=np.float32).T
        else:
            depth_rgbd_raw = np.asarray(f["depths"][index], dtype=np.float32).T

        depth_gt = np.asarray(f["depths"][index], dtype=np.float32).T

    rgb_image = Image.fromarray(rgb).convert("RGB")

    depth_rgbd_raw[depth_rgbd_raw <= 0] = np.nan
    depth_gt[depth_gt <= 0] = np.nan

    depth_rgbd_vis = normalize_depth_for_visualization(depth_rgbd_raw)
    return rgb_image, depth_rgbd_raw, depth_gt, depth_rgbd_vis
