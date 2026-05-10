import numpy as np
import cv2
 
EPS = 1e-8
 
 
def compute_scale_and_shift(prediction, target, mask):
    """
    Compute optimal scale and shift to align prediction to target on valid pixels.
    """
    pred = prediction[mask].astype(np.float64)
    tgt = target[mask].astype(np.float64)
 
    a00 = np.sum(pred * pred)
    a01 = np.sum(pred)
    a11 = float(pred.size)
 
    b0 = np.sum(pred * tgt)
    b1 = np.sum(tgt)
 
    det = a00 * a11 - a01 * a01
    if det <= EPS:
        return 1.0, 0.0
 
    scale = (a11 * b0 - a01 * b1) / det
    shift = (-a01 * b0 + a00 * b1) / det
    return float(scale), float(shift)
 
 
def prepare_dpt_for_metrics(depth_dpt_raw, depth_rgbd_raw,
                            min_depth=0.1, max_depth=10.0):
    """
    Align DPT disparity-like output to metric depth using valid RGB-D pixels.
    """
    dpt = depth_dpt_raw.copy().astype(np.float32)
    rgbd = np.nan_to_num(depth_rgbd_raw, nan=0.0).astype(np.float32)
 
    if rgbd.shape != dpt.shape:
        rgbd = cv2.resize(
            rgbd,
            (dpt.shape[1], dpt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
 
    valid = np.isfinite(rgbd) & (rgbd > min_depth) & (rgbd < max_depth)
    if valid.sum() < 10:
        return np.full_like(dpt, np.nan, dtype=np.float32)
 
    gt_disp = np.zeros_like(rgbd)
    gt_disp[valid] = 1.0 / rgbd[valid]
 
    scale, shift = compute_scale_and_shift(dpt, gt_disp, valid)
    dpt_disp_aligned = scale * dpt + shift
 
    dpt_disp_aligned = np.maximum(dpt_disp_aligned, 1.0 / max_depth)
    dpt_metric = 1.0 / dpt_disp_aligned
    dpt_metric = np.clip(dpt_metric, min_depth, max_depth)
    dpt_metric[~np.isfinite(dpt_metric)] = min_depth
 
    return dpt_metric.astype(np.float32)
 
 
def compute_metrics(gt_depth, pred_depth):
    """
    Compute error metrics between ground truth and predicted depth maps.
    """
 
    valid = np.isfinite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 10.0)
    valid &= np.isfinite(pred_depth) & (pred_depth > 0)
 
    if not np.any(valid):
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "AbsRel": np.nan,
            "SqRel": np.nan,
            "δ<1.25": np.nan,
            "δ<1.25²": np.nan,
            "δ<1.25³": np.nan,
            "scale": np.nan,
        }
 
    gt_v = gt_depth[valid]
    pred_v = pred_depth[valid]
 
    scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
    pred_v = pred_v * scale
 
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
    mae = np.mean(np.abs(pred_v - gt_v))
    
    abs_rel = np.mean(np.abs(pred_v - gt_v) / gt_v)
    sq_rel = np.mean(((pred_v - gt_v) ** 2) / gt_v) 
 
    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
 
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "AbsRel": round(abs_rel, 4),
        "SqRel": round(sq_rel, 4),
        "δ<1.25": round(delta1 * 100, 2),
        "δ<1.25²": round(delta2 * 100, 2),
        "δ<1.25³": round(delta3 * 100, 2),
        "scale": round(scale, 4),
    }