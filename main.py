from src.fusion import fuse_depth_maps
from src.metrics import compute_metrics, prepare_dpt_for_metrics
from src.metrics_logger import append_metrics, init_metrics_file
from src.monocular import estimate_monocular_depth, load_monocular_model
from src.rgbd import load_rgbd_data
from src.visualisation import visualize_depths

# 21 representative NYU Depth V2 samples used in the thesis experiments.
INDEXES = [
    11, 142, 314,   # kitchen
    525, 780, 1290,  # living room
    654, 921, 1067,  # bedroom
    105, 676, 836,  # bathroom
    370, 394, 619,       # office
    432, 1355, 1412,       # dining room
    431, 441, 447,       # children's room
]

DATA_PATH = "data/raw/nyu_depth_v2_labeled.mat"
DPT_METRICS_PATH = "data/results/dpt_metrics.csv"
FUSION_METRICS_PATH = "data/results/fusion_metrics.csv"

def main():
    processor, model, device = load_monocular_model()

    init_metrics_file(FUSION_METRICS_PATH, overwrite=True)
    init_metrics_file(DPT_METRICS_PATH, overwrite=True)

    for idx in INDEXES:
        print(f"Processing image {idx:04d}...")

        rgb, depth_rgbd_raw, depth_gt, depth_rgbd_vis = load_rgbd_data(DATA_PATH, index=idx)
        depth_dpt_vis, depth_dpt_raw = estimate_monocular_depth(rgb, processor, model, device)

        # DPT-Large is affine-invariant. Before metric evaluation we align the
        # raw disparity-like prediction to dense metric depth.
        depth_dpt_metrics = prepare_dpt_for_metrics(
            depth_dpt_raw,
            depth_gt,
            )

        # Fusion uses raw RGB-D measurements (with holes) and the raw DPT output.
        depth_fused_raw, depth_fused_vis = fuse_depth_maps(depth_rgbd_raw, depth_dpt_raw)

        dpt_metrics = compute_metrics(
            depth_gt,
            depth_dpt_metrics,
        )
        fusion_metrics = compute_metrics(
            depth_gt,
            depth_fused_raw,
        )

        append_metrics(DPT_METRICS_PATH, image_id=f"image_{idx:04d}", metrics_dict=dpt_metrics)
        append_metrics(FUSION_METRICS_PATH, image_id=f"image_{idx:04d}", metrics_dict=fusion_metrics)

        visualize_depths(
            rgb_image=rgb,
            depth_rgbd=depth_rgbd_vis,
            depth_dpt=depth_dpt_vis,
            depth_fused=depth_fused_vis,
            output_path=f"data/processed/depth_comparison_{idx:04d}.png",
        )


if __name__ == "__main__":
    main()
