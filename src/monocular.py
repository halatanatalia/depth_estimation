import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

from src.rgbd import normalize_depth_for_visualization

EPS = 1e-8


def load_monocular_model():
    """Load the zero-shot Intel DPT-Large model from Hugging Face."""
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def estimate_monocular_depth(image, processor, model, device):
    """Estimate monocular depth with DPT-Large.

    Returns:
        formatted_vis: uint8 visualization (depth-like orientation)
        prediction:    raw DPT output as float32; for Intel/dpt-large this is
                       best treated as relative inverse depth / disparity.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)

    depth_like = 1.0 / np.clip(prediction, EPS, None)
    formatted_vis = normalize_depth_for_visualization(depth_like)
    return formatted_vis, prediction
