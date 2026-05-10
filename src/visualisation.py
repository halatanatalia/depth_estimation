import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for saving images
from PIL import Image
from pathlib import Path

def visualize_depths(rgb_image, depth_rgbd, depth_dpt, depth_fused, output_path):
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("RGB Image")
    plt.imshow(rgb_image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Depth Map (RGB-D)")
    plt.imshow(depth_rgbd, cmap='magma')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Depth Map (DPT)")
    plt.imshow(depth_dpt, cmap='magma')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Fused Depth Map")
    plt.imshow(depth_fused, cmap='magma')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()