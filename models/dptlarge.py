import h5py
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import scipy.io


# Load a sample image and depth map from NYU Depth V2 dataset
data = h5py.File("data/raw/nyu_depth_v2_labeled.mat", "r")
images = np.array(data['images'])
depths = np.array(data['depths'])

# Reorder dimensions to HWC format
images = np.transpose(images, (3, 2, 1, 0))  
depths = np.transpose(depths, (2, 1, 0))      


# Select an image and its corresponding depth map
index = 0
image = Image.fromarray(images[:, :, :, index])
depth_map = depths[:, :, index]

image = image.convert("RGB")


processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# prepare image for the model
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth_map = Image.fromarray(formatted)
image.show()
depth_map.show()