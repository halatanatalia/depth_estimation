from transformers import pipeline
from accelerate import Accelerator
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import requests
import torch
device = Accelerator().device
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint, device=device)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
predictions = pipe(image)
depth_map = predictions["depth"]
depth_map.show()
