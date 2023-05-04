# diffusers-0.15.0 safetensors-0.3.0 transformers-4.28.1
# pip install accelerate -i https://pypi.douban.com/simple

import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a bike on moon"
# prompt = 'a single chinese girl with black eyes holds a white picture'
prompt = "一头猪"
image = pipe(prompt).images[0]  

image.save("astronaut_rides_horse.png")

