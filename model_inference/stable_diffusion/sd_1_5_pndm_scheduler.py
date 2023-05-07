import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    "onnx_1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
# prompt = 'a single chinese girl with black eyes holds a white picture'

pipe.enable_attention_slicing()
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")