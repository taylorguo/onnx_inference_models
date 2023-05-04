import torch
from diffusers import DiffusionPipeline

model_name = "nicky007/stable-diffusion-logo-fine-tuned"

pipe = DiffusionPipeline.from_pretrained(
    model_name,    
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a cartoon logo of a little cute lion"

pipe.enable_attention_slicing()
image = pipe(prompt).images[0]

image.save("logo.png")