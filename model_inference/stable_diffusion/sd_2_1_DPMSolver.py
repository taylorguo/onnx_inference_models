# PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 python sd_2_1_DPMSolver.py

import torch
# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1",    
#     torch_dtype=torch.float16,
# )
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# prompt = 'a single chinese girl with black eyes holds a white picture'

# # pipe.enable_attention_slicing()
# # pipe.enable_sequential_cpu_offload()
# with torch.cuda.amp.autocast():
#     image = pipe(prompt).images[0]

#     image.save("rides_horse.png")


from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
# with torch.cuda.amp.autocast():
image = pipe(prompt).images[0]    
image.save("astronaut_rides_horse.png")
