from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1").to("cuda")

prompt = '飞流直下三千尺，油画'
prompt = "那人却在灯火阑珊处，色彩艳丽，古风，资深插画师作品，桌面高清壁纸。"
prompt = "日出, 海面上, 4k壁纸"
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("日出.png")
