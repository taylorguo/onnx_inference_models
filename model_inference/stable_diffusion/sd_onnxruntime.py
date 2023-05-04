import re
import argparse
# We need os.path for isdir
import os.path
# Numpy is used to provide a random generator
import numpy


from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default= "stabilityai/stable-diffusion-2-1",
        required=False, #True,
        help="Directory in current location to load model from",
    )

    parser.add_argument(
        "--size",
        default=512,
        type=int,
        required=False,
        help="Width/Height of the picture, defaults to 512, use 768 when appropriate",
    )

    parser.add_argument(
        "--steps",
        default=30,
        type=int,
        required=False,
        help="Scheduler steps to use",
    )

    parser.add_argument(
        "--scale",
        default=7.5,
        type=float,
        required=False,
        help="Guidance scale (how strict it sticks to the prompt)"
    )

    parser.add_argument(
        "--prompt",
        default="a dog on a lawn with the eifel tower in the background",
        type=str,
        required=False,
        help="Text prompt for generation",
    )

    parser.add_argument(
        "--negprompt",
        default="blurry, low quality",
        type=str,
        required=False,
        help="Negative text prompt for generation (what to avoid)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Seed for generation, allows you to get the exact same image again",
    )

    parser.add_argument(
        "--cpu-textenc", "--cpuclip",
        action="store_true",
        help="Load Text Encoder on CPU to save VRAM"
    )

    parser.add_argument(
        "--cpuvae",
        action="store_true",
        help="Load VAE on CPU, this will always load the Text Encoder on CPU too"
    )

    args = parser.parse_args()

    VAECPU = TECPU = False
    if args.cpuvae:
        VAECPU = TECPU = True
    if args.cpu_textenc:
        TECPU=True
    print(" ****** TECPU: ", TECPU, "  ****** VAECPU: ", VAECPU)
    # if match := re.search(r"([^/\\]*)[/\\]?$", args.model):
    #     fmodel = match.group(1)
    fmodel = ""
    generator=numpy.random
    imgname="testpicture-"+fmodel+"_"+str(args.size)+".png"
    if args.seed is not None:
        generator.seed(args.seed)
        imgname="testpicture-"+fmodel+"_"+str(args.size)+"_seed"+str(args.seed)+".png"

    if  os.path.isdir(args.model+"/unet"):
        height=args.size
        width=args.size
        num_inference_steps=args.steps
        guidance_scale=args.scale
        prompt = args.prompt
        negative_prompt = args.negprompt
        if TECPU:
            cputextenc=OnnxRuntimeModel.from_pretrained(args.model+"/text_encoder")
            if VAECPU:
                cpuvae=OnnxRuntimeModel.from_pretrained(args.model+"/vae_decoder")
                pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                    provider="DmlExecutionProvider", text_encoder=cputextenc, vae_decoder=cpuvae,
                    vae_encoder=None)
            else:
                pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                    provider="DmlExecutionProvider", text_encoder=cputextenc)
        else:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                provider="DmlExecutionProvider")
        image = pipe(prompt, width, height, num_inference_steps, guidance_scale,
                            negative_prompt,generator=generator).images[0]
        image.save(imgname)
    else:
        print("model not found")