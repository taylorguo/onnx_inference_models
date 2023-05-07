from pathlib import Path
import os
import time
import numpy
import torch
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline

from models.models import make_tokenizer
from models.utilities import DDIMScheduler, DPMScheduler, PNDMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler

from packaging import version

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        torch.onnx.export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        torch.onnx.export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16: bool = False, enable_convert: bool = False):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    output_path = Path(output_path)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    if enable_convert:
        onnx_export(
            pipeline.text_encoder,
            # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
            model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
            output_path=output_path / "text_encoder" / "model.onnx",
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=opset,
        )
    del pipeline.text_encoder

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    if enable_convert:
        onnx_export(
            pipeline.unet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.randn(2).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=unet_path,
            ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
            output_names=["out_sample"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset=opset,
            # use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    # unet = onnx.load(unet_model_path)
    # # clean up existing tensor files
    # shutil.rmtree(unet_dir)
    # os.mkdir(unet_dir)
    # # collate external tensor files into one
    # onnx.save_model(
    #     unet,
    #     unet_model_path,
    #     save_as_external_data=True,
    #     all_tensors_to_one_file=True,
    #     location="weights.pb",
    #     convert_attribute=False,
    # )
    del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
    if enable_convert:
        onnx_export(
            vae_encoder,
            model_args=(
                torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_encoder" / "model.onnx",
            ordered_input_names=["sample", "return_dict"],
            output_names=["latent_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=opset,
        )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    if enable_convert:
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_decoder" / "model.onnx",
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=opset,
        )
    del pipeline.vae

    # SAFETY CHECKER
    if pipeline.safety_checker is not None:
        safety_checker = pipeline.safety_checker
        clip_num_channels = safety_checker.config.vision_config.num_channels
        clip_image_size = safety_checker.config.vision_config.image_size
        safety_checker.forward = safety_checker.forward_onnx
        if enable_convert:
            onnx_export(
                pipeline.safety_checker,
                model_args=(
                    torch.randn(
                        1,
                        clip_num_channels,
                        clip_image_size,
                        clip_image_size,
                    ).to(device=device, dtype=dtype),
                    torch.randn(1, vae_sample_size, vae_sample_size, vae_out_channels).to(device=device, dtype=dtype),
                ),
                output_path=output_path / "safety_checker" / "model.onnx",
                ordered_input_names=["clip_input", "images"],
                output_names=["out_images", "has_nsfw_concepts"],
                dynamic_axes={
                    "clip_input": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                    "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
                },
                opset=opset,
            )
        del pipeline.safety_checker
        # safety_checker = OnnxRuntimeModel.from_pretrained(output_path / "safety_checker")
        # feature_extractor = pipeline.feature_extractor
    else:
        safety_checker = None
        feature_extractor = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        # vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        # vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained("onnx/vae.opt.onnx"),
        # text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained("onnx/clip.opt.onnx"),
        tokenizer=pipeline.tokenizer,
        # unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        unet=OnnxRuntimeModel.from_pretrained("onnx/unet.opt.onnx"),
        scheduler=pipeline.scheduler,
        # safety_checker=safety_checker,
        # feature_extractor=feature_extractor,
        # requires_safety_checker=safety_checker is not None,
    )

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    onnx_pipeline = OnnxStableDiffusionPipeline.from_pretrained(output_path, provider="CPUExecutionProvider")
    print("ONNX pipeline is loadable")


    prompt = "a cat on a lawn with the eifel tower in the background"
    negative_prompt = "blurry, low quality"
    width = 512
    height = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    generator=None         # numpy.random.seed(args.seed)

    image = onnx_pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, 
                          guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                          generator=generator).images[0]
    
    imgname="testpicture_"+str(width)+"_"+".png"
    image.save(imgname)


def onnx_runtime(output_path):

    safety_checker = None
    feature_extractor = None
    # version = "2.1"
    version = "1.5"
    hf_token = None
    scheduler_name = "PNDM"
    device = "cuda"

    # Schedule options
    sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
    if version in ("2.0", "2.1"):
        sched_opts['prediction_type'] = 'v_prediction'
    else:
        sched_opts['prediction_type'] = 'epsilon'
    
    if scheduler_name == "DDIM":
        scheduler = DDIMScheduler(device=device, **sched_opts)
    elif scheduler_name == "DPM":
        scheduler = DPMScheduler(device=device, **sched_opts)
    elif scheduler_name == "EulerA":
        scheduler = EulerAncestralDiscreteScheduler(device=device, **sched_opts)
    elif scheduler_name == "LMSD":
        scheduler = LMSDiscreteScheduler(device=device, **sched_opts)
    elif scheduler_name == "PNDM":
        sched_opts["steps_offset"] = 1
        scheduler = PNDMScheduler(device=device, **sched_opts)
    else:
        raise ValueError(f"Scheduler should be either DDIM, DPM, EulerA, LMSD or PNDM")

    # onnx_pipeline = OnnxStableDiffusionPipeline(
    #     vae_encoder=OnnxRuntimeModel.from_pretrained("onnx/vae.opt.onnx"),
    #     # vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
    #     text_encoder=OnnxRuntimeModel.from_pretrained("onnx/clip.opt.onnx"),
    #     tokenizer=make_tokenizer(version, hf_token),
    #     unet=OnnxRuntimeModel.from_pretrained("onnx/unet.opt.onnx"),
    #     scheduler=scheduler,
    #     safety_checker=safety_checker,
    #     feature_extractor=feature_extractor,
    #     # requires_safety_checker=safety_checker is not None,
    # )

    onnx_pipeline = OnnxStableDiffusionPipeline.from_pretrained(output_path, provider='CUDAExecutionProvider') # "CPUExecutionProvider")
    
    print(" *** ONNX models is loaded")


    prompt = "a dog on a lawn with the eifel tower in the background"
    negative_prompt = "blurry, low quality"
    width = 512
    height = 512
    num_inference_steps = 50
    guidance_scale = 7.5
    generator=None         # numpy.random.seed(args.seed)

    image = onnx_pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, 
                          guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                          generator=generator).images[0]
    
    imgname="testpicture_"+str(width)+"_"+".png"
    image.save(imgname)


onnx_runtime("onnx_15")
# onnx_runtime("/home/gyf/models/onnx_inference_models/model_inference/stable_diffusion/onnx/")
# convert_models("runwayml/stable-diffusion-v1-5", "onnx-1-5", 14)