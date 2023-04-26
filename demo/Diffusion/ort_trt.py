# Note: The <name>_ort.onnx files were created as follows:
#    python -m onnxruntime.tools.symbolic_shape_infer --input clip.onnx --output clip_ort.onnx --auto_merge
#    python -m onnxruntime.tools.symbolic_shape_infer --input unet.onnx --output unet_ort.onnx --auto_merge
#    python -m onnxruntime.tools.symbolic_shape_infer --input vae.onnx --output vae_ort.onnx --auto_merge

import argparse
import numpy as np
import os
import onnxruntime as ort
import torch
import time

from tqdm import tqdm
from onnxruntime.transformers.benchmark_helper import measure_memory
from transformers import CLIPTokenizer
from utilities import LMSDiscreteScheduler, DPMScheduler, save_image


def get_args():
    parser = argparse.ArgumentParser()
    # User settings
    parser.add_argument(
        "--prompt",
        default="a beautiful photograph of Mt. Fuji during cherry blossom",
        type=str,
        help="Text prompt(s) to guide image generation",
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--num-warmup-runs", default=5, type=int)
    parser.add_argument(
        "--denoising-steps", default=50, type=int, help="Number of inference steps"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--onnx-dir",
        default="./onnx",
        type=str,
        help="Output directory for ONNX export",
    )

    # Pipeline configuration
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--tokenizer",
        default=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    )
    parser.add_argument("--scheduler", default="lmsd", choices=["dpm", "lmsd"])
    parser.add_argument("--io-binding", action="store_true")
    parser.add_argument(
        "--denoising-prec",
        default="fp16",
        choices=["fp16", "fp32"],
        help="Denoiser model precision",
    )

    parser.add_argument(
        "--num_images", default=1, type=int, help="Number of images per prompt"
    )
    parser.add_argument(
        "--negative-prompt",
        default=[""],
        help="The negative prompt(s) to guide the image generation.",
    )
    parser.add_argument("--guidance-scale", default=7.5, type=float)
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for logs and image artifacts",
    )

    parser.add_argument("--use_explicit_profiles", default=1, type=int)

    args = parser.parse_args()

    # Set prompt sizes
    args.prompt = [args.prompt for _ in range(args.batch_size)]
    args.negative_prompt = args.negative_prompt * args.batch_size

    # Set scheduler
    sched_opts = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012}
    if args.scheduler == "lmsd":
        setattr(
            args, "scheduler", LMSDiscreteScheduler(device=args.device, **sched_opts)
        )
    else:
        setattr(args, "scheduler", DPMScheduler(device=args.device, **sched_opts))
    args.scheduler.set_timesteps(args.denoising_steps)
    args.scheduler.configure()
    return args


def to_np(tensor, new_dtype):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()
    assert isinstance(tensor, np.ndarray)
    return tensor.astype(new_dtype) if tensor.dtype != new_dtype else tensor


def to_pt(tensor, new_dtype):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    assert torch.is_tensor(tensor)
    return tensor.type(new_dtype) if tensor.dtype != new_dtype else tensor


def run_ort_trt(model_path, input_args, output_args=None, use_io_binding=False):
    """
    Run the stable diffusion pipeline with ORT-TRT.

    Args:
        model_path (str):
            Path to the ONNX model.
        input_args (dict):
            The input arguments needed to run the InferenceSession. This can be in two formats:

            With IO Binding:
                {
                    'name': <input name>,
                    'device_type': 'cuda',
                    'device_id': 0,
                    'element_type': <input element type>,
                    'shape': <input shape>,
                    'buffer_ptr': <pointer to allocated input>
                }

            Without IO Binding:
                {
                    '<model-input-1-name>': <model-input-1-data>,
                    '<model-input-2-name>': <model-input-2-data>,
                    ...
                }
        output_args (dict, optional):
            The output arguments needed for IO Binding. The format is:

            {
                'name': <output name>,
                'device_type': 'cuda',
                'device_id': 0,
                'element_type': <output element type>,
                'shape': <output shape>,
                'buffer_ptr': <pointer to allocated output>
            }
        use_io_binding (bool):
            Whether to use IO Binding or not

            For details on IO Binding, visit https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
    """
    sess = ort.InferenceSession(
        model_path, providers=["TensorrtExecutionProvider"]
    )  # , 'CUDAExecutionProvider'])
    if use_io_binding:
        io_binding = sess.io_binding()
        if isinstance(input_args, list):
            for input_arg in input_args:
                assert isinstance(input_arg, dict)
                io_binding.bind_input(**input_arg)
        else:
            assert isinstance(input_args, dict)
            io_binding.bind_input(**input_args)

        if isinstance(output_args, list):
            for output_arg in output_args:
                assert isinstance(output_arg, dict)
                io_binding.bind_output(**output_arg)
        else:
            assert isinstance(output_args, dict)
            io_binding.bind_output(**output_args)

        outputs = sess.run_with_iobinding(io_binding)
        # return io_binding.copy_outputs_to_cpu()
    else:
        outputs = sess.run(None, input_args)

    return outputs


# Modified from demo-diffusion.py
def run_pipeline(args, clip_sess, unet_sess, vae_sess):
    latent_height, latent_width = args.height // 8, args.width // 8
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with torch.inference_mode(), torch.autocast(args.device):
        # latents need to be generated on the target device
        unet_channels = 4
        latents_shape = (
            args.batch_size * args.num_images,
            unet_channels,
            latent_height,
            latent_width,
        )
        latents_dtype = torch.float32
        latents = torch.randn(
            latents_shape, device=args.device, dtype=latents_dtype, generator=generator
        )

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * args.scheduler.init_noise_sigma

        torch.cuda.synchronize()
        start_time = time.time()

        # Tokenizer input
        torch.cuda.synchronize()
        clip_start_time = time.time()

        text_input_ids = args.tokenizer(
            args.prompt,
            padding="max_length",
            max_length=args.tokenizer.model_max_length,
            return_tensors="np",
        ).input_ids.astype(np.int32)

        # CLIP text encoder with text embeddings
        text_input_args = {"input_ids": text_input_ids}
        text_embeddings = clip_sess.run(None, text_input_args)[0]
        text_embeddings = to_pt(text_embeddings, torch.float32)

        # Duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, args.num_images, 1)
        text_embeddings = text_embeddings.view(bs_embed * args.num_images, seq_len, -1)

        max_length = text_input_ids.shape[-1]
        uncond_input_ids = args.tokenizer(
            args.negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        ).input_ids.astype(np.int32)

        # CLIP text encoder with uncond embeddings
        uncond_input_args = {"input_ids": uncond_input_ids}
        uncond_embeddings = clip_sess.run(None, uncond_input_args)[0]
        uncond_embeddings = to_pt(uncond_embeddings, torch.float32)

        # Duplicate unconditional embeddings for each generation per prompt
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, args.num_images, 1)
        uncond_embeddings = uncond_embeddings.view(
            args.batch_size * args.num_images, seq_len, -1
        )

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if args.denoising_prec == "fp16":
            text_embeddings = text_embeddings.to(dtype=torch.float16)

        torch.cuda.synchronize()
        clip_end_time = time.time()

        torch.cuda.synchronize()
        unet_start_time = time.time()
        for step_index, timestep in enumerate(tqdm(args.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            # LMSDiscreteScheduler.scale_model_input()
            latent_model_input = args.scheduler.scale_model_input(
                latent_model_input, step_index
            )

            # predict the noise residual
            dtype = np.float16 if args.denoising_prec == "fp16" else np.float32
            if timestep.dtype != torch.float32:
                timestep_float = timestep.float()
            else:
                timestep_float = timestep

            # UNet with sample, timestep, and encoder hidden states
            unet_args = {
                "sample": to_np(latent_model_input, np.float32),
                "timestep": np.array(
                    [to_np(timestep_float, np.float32)], dtype=np.float32
                ),
                "encoder_hidden_states": to_np(text_embeddings, dtype),
            }
            noise_pred = unet_sess.run(None, unet_args)[0]
            noise_pred = to_pt(noise_pred, torch.float16).to(args.device)

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = args.scheduler.step(noise_pred, latents, step_index, timestep)

        latents = 1.0 / 0.18215 * latents
        torch.cuda.synchronize()
        unet_end_time = time.time()

        # VAE with latents
        torch.cuda.synchronize()
        vae_start_time = time.time()

        vae_input_args = {"latent": to_np(latents, np.float32)}
        images = vae_sess.run(None, vae_input_args)[0]
        images = to_pt(images, torch.float32)

        torch.cuda.synchronize()
        vae_end_time = time.time()

        torch.cuda.synchronize()
        end_time = time.time()
        if not args.warmup:
            print("|------------|--------------|")
            print("| {:^10} | {:^12} |".format("Module", "Latency"))
            print("|------------|--------------|")
            print(
                "| {:^10} | {:>9.2f} ms |".format(
                    "CLIP", (clip_end_time - clip_start_time) * 1000
                )
            )
            print(
                "| {:^10} | {:>9.2f} ms |".format(
                    "UNet x " + str(args.denoising_steps),
                    (unet_end_time - unet_start_time) * 1000,
                )
            )
            print(
                "| {:^10} | {:>9.2f} ms |".format(
                    "VAE", (vae_end_time - vae_start_time) * 1000
                )
            )
            print("|------------|--------------|")
            print(
                "| {:^10} | {:>10.2f} s |".format("Pipeline", (end_time - start_time))
            )
            print("|------------|--------------|")

            # Save image
            image_name_prefix = (
                "sd-"
                + args.denoising_prec
                + "".join(
                    set(
                        [
                            "-" + args.prompt[i].replace(" ", "_")[:10]
                            for i in range(args.batch_size)
                        ]
                    )
                )
                + "-"
            )
            save_image(images, args.output_dir, image_name_prefix)


def main():
    args = get_args()
    print(args)

    # Load models and convert to FP16 with first inference passes to reduce latency
    batch_size = args.batch_size
    min_batch = 1
    max_batch = 16
    embed_dim = 768
    unet_dim = 4
    max_text_len = 77
    latent_height = args.height // 8
    latent_width = args.width // 8

    min_image_shape = 256  # min image resolution: 256x256
    max_image_shape = 1024  # max image resolution: 1024x1024
    min_latent_shape = self.min_image_shape // 8
    max_latent_shape = self.max_image_shape // 8
    min_latent_height = min_latent_shape
    min_latent_width = min_latent_shape
    max_latent_height = min_latent_shape
    max_latent_width = min_latent_shape
    opt_batch = 1
    opt_latent_height = 64
    opt_latent_width = 64

    engine_tag = f"{max_batch}_{latent_height}_{latent_width}_{embed_dim}_v1"

    if args.seed > 0:
        ort.set_default_logger_severity(0)

    print("Loading CLIP model.")

    trt_ep_default_options = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_max_workspace_size": 0,
        #"trt_timing_cache_enable": True,
        #"trt_detailed_build_log": True,
    }

    if args.use_explicit_profiles:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/clip_ort_engine_{engine_tag}",
            "trt_profile_min_shapes": f"input_ids:{min_batch}x{max_text_len}",
            "trt_profile_max_shapes": f"input_ids:{max_batch}x{max_text_len}",
            "trt_profile_opt_shapes": f"input_ids:{opt_batch}x{max_text_len}",
            "trt_engine_cache_built_with_explicit_profiles": True,
        }
    else:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/clip_ort_engine_{batch_size}",
        }

    trt_ep_options.update(trt_ep_default_options)

    clip_sess = ort.InferenceSession(
        "./onnx_1.5/clip_ort.onnx",
        providers=[
            ("TensorrtExecutionProvider", trt_ep_options),
            "CUDAExecutionProvider",
        ],
    )
    clip_sess.run(
        None, {"input_ids": np.zeros((batch_size, max_text_len), dtype=np.int32)}
    )

    print("Loading UNet model.")
    sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    if args.use_explicit_profiles:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/unet_ort_engine_{engine_tag}",
            "trt_profile_min_shapes": f"sample:{2 * min_batch}x{unet_dim}x{min_latent_height}x{min_latent_width},encoder_hidden_states:{2 * min_batch}x{max_text_len}x{embed_dim},timestep:1",
            "trt_profile_max_shapes": f"sample:{2 * max_batch}x{unet_dim}x{max_latent_height}x{max_latent_width},encoder_hidden_states:{2 * max_batch}x{max_text_len}x{embed_dim},timestep:1",
            "trt_profile_opt_shapes": f"sample:{2 * opt_batch}x{unet_dim}x{opt_latent_height}x{opt_latent_width},encoder_hidden_states:{2 * opt_batch}x{max_text_len}x{embed_dim},timestep:1",
            "trt_engine_cache_built_with_explicit_profiles": True,
        }
    else:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/unet_ort_engine_{batch_size}",
        }

    trt_ep_options.update(trt_ep_default_options)

    unet_sess = ort.InferenceSession(
        "./onnx_1.5/unet_ort.onnx",
        providers=[
            ("TensorrtExecutionProvider", trt_ep_options),
            "CUDAExecutionProvider",
        ],
    )
    unet_args = {
        "sample": np.zeros(
            (2 * batch_size, unet_dim, latent_height, latent_width), dtype=np.float32
        ),
        "timestep": np.ones((1,), dtype=np.float32),
        "encoder_hidden_states": np.zeros(
            (2 * batch_size, max_text_len, embed_dim),
            dtype=np.float16 if args.denoising_prec == "fp16" else np.float32,
        ),
    }
    unet_sess.run(None, unet_args)

    if args.use_explicit_profiles:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/vae_ort_engine_{engine_tag}",
            "trt_profile_min_shapes": f"latent:{min_batch}x{unet_dim}x{min_latent_height}x{min_latent_width}",
            "trt_profile_max_shapes": f"latent:{max_batch}x{unet_dim}x{max_latent_height}x{max_latent_width}",
            "trt_profile_opt_shapes": f"latent:{opt_batch}x{unet_dim}x{opt_latent_height}x{opt_latent_width}",
            "trt_engine_cache_built_with_explicit_profiles": True,
        }
    else:
        trt_ep_options = {
            "trt_engine_cache_path": f"./onnx_1.5/vae_ort_engine_{batch_size}",
        }

    trt_ep_options.update(trt_ep_default_options)

    print("Loading VAE model.")
    vae_sess = ort.InferenceSession(
        "./onnx_1.5/vae_ort.onnx",
        sess_options,
        providers=[
            ("TensorrtExecutionProvider", trt_ep_options),
            "CUDAExecutionProvider",
        ],
    )
    vae_sess.run(
        None,
        {
            "latent": np.zeros(
                (batch_size, unet_dim, latent_height, latent_width), dtype=np.float32
            )
        },
    )

    # Warm up pipeline
    setattr(args, "warmup", True)
    print("Warming up pipeline.")
    for _ in tqdm(range(args.num_warmup_runs)):
        run_pipeline(args, clip_sess, unet_sess, vae_sess)
    setattr(args, "warmup", False)

    run_pipeline(args, clip_sess, unet_sess, vae_sess)

    # Measure each batch size
    init_prompt = args.prompt[0]
    init_negative_prompt = args.negative_prompt[:1]
    for bs in [args.batch_size]:
        # args.batch_size = bs
        print(f"\nBatch size = {bs}\n")

        args.prompt = [init_prompt for _ in range(args.batch_size)]
        args.negative_prompt = init_negative_prompt * args.batch_size

        # Measure latency
        print("Measuring latency.")
        run_pipeline(args, clip_sess, unet_sess, vae_sess)

        # Measure memory usage
        print("Measuring memory usage. Ignore any latency metrics or any images saved.")
        measure_memory(
            is_gpu=(args.device == "cuda"),
            func=lambda: run_pipeline(args, clip_sess, unet_sess, vae_sess),
        )
        print("Measured memory usage.")


if __name__ == "__main__":
    main()
