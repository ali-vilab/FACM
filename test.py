"""
FLOW-ANCHORED CONSISTENCY MODELS
Copyright (c) 2024 The FACM Authors. All Rights Reserved.
"""


import argparse
import os
import zipfile

import numpy as np
import torch
import torch_fidelity
import torchvision
import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from ema_pytorch import PostHocEMA

from ldit.model_manager import create_models_from_config
from sampler import consistency_model_sampler, euler_sampler
from utils import ImagePack, RandomStateManager, decode_image, get_latent_stats, log


@torch.no_grad()
def evaluate(
    args,
    model,
    vae,
    accelerator,
    step,
    visualize_dir,
    noise=None,
    cond=None,
    start=1000,
    cfgw=1,
    fid=False,
    num_samples=5000,
    save_images=False,
):

    all_samples = []
    device = accelerator.device
    temp_dir_name = args.temp if hasattr(args, "temp") and args.temp else "temp"
    tmp_dir = os.path.join(
        "output", temp_dir_name, f"fid_image_{args.sampling_steps}_steps", "generated"
    )
    if fid and accelerator.is_main_process:
        if os.path.exists(tmp_dir):
            tmp_dir_base = tmp_dir
            counter = 1
            while os.path.exists(tmp_dir):
                tmp_dir = f"{tmp_dir_base}_{counter}"
                counter += 1

        os.makedirs(tmp_dir, exist_ok=True)

    save_dir = f"{visualize_dir}/sample_{step:07d}"
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    accelerator.wait_for_everyone()

    with torch.no_grad():
        eval_batch_size = 8 if not fid else 125

        # Get process count and global rank
        world_size = accelerator.num_processes
        global_rank = accelerator.process_index

        num_collect = eval_batch_size * world_size

        # If not computing FID, save images for each step count
        if not fid:
            # Define step sizes to test
            step_sizes = [1, 2, 4, 8]

            # Fix initial noise and conditions to compare sampling results across different step counts
            if noise is None:
                fixed_noise = torch.randn(
                    eval_batch_size,
                    32,
                    args.image_size // 16,
                    args.image_size // 16,
                    device=device,
                )
            else:
                fixed_noise = noise.clone()

            if cond is None:
                fixed_cond = torch.randint(
                    0, 1000, (eval_batch_size,), device=device, dtype=torch.int64
                )
            else:
                fixed_cond = cond.clone()

            # Generate samples for each step count
            for num_steps in step_sizes:
                # Use the same noise and conditions
                noise_copy = fixed_noise.clone()

                # Sample
                if args.sampler == "euler":
                    sampled_x = euler_sampler(
                        model,
                        noise_copy,
                        fixed_cond,
                        num_steps=num_steps,
                        cfg_scale=cfgw,
                        guidance_low=args.glow,
                        timestep_shift=args.timestep_shift,
                    )
                else:
                    sampled_x = consistency_model_sampler(
                        model,
                        noise_copy,
                        fixed_cond,
                        num_steps=num_steps,
                        cfg_scale=cfgw,
                        guidance_low=args.glow,
                    )

                _x_hat = (decode_image(sampled_x, vae) + 1) / 2

                # Collect results
                x_hat_gathered = accelerator.gather(_x_hat)
                label_gathered = accelerator.gather(fixed_cond)

                # Save images
                if accelerator.is_main_process:
                    img = torchvision.utils.make_grid(x_hat_gathered[:64], nrow=8)
                    img = torchvision.transforms.functional.to_pil_image(
                        img.cpu().float()
                    )
                    img.save(f"{save_dir}/t{start:04d}_cfgw{cfgw}_steps{num_steps}.png")

                log(
                    f"(step={step}) <t={start}> steps={num_steps} images saved to {save_dir}/t{start:04d}_cfgw{cfgw}_steps{num_steps}.png",
                    accelerator,
                )

            # All step count results have been generated and saved, can return
            return 0

        # Original FID calculation logic
        x_hat = torch.zeros(
            (eval_batch_size, 3, args.image_size, args.image_size), device=device
        )
        label = torch.zeros((eval_batch_size,), device=device, dtype=torch.int64)

        if fid:
            conds = torch.arange(1000, device=device, dtype=torch.int64).repeat(100)

        num_batches = int(
            np.ceil(
                (args.num_eval if not fid else num_samples)
                / eval_batch_size
                / world_size
            )
        )
        bar = tqdm.trange(
            num_batches,
            desc="FID",
            disable=(not accelerator.is_main_process or not fid),
        )
        for i in bar:
            x_noise = (
                torch.randn(
                    eval_batch_size,
                    32,
                    args.image_size // 16,
                    args.image_size // 16,
                    device=device,
                )
                if (noise is None or fid)
                else noise
            )
            if fid:
                cond = conds[
                    i * num_collect
                    + global_rank * eval_batch_size : i * num_collect
                    + (global_rank + 1) * eval_batch_size
                ]
            elif cond is None:
                cond = torch.randint(
                    0, 1000, (eval_batch_size,), device=device, dtype=torch.int64
                )

            if args.sampler == "euler":
                x_noise = euler_sampler(
                    model,
                    x_noise,
                    cond,
                    num_steps=args.sampling_steps,
                    cfg_scale=cfgw,
                    guidance_low=args.glow,
                    timestep_shift=args.timestep_shift,
                )
            else:
                x_noise = consistency_model_sampler(
                    model,
                    x_noise,
                    cond,
                    num_steps=args.sampling_steps,
                    cfg_scale=cfgw,
                    guidance_low=args.glow,
                )

            _x_hat = (decode_image(x_noise, vae) + 1) / 2
            x_hat = _x_hat
            label = cond

            if fid:
                # Use accelerate to collect results
                x_hat_gathered = accelerator.gather(x_hat).cpu()
                label_gathered = accelerator.gather(label)
                all_samples.append(x_hat_gathered)

                # Save individual images if requested
                if save_images and accelerator.is_main_process:
                    for j in range(x_hat_gathered.shape[0]):
                        img_idx = i * num_collect + j
                        if img_idx >= num_samples:
                            break
                        img = torchvision.transforms.functional.to_pil_image(
                            x_hat_gathered[j].cpu().float()
                        )
                        img.save(f"{tmp_dir}/{img_idx:05d}.png")

        ret = 0
        if fid:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Create zip file if individual images were saved
                if save_images:
                    log(
                        f"Compressing {len(os.listdir(tmp_dir))} images to zip file...",
                        accelerator,
                    )
                    with zipfile.ZipFile(
                        f"{save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip", "w"
                    ) as img_zip:
                        for filename in os.listdir(tmp_dir):
                            if filename.endswith(".png"):
                                img_zip.write(os.path.join(tmp_dir, filename), filename)
                    log(
                        f"Individual images saved to {save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip",
                        accelerator,
                    )

                # Calculate FID
                stat = f"fid-{int(np.ceil(num_samples / 1000))}k-{args.image_size}.npz"
                all_samples = torch.cat(all_samples)[:num_samples]
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=ImagePack(all_samples),
                    input2=None,
                    fid_statistics_file=f"cache/{stat}",
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=True,
                )
                inception_score = metrics_dict["inception_score_mean"]
                inception_score_std = metrics_dict["inception_score_std"]
                ret = fid_score = metrics_dict["frechet_inception_distance"]

                # Update log message based on whether images were saved
                if save_images:
                    log(
                        f"(step={step}) <t={start}> FID={fid_score:.4f}, \
                        IS={inception_score:.4f}±{inception_score_std:.4f}, \
                        images saved to {save_dir}/eval_{num_samples}_cfgw{cfgw}_images.zip",
                        accelerator,
                    )
                else:
                    log(
                        f"(step={step}) <t={start}> FID={fid_score:.4f}, \
                        IS={inception_score:.4f}±{inception_score_std:.4f}",
                        accelerator,
                    )

                del all_samples, metrics_dict

            accelerator.wait_for_everyone()

    return ret


def main(args):

    accelerator = Accelerator()
    device = accelerator.device

    set_seed(args.global_seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        log_file = f"{args.results_dir}/test_log.txt"
        log.log_file = log_file

    # init models
    vae, model = create_models_from_config(
        config_path="ldit/lightningdit.yaml", num_classes=args.num_classes
    )

    checkpoint_path = args.ckpt_path
    ema_checkpoint_dir = os.path.dirname(checkpoint_path) + "/ema_checkpoints"
    emas = PostHocEMA(
        model,
        sigma_rels=(0.05, 0.20),
        checkpoint_folder=ema_checkpoint_dir,
    )

    for blocks in model.blocks:
        blocks.attn.fused_attn = True

    log(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", accelerator
    )
    log(f"Loading model from {args.ckpt_path}......", accelerator)

    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    except:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    if "emas" in checkpoint:
        missing_keys, unexpected_keys = emas.load_state_dict(
            checkpoint["emas"], strict=False
        )
        model = emas.ema_models[1]
    else:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["ema"], strict=False
        )

    model = model.to(device)

    for sigma_rel in [args.sigma_rel]:
        if "emas" in checkpoint and sigma_rel != 0.2:
            log(f"Synthesizing EMA model, sigma_rel={sigma_rel}", accelerator)
            model = emas.synthesize_ema_model(sigma_rel=sigma_rel).to(device)
            model = model.to(device)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            log(f"Missing keys: {missing_keys}", accelerator)
        log(f"Unexpected keys: {unexpected_keys}", accelerator)

        log("Evaluation only", accelerator)

        # Select sampler
        log(f"Using {args.sampler} sampler", accelerator)

        with RandomStateManager(eval_seed=args.global_seed + accelerator.process_index):
            fid = evaluate(
                args,
                model,
                vae,
                accelerator,
                0,
                args.results_dir,
                cfgw=args.cfgw,
                fid=True,
                num_samples=args.num_eval,
                save_images=args.save_images,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="output/evalulation/try",
        help="Results save directory",
    )
    parser.add_argument(
        "--image-size", type=int, choices=[256, 512], default=256, help="Image size"
    )
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Number of classes"
    )
    parser.add_argument("--global-seed", type=int, default=0, help="Global random seed")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="output/0000000.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--num-eval", type=int, default=50000, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--sampling-steps", type=int, default=2, help="Number of sampling steps"
    )
    parser.add_argument("--cfgw", type=float, default=1.0, help="Configuration weight")
    parser.add_argument(
        "--temp", type=str, default="temp", help="Temporary directory name"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "consistency"],
        default="consistency",
        help="Choose sampler type: euler or consistency",
    )
    parser.add_argument("--glow", type=float, default=0.125)
    parser.add_argument("--timestep-shift", type=float, default=0.0)
    parser.add_argument("--sigma-rel", type=float, default=0.2)
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save individual images as zip file when computing FID",
    )
    args = parser.parse_args()
    main(args)
