"""
FLOW-ANCHORED CONSISTENCY MODELS
Copyright (c) 2024 The FACM Authors. All Rights Reserved.
"""


import os
import copy
import argparse
import time
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from ema_pytorch import PostHocEMA

from ldit.lightningdit import LightningDiT_models
from ldit.model_manager import create_vae_from_config, create_dit_from_config
from ldit.vavae import VA_VAE
from losses import FACMLoss, MeanFlowLoss, sCMLoss
from utils import (
    log, LinearScheduler, DummyScheduler, RandomStateManager,
    get_dataset, load_ckpt, save_ckpt, evaluate
)


def setup_experiment(args, accelerator):
    """Setup experiment directories and logging"""
    checkpoint_dir = f'{args.results_dir}/checkpoint'
    visualize_dir = f'{args.results_dir}/visualize'
    ema_checkpoint_dir = f'{checkpoint_dir}/ema_checkpoints'

    if accelerator.is_main_process:
        log_file = f'{args.results_dir}/training_log.txt'
        log.log_file = log_file
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        os.makedirs(ema_checkpoint_dir, exist_ok=True)
        log(f'Created experiment directory at {args.results_dir}', accelerator)

    return checkpoint_dir, visualize_dir, ema_checkpoint_dir


def auto_resume_checkpoint(args, accelerator):
    """Auto-resume from latest checkpoint if exists"""
    latest_ckpt = os.path.join(args.results_dir, 'checkpoint', 'latest.pt')
    if os.path.exists(latest_ckpt):
        original_ckpt_path = args.ckpt_path
        args.ckpt_path = latest_ckpt
        log(f">>>>>> Auto-resuming from {args.ckpt_path} <<<<<<", accelerator)
        return original_ckpt_path
    return args.ckpt_path


def create_model_and_ema(args, accelerator, ema_checkpoint_dir):
    config_path = "ldit/lightningdit.yaml"
    
    # Create model using unified configuration manager
    model = create_dit_from_config(
        config_path=config_path,
        num_classes=args.num_classes
    )
    
    log(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}", accelerator)
    
    # Create EMA
    ema = PostHocEMA(
        model,
        sigma_rels=[0.05, args.sigma_rel],
        update_every=1,
        checkpoint_every_num_steps=5000,
        checkpoint_folder=ema_checkpoint_dir,
    )

    if not args.debug:
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    return model, compiled_model, ema


def setup_training_components(model, args):
    """Setup optimizer and scheduler"""
    all_params = list(model.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta), eps=args.eps)
    scheduler = DummyScheduler()
    facm_loss = FACMLoss()
    
    return opt, scheduler, facm_loss


def setup_teacher_model(args, model, start_step, original_ckpt_path, accelerator):
    """Setup teacher model for distillation"""
    if not args.distill:
        return None
        
    freezed_teacher = copy.deepcopy(model.module)
    
    if start_step > 0 and original_ckpt_path:
        original_checkpoint = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
        log(f'Loading Teacher model from {original_ckpt_path}', accelerator)
        missing_keys, unexpected_keys = freezed_teacher.load_state_dict(
            original_checkpoint['ema'], strict=False
        )
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            log(f'Teacher model - Missing keys: {missing_keys}', accelerator)
            log(f'Teacher model - Unexpected keys: {unexpected_keys}', accelerator)
    
    # Freeze teacher parameters
    for param in freezed_teacher.parameters():
        param.requires_grad = False
    freezed_teacher.eval()
    
    if not args.debug:
        freezed_teacher = torch.compile(freezed_teacher)
    
    return freezed_teacher


def perform_evaluation_and_checkpointing(args, model, emas, vae, accelerator, train_steps, 
                                       visualize_dir, eval_noise, eval_cond, start_step, 
                                       checkpoint_dir, opt, epoch, scheduler):
    """Handle evaluation and checkpointing during training"""
    cfgw = args.cfgw if not args.distill else 1.0
    
    # Regular checkpointing
    if train_steps % args.ckpt_every == 0 and train_steps > start_step:
        save_ckpt(args, model, emas, opt, epoch, train_steps, checkpoint_dir, 
                 accelerator, scheduler=scheduler)

    # Regular evaluation
    if train_steps % args.eval_every == 0:
        synthesized_ema = emas.ema_models[1]
        with RandomStateManager(eval_seed=args.global_seed + accelerator.process_index):
            evaluate(args, synthesized_ema, vae, accelerator, train_steps, visualize_dir, 
                    noise=eval_noise, cond=eval_cond, cfgw=cfgw)

    # FID evaluation
    if train_steps % args.fid_every == 0 and train_steps > start_step:
        synchronize_emas_across_processes(emas, train_steps, accelerator, args, force_sync=True)
        _perform_fid_evaluation(args, model, emas, vae, accelerator, train_steps, visualize_dir, 
                              cfgw, checkpoint_dir, opt, epoch, scheduler)


def _perform_fid_evaluation(args, model, emas, vae, accelerator, train_steps, visualize_dir, 
                          cfgw, checkpoint_dir, opt, epoch, scheduler):
    """Perform FID evaluation with multiple sampling steps"""
    synthesized_ema = emas.ema_models[1]
    
    for step in [1, 2, 4]:
        args.sampling_steps = step
        with RandomStateManager(eval_seed=args.global_seed + accelerator.process_index):
            outfid_local = evaluate(args, synthesized_ema, vae, accelerator, train_steps, 
                                    visualize_dir, cfgw=cfgw, fid=True, num_samples=50000, save_images=args.save_images) 
        outfid_tensor = torch.tensor(outfid_local, device=accelerator.device, dtype=torch.float32)
        outfid50k = accelerator.reduce(outfid_tensor, reduction="max").item()


def synchronize_emas_across_processes(emas, train_steps, accelerator, args, force_sync=False):
    """Synchronize EMA parameters across all processes"""
    if accelerator.num_processes <= 1:
        return
    if not torch.distributed.is_initialized():
        return
    if train_steps % (args.accumulation * 10) == 0 or force_sync:
        with torch.no_grad():
            for ema_model in emas.ema_models:
                for param in ema_model.parameters():
                    torch.distributed.broadcast(param.data, src=0)


def training_loop(args, model, compiled_model, emas, opt, scheduler, facm_loss, loader_train, accelerator, 
                 start_epoch, start_step, device, eval_noise, eval_cond, vae, 
                 visualize_dir, checkpoint_dir, freezed_teacher):
    """Main training loop"""
    epoch, train_steps = start_epoch, start_step
    log_steps, running_loss = 0, 0
    running_cm_loss, running_fm_loss = 0, 0
    start_time = time.time()
    
    while train_steps < args.max_steps:
        log(f"Starting epoch {epoch}...", accelerator)
        
        for data in loader_train:
            if train_steps >= args.max_steps:
                break

            with accelerator.accumulate(model):
                images, labels = data
                x = images.to(device)
                y = labels.to(device)
                
                # Perform evaluation and checkpointing
                perform_evaluation_and_checkpointing(
                    args, model, emas, vae, accelerator, train_steps, visualize_dir, 
                    eval_noise, eval_cond, start_step, checkpoint_dir, opt, epoch, scheduler
                )

                # Forward pass
                model_kwargs = dict(y=y)
                loss, cm_loss, fm_loss = facm_loss(
                    accelerator, model, compiled_model, emas.ema_models[1], x, train_steps,
                    model_kwargs=model_kwargs, args=args,
                    freezed_teacher=freezed_teacher
                )
                loss = torch.nan_to_num(loss, nan=0.0)
            
                # Backward pass and optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                
            if accelerator.sync_gradients:
                emas.update()
                synchronize_emas_across_processes(emas, train_steps, accelerator, args)
            
            # Logging
            running_loss += loss.item()
            running_cm_loss += cm_loss.item()
            running_fm_loss += fm_loss.item()
            log_steps += 1
            train_steps += 1

            # Periodic logging
            if train_steps % args.log_every == 0:
                _log_training_stats(accelerator, device, args, running_loss, running_cm_loss, 
                                  running_fm_loss, log_steps, start_time, grad_norm, train_steps)
                running_loss, running_cm_loss, running_fm_loss = 0, 0, 0
                log_steps = 0
                start_time = time.time()
            
        epoch += 1

        if epoch >= args.epochs and train_steps < args.max_steps:
            log(f"Completed all preset epochs ({args.epochs}), continuing with continuous learning...", accelerator)


def _log_training_stats(accelerator, device, args, running_loss, running_cm_loss, 
                       running_fm_loss, log_steps, start_time, grad_norm, train_steps):
    """Log training statistics"""
    torch.cuda.synchronize()
    end_time = time.time()
    steps_per_sec = log_steps / (end_time - start_time)

    loss_names = ['loss', 'loss_cm', 'loss_fm']
    running_losses = [running_loss, running_cm_loss, running_fm_loss]
    avg_losses = {}
    
    for name, running_val in zip(loss_names, running_losses):
        avg_val = torch.tensor(running_val / log_steps, device=device)
        avg_val = accelerator.gather(avg_val).sum() / accelerator.num_processes
        avg_losses[name] = avg_val.item()
    
    avg_loss = [avg_losses[name] for name in loss_names]
    
    log(f"(step={train_steps}) loss_cm: {avg_loss[1]:.4f}, loss_fm: {avg_loss[2]:.4f}, "
        f"steps/sec: {steps_per_sec:.2f}, grad_norm: {grad_norm:.4f}", accelerator)


def main(args):
    """Main training function"""
    # Setup
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    device = accelerator.device
    args.num_eval = 8 * accelerator.num_processes if args.num_eval < 0 else args.num_eval
    set_seed(args.global_seed + accelerator.process_index)

    # Setup experiment
    checkpoint_dir, visualize_dir, ema_checkpoint_dir = setup_experiment(args, accelerator)
    original_ckpt_path = auto_resume_checkpoint(args, accelerator)

    # Log arguments
    log('Args:\n' + '\n'.join([f'\t{arg}: {getattr(args, arg)}' for arg in vars(args)]), accelerator)
    
    # Load data
    loader_train, dataset_train, loader_len = get_dataset(args, accelerator, args.data_dir)
    log(f"Dataset loaded", accelerator)

    # Setup model components
    vae = create_vae_from_config("ldit/lightningdit.yaml")
    model, compiled_model, emas = create_model_and_ema(args, accelerator, ema_checkpoint_dir)
    opt, scheduler, facm_loss = setup_training_components(model, args)

    # Prepare for training
    model, compiled_model, opt, scheduler, emas, loader_train = accelerator.prepare(
        model, compiled_model, opt, scheduler, emas, loader_train
    )
        
    # Load checkpoint and setup teacher
    start_epoch, start_step = load_ckpt(args, model, emas, opt, accelerator,
                                        scheduler=scheduler)
    freezed_teacher = setup_teacher_model(args, model, start_step, original_ckpt_path, accelerator)
    
    model.train()
    emas.eval()

    # Setup evaluation data
    eval_noise = torch.randn(8, 32, args.image_size // 16, args.image_size // 16, device=device)
    eval_cond = torch.randint(0, 1000, (8,), device=device, dtype=torch.int64)
    
    log(f"Training for {args.epochs} epochs...", accelerator)
    total_steps = args.epochs * loader_len // accelerator.num_processes
    log(f"Total steps: {total_steps}", accelerator)

    # Main training loop
    training_loop(args, model, compiled_model, emas, opt, scheduler, facm_loss, loader_train, accelerator, 
                 start_epoch, start_step, device, eval_noise, eval_cond, vae, 
                 visualize_dir, checkpoint_dir, freezed_teacher)
    
    # Final save
    save_ckpt(args, model, emas, opt, start_epoch, train_steps, checkpoint_dir, 
             accelerator, scheduler=scheduler)
    log("Training complete", accelerator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data and model arguments
    parser.add_argument("--results-dir", type=str, default="output")
    parser.add_argument("--data-dir", type=str, default="path/to/your/dataset")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument('--sigma-rel', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    
    # Logging and checkpointing
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--fid-every", type=int, default=10000)
    parser.add_argument('--ckpt-path', type=str, default=None)
    
    # Evaluation arguments
    parser.add_argument('--num-eval', type=int, default=-1)
    parser.add_argument('--sampling-steps', type=int, default=1)
    parser.add_argument('--sampling-method', type=str, default='consistency', choices=['euler', 'consistency'])
    parser.add_argument('--temp', type=str, default='temp')
    
    # Model specific arguments
    parser.add_argument('--t-type', type=str, default='default')
    parser.add_argument('--cfgw', type=float, default=1.75)
    parser.add_argument('--glow', type=float, default=0.125)
    parser.add_argument('--ghigh', type=float, default=1.0)
    parser.add_argument('--mean', type=float, default=0.8)
    parser.add_argument('--std', type=float, default=1.6)
    parser.add_argument('--p', type=float, default=0.5)

    # Training modes
    parser.add_argument('--distill', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--use-checkpoint', action='store_true', default=False)
    parser.add_argument('--max-steps', type=float, default=float('inf'))
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save individual images as zip file when computing FID')
    
    args = parser.parse_args()
    
    main(args)