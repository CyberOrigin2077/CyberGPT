"""
Trainging Code for World Model Trainging.
Implemented by CyberOrigin
"""
import os
import time
import math
import logging
import argparse

import mup
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from omegaconf import OmegaConf
from tqdm import tqdm

from cyber.dataset.RawTokenDataset import RawTokenDataset
from cyber.models.world import WorldModel


def parse_args():
    parser = argparse.ArgumentParser(description="Training Code for World Model Training")
    parser.add_argument('--train-config-path', type=str, required=True, help='Path to the training configuration file')
    parser.add_argument('--model-config-path', type=str, required=True, help='Path to the model configuration file')
    args_ = parser.parse_args()
    return args_


def save_checkpoint(model, optimizer, lr_scheduler, cfg, epoch, filename):
    """
    Save the model, optimizer, and lr_scheduler state to a checkpoint file.
    """
    save_path = os.path.join(cfg.output_dir, filename)
    os.makedirs(save_path, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'completed_steps': float(filename.split('step_')[-1]),
        'epoch': epoch,
    }, os.path.join(save_path, 'checkpoint.pth'))


def main():
    ##############################################
    # Config
    ##############################################
    args_ = parse_args()
    # Load the training and model configuration file
    train_cfg = OmegaConf.load(args_.train_config_path)
    model_cfg = OmegaConf.load(args_.model_config_path)
    # Merge the two configurations
    cfg = OmegaConf.merge(train_cfg, model_cfg)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    ##############################################
    # Dataloader and Model Definition
    ##############################################
    train_token_dataset = RawTokenDataset(cfg.train_data_dir, window_size=16)
    val_token_dataset = RawTokenDataset(cfg.val_data_dir, window_size=16)
    world_model = WorldModel(cfg).to(device)

    train_token_dataset.valid_start_inds = train_token_dataset.valid_start_inds[:12]
    val_token_dataset.valid_start_inds = val_token_dataset.valid_start_inds[:12]

    latent_side_len, vocab_size, hz = [train_token_dataset.metadata[key] for key in ("s", "vocab_size", "hz")]
    if cfg.mu_transfer:
        world_model.dynamic.set_mup_shapes(rescale_params=True)
        world_model.dynamic.init_weights()

    if cfg.dynamic.name == "Genie":
        from cyber.models.world.dynamic.genie.utils import get_maskgit_collator
        collate_fn = get_maskgit_collator(cfg.dynamic.init_args)

    train_dataloader = DataLoader(
        train_token_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=cfg.per_device_train_batch_size, num_workers=4, pin_memory=True,
    )
    eval_dataloader = DataLoader(
        val_token_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=cfg.per_device_train_batch_size, num_workers=4, pin_memory=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    ##############################################
    # Optimizer and Scheduler
    ##############################################

    # Optimizer. Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in world_model.dynamic.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in world_model.dynamic.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt_class = mup.MuAdamW if cfg.mu_transfer else torch.optim.AdamW
    optimizer = opt_class(optimizer_grouped_parameters, lr=cfg.learning_rate,
                          betas=(cfg.adam_beta_1, cfg.adam_beta_2), eps=cfg.adam_eps)

    if cfg.lr_scheduler_type == "custom_cosine":  # decay to `end_ratio` of the peak learning rate
        def get_lr_wrapper(warmup_steps, max_steps, end_ratio=0.1):
            def get_lr(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps

                remaining_steps = max_steps - warmup_steps
                return ((1 + math.cos(math.pi * (step - warmup_steps) / remaining_steps)) / 2) \
                    * (1 - end_ratio) + end_ratio
            return get_lr

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, get_lr_wrapper(cfg.num_warmup_steps, cfg.max_train_steps)
        )
    else:
        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=cfg.max_train_steps
        )

    # Enable gradient checkpointing to save memory
    if cfg.gradient_checkpointing:
        logging.info("Enabling gradient checkpointing")
        world_model.dynamic.gradient_checkpointing_enable()
        world_model.dynamic.config.use_cache = False

    if not cfg.no_compile:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674
        # TODO: https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
        world_model.dynamic = torch.compile(world_model.dynamic)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save states
    checkpointing_steps = cfg.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    ##############################################
    # Training
    ##############################################

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint is not None or cfg.resume_from_checkpoint != "":
            checkpoint_path = cfg.resume_from_checkpoint
            path = os.path.basename(cfg.resume_from_checkpoint.rstrip("/"))
        else:
            raise ValueError("Must provide a checkpoint path to resume training from")

        checkpoint = torch.load(checkpoint_path)
        world_model.dynamic.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        completed_steps = int(checkpoint['completed_steps'])
        starting_epoch = int(checkpoint['epoch'])

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    loss_info = torch.zeros(2, device=device)  # sum, count

    for epoch in range(starting_epoch, cfg.num_train_epochs):
        world_model.dynamic.train()
        active_dataloader = train_dataloader

        _time = time.time()
        for step, batch in enumerate(active_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            batch_size = batch["input_ids"].size(0)
            # Manual gradient accumulation because accelerator somehow taking a lot of memory
            is_update_step = (step + 1) % cfg.gradient_accumulation_steps == 0
            with torch.set_grad_enabled(True):

                outputs = world_model.dynamic(**batch)
                loss = outputs[0]
                loss_info[0] += loss.detach() * batch_size
                loss_info[1] += batch_size

                loss.backward()

            if not is_update_step:
                continue

            # Everything below only happens on update step

            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(world_model.dynamic.parameters(), cfg.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            avg_train_loss = (loss_info[0] / loss_info[1]).item()  # sum / count
            loss_info *= 0  # reset sum and count
            try:
                perplexity = math.exp(avg_train_loss)
            except OverflowError:
                perplexity = float("inf")

            batch_time = time.time() - _time  # accumulated batch
            _time = time.time()
            logging.info(
                {
                    "train_perplexity": perplexity,
                    "train_loss": avg_train_loss,
                    "epoch": epoch,
                    "update_step": completed_steps,
                    "examples_processed": completed_steps * cfg.per_device_train_batch_size
                                          * cfg.gradient_accumulation_steps,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                })

            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                save_checkpoint(world_model.dynamic, optimizer, lr_scheduler, cfg, epoch, f"step_{completed_steps}")

            # if completed_steps % cfg.eval_every_n_steps == 0:
            if True:
                world_model.dynamic.eval()

                eval_losses = []

                # Compute token-level accuracy (w/ teacher forcing)
                num_correct = 0
                num_total = 0
                for step, batch in enumerate(eval_dataloader):
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    batch_size = len(batch["input_ids"])  # Last batch might not be full
                    with torch.no_grad():
                        outputs = world_model.dynamic(**batch)

                    loss = outputs[0]
                    eval_losses.append(loss.repeat(batch_size))

                    if outputs[1] is not None:
                        num_correct += outputs[1].mean().item() * batch_size
                        num_total += batch_size
                    else:
                        shifted_preds = torch.argmax(outputs[0][:, :-1, :], dim=-1)
                        shifted_labels = batch["labels"][:, 1:]
                        num_correct += (shifted_preds == shifted_labels).sum().item()
                        num_total += torch.numel(shifted_labels)

                    if step >= cfg.max_eval_steps:
                        break

                eval_losses = torch.cat(eval_losses)
                eval_loss = torch.mean(eval_losses).item()
                eval_teacher_acc = num_correct / num_total
                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logging.info(f"{completed_steps=} {perplexity=} {eval_loss=} {eval_teacher_acc=}")

                # Switch back to train mode
                world_model.dynamic.train()

            if completed_steps >= cfg.max_train_steps:
                break

        if cfg.checkpointing_steps == "epoch":
            save_checkpoint(world_model.dynamic, optimizer, lr_scheduler, cfg, epoch, f"step_{completed_steps}")

    save_checkpoint(world_model.dynamic, optimizer, lr_scheduler, cfg, epoch, f"step_{completed_steps}")


if __name__ == "__main__":
    main()
