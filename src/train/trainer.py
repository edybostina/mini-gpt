import time
import torch
import torch.nn as nn

from ..utils import get_learning_rate, eval_loss, set_seed, register_signal_handlers
from .checkpointing import save_checkpoint



def train_loop(model, data_loader, optimizer, device, cfg, logger, checkpoint_path="checkpoints", start_step: int = 0):

    seed = cfg.get("seed", 69)
    batch_size = cfg["batch_size"]
    micro_batch_size = cfg["micro_batch_size"]
    seq_length = cfg["seq_length"]
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    min_lr = cfg["min_learning_rate"]
    max_lr = cfg["max_learning_rate"]
    eval_interval = cfg.get("eval_interval", 50)
    eval_iters = cfg.get("eval_iters", 50)

    step_holder = {"step": start_step}
    register_signal_handlers(model, optimizer, step_holder, logger)

    set_seed(seed, deterministic=cfg.get("deterministic", False))

    assert batch_size % (micro_batch_size * seq_length) == 0, \
        "batch_size must be divisible by micro_batch_size * seq_length"

    gradient_accum_steps = batch_size // (micro_batch_size * seq_length)
    logger.info(f"Gradient accumulation steps: {gradient_accum_steps}")

    torch.set_float32_matmul_precision("high")
    model.to(device)

    try:
        model = torch.compile(model)
        logger.info("torch.compile applied to model")
    except Exception as e:
        logger.warning(f"torch.compile not applied: {e}")

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for step in range(start_step, max_steps):
        step_holder["step"] = step

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        # gradient accumulation
        for _ in range(gradient_accum_steps):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / gradient_accum_steps

            accumulated_loss += float(loss.item())
            scaler.scale(loss).backward()

        # optimizer update
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_learning_rate(
            step,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            min_learning_rate=min_lr,
            max_learning_rate=max_lr
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        # stats
        dt = time.time() - t0
        tps = (micro_batch_size * seq_length * gradient_accum_steps) / dt

        logger.info(
            f"step {step:4d} | "
            f"lr {lr:.4e} | "
            f"loss {accumulated_loss:.6f} | "
            f"time {dt*1000:.1f}ms | "
            f"grad_norm {norm:.3f} | "
            f"tok/s {tps:.1f}"
        )

        if (step + 1) % eval_interval == 0:
            val = eval_loss(model, data_loader, iters=eval_iters, device=device)
            logger.info(f"[eval] step {step+1} val_loss {val:.4f}")
            save_checkpoint(f"{checkpoint_path}/ckpt_step_{step+1}.pt", model, optimizer, step=step)
