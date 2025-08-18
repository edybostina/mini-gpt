import logging
import torch
from .checkpointing import save_checkpoint
from utils import eval_loss, get_learning_rate
import time


def train_loop(model, data_loader, optimizer, device, cfg, logger):
    seed = cfg['seed']
    batch_size = cfg['batch_size']
    micro_batch_size = cfg['micro_batch_size']
    seq_length = cfg['seq_length']
    max_steps = cfg['max_steps']
    warmup_steps=cfg['warmup_steps']
    min_learning_rate=cfg['min_learning_rate']
    max_learning_rate=cfg['max_learning_rate']


    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    assert batch_size % (micro_batch_size * seq_length) == 0, "batch size must be divisible by micro batch size times seq length"
    gradient_accumulation_steps = batch_size // (micro_batch_size * seq_length)
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    torch.set_float32_matmul_precision('high') # use TF32 (or try to)

    model.to(device)
    model = torch.compile(model) # this might make it run faster (try to run it with and without)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    for i in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device,enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / gradient_accumulation_steps
            accumulated_loss += float(loss.item())
            scaler.scale(loss).backward()
            # logger.info(f"micro_step_loss: {loss.item():.6f}")


        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_learning_rate(i, max_steps=max_steps, warmup_steps=warmup_steps, min_learning_rate=min_learning_rate, max_learning_rate=max_learning_rate)
        for g in optimizer.param_groups:
            g['lr'] = lr
        scaler.step(optimizer)
        scaler.update()

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        dt = (time.time() - t0)
        tps = (micro_batch_size * seq_length * gradient_accumulation_steps) / dt
        logger.info(f"step {i:4d} | lr {lr:.4e} | loss {accumulated_loss:.6f} | time {dt*1000:.1f}ms | grad_norm {norm:.3f} | tok/s {tps:.1f}")

        if (i+1) % 50 == 0:
            val = eval_loss(model, data_loader, iters=50, device=device)  # quick&dirty on same data file
            logger.info(f"[eval] val_loss {val:.4f}")
            save_checkpoint(f"ckpt_step_{i+1}.pt", model, optimizer)
