import torch
import math
import yaml
import random
import numpy as np
import os


@torch.no_grad()
def eval_loss(model, loader, iters=50, device='cpu'):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# can also use https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
# but where is the fun in that?
def get_learning_rate(iteration, max_steps=50, warmup_steps=10, min_learning_rate=6e-5, max_learning_rate=6e-4):
    if iteration < warmup_steps:
        return max_learning_rate * (iteration + 1) / warmup_steps
    
    if iteration > max_steps:
        return min_learning_rate
    
    cosine_decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= cosine_decay_ratio <= 1
    coefficient = 0.5 * (1 + math.cos(cosine_decay_ratio * math.pi))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * coefficient

def get_optimal_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


# load config from yaml file
def load_config(path):
    assert path.endswith('.yaml'), "Config file must be a YAML file"
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int, deterministic: bool = False) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # optional deterministic behavior (may slow down and might not be fully deterministic across platforms)
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
