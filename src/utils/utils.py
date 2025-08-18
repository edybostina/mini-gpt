import torch
import tiktoken
import math
import yaml

@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=100, temperature=1.0, top_p=0.9, device='cpu'):
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # crop to block_size to avoid pos-embed overflow
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-9, temperature)
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > top_p
        sorted_probs[cutoff] = 0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))

        next_id = torch.gather(sorted_idx, -1, torch.multinomial(sorted_probs, num_samples=1))
        idx = torch.cat([idx, next_id], dim=1)

    return enc.decode(idx[0].tolist())

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
