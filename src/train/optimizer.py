import torch
import inspect

def get_optimizer(model, logger, weight_decay=0.1, learn_rate=6e-4, device='cpu'):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if isinstance(learn_rate, str):
        learn_rate = float(learn_rate)

    sig = inspect.signature(torch.optim.AdamW).parameters
    adamw_kwargs = dict(lr=learn_rate, betas=(0.9, 0.95), eps=1e-8)
    if 'weight_decay' not in adamw_kwargs:
        pass
    if ('fused' in sig) and (device == 'cuda'):
        adamw_kwargs['fused'] = True  # only pass when available

    logger.info(f"Using {'fused' if adamw_kwargs.get('fused', False) else 'non-fused'} AdamW")
    return torch.optim.AdamW(groups, **adamw_kwargs)