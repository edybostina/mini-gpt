import torch

def save_checkpoint(path, model, optimizer, step=0):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': model.config.__dict__,
        'step': step,
    }, path)

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            setattr(model.config, key, value)
    model.to(device)
    return model, optimizer