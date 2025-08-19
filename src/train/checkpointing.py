import torch

def save_checkpoint(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': model.config.__dict__,
    }, path)

def load_checkpoint(model, optimizer, device, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            setattr(model.config, key, value)
    model.to(device)
    return model, optimizer