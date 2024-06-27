import torch
import os

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'model_best.pth.tar')

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{filename}'")
