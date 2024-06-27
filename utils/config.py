import torch.optim as optim
from models.model_architecture import MuSTGAN_MFAS

def get_model_optimizer():
    model = MuSTGAN_MFAS(num_modalities=3)
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    return model, optimizer
