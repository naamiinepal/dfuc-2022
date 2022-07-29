import torch

def get_optimizer(name, model, learning_rate):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), learning_rate)

