import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("loading models module with device:", DEVICE)

