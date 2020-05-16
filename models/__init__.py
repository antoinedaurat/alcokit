import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("loading models module with device:", device)