import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

print(torch.xpu.is_available()) # Prints if intel GPU or iGPU is avail

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")