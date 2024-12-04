import torch
from torch import nn, optim
from torchvision.transforms import transforms
from torchvision import datasets
import numpy as np

import wandb
import random
import itertools

from train import train_epoch
from val import val_epoch

from NiN import NiN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.mps:
    device = "mps"

print(f"device:{device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.FashionMNIST(root="../data", download=True, train=True, transform=transform)
val_dataset = datasets.FashionMNIST(root="../data", download=False, train=False, transform=transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        
def main():
    lr = 1e-3
    num_epochs = 300
    
    model = NiN(1, 10).to(device)
    # 用SGD的话根本无法收敛
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch+1, num_epochs, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, epoch+1, num_epochs, device)
        
    return model
        
if __name__ == "__main__":
    model = main()