from tqdm import tqdm
import torch

def train_epoch(model, data_loader, optimizer, criterion, epoch, num_epochs, device='cpu'):
    total_correct = 0
    total_loss = 0
    total_item = 0
    
    pbar = tqdm(data_loader)
    for image, label in pbar:
        image, label = image.to(device), label.to(device)
        pred = model(image)

        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().item() * image.shape[0]
        correct = (pred.argmax(dim=1) == label).sum().cpu().item()
        total_correct += correct
        
        total_item += image.shape[0]

        pbar.set_description(f"[{epoch}/{num_epochs}] Train: loss={loss.item():.3f}, acc={correct/image.shape[0]:.3f}")

    pbar.write(f"[{epoch}/{num_epochs}] Train: loss={total_loss/total_item}, acc={total_correct/total_item}")

    return total_loss/total_item, total_correct/total_item
