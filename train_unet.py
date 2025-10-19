import torch
from torch.utils.data import DataLoader, TensorDataset
from models import AttentionUNet
import torch.optim as optim

def train_model(train_x, train_y, device='cuda'):
    model = AttentionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(train_x.to(device))
        loss = loss_fn(pred, train_y.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    torch.save(model.state_dict(), "runs/unet/best.pth")
