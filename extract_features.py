import torch, numpy as np
from models import AttentionUNet

def extract_features(volume):
    model = AttentionUNet()
    model.load_state_dict(torch.load("runs/unet/best.pth", map_location='cpu'))
    model.eval()
    with torch.no_grad():
        _, feats = model(torch.tensor(volume).unsqueeze(0).float())
        vector = np.concatenate([f.flatten() for f in feats], axis=0)
        np.save("features.npy", vector)
