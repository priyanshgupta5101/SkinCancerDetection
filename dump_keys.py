import torch
import sys

try:
    checkpoint = torch.load('model_best.pth', map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    with open('model_keys.txt', 'w') as f:
        for k, v in state_dict.items():
            f.write(f"{k}: {v.shape}\n")
            
except Exception as e:
    print(e)
