import torch
import traceback

try:
    checkpoint = torch.load('model_best.pth', map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    keys = ['1.4.weight', 'module.1.4.weight', '1.8.weight', 'module.1.8.weight']
    for k in keys:
        if k in state_dict:
            print(f"{k}: {state_dict[k].shape}")
        
except Exception:
    traceback.print_exc()
