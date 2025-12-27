import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the model structure (FastAI style ResNet34)
def get_model(num_classes=7):
    # Load DenseNet169 backbone
    backbone = models.densenet169(weights=None)
    
    # FastAI body for DenseNet is just the features
    # Checkpoint keys are 0.0.conv0... which implies body is nn.Sequential(features)
    body = nn.Sequential(backbone.features)
    
    # Head
    # DenseNet121 features = 1024
    # If AvgPool: 1024 -> 1024
    # If ConcatPool: 1024 -> 2048
    # Checkpoint has Linear(1024, 512) -> So AvgPool
    
    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(3328),
        nn.Dropout(0.25),
        nn.Linear(3328, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    model = nn.Sequential(body, head)
    return model

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def predict_image_pytorch(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=7)
    
    # Load weights
    try:
        checkpoint = torch.load('model_best.pth', map_location=device, weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Handle DataParallel module. prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False) # strict=False to be safe, but we hope for match
    except Exception as e:
        print(f"Error loading model: {e}")
        return "Error"

    model.eval()
    model.to(device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            idx = predicted.item()
            
        # Mapping (Assuming HAM10000 alphabetical)
        # 0: akiec (Cancer)
        # 1: bcc (Cancer)
        # 2: bkl (Non-Cancer)
        # 3: df (Non-Cancer)
        # 4: mel (Cancer)
        # 5: nv (Non-Cancer)
        # 6: vasc (Non-Cancer)
        
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        class_name = classes[idx]
        
        cancer_classes = ['akiec', 'bcc', 'mel'] # vasc is usually benign
        
        # Calculate confidence
        confidence = probs[0][idx].item() * 100
        
        if class_name in cancer_classes:
            return "Cancer", confidence
        else:
            return "NonCancer", confidence
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {str(e)}", 0.0

if __name__ == "__main__":
    # Test loading
    print("Testing model loading...")
    model = get_model()
    try:
        checkpoint = torch.load('model_best.pth', map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Debug shapes
        print(f"Model 1.4.weight shape: {model[1][4].weight.shape}")
        if '1.4.weight' in state_dict:
             print(f"Checkpoint 1.4.weight shape: {state_dict['1.4.weight'].shape}")
        
        print(f"Model 0.0.conv0.weight shape: {model[0][0].conv0.weight.shape}")
        if '0.0.conv0.weight' in state_dict:
             print(f"Checkpoint 0.0.conv0.weight shape: {state_dict['0.0.conv0.weight'].shape}")
             
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print("Model loaded successfully with strict=True!")
    except Exception as e:
        print(f"Failed to load with strict=True: {e}")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded with strict=False.")
        except Exception as e2:
            print(f"Failed to load even with strict=False: {e2}")
