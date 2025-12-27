import torch
from pytorch_model import predict_image_pytorch
import os

# Create a dummy image if one doesn't exist, or use an existing one
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Create a simple dummy image
from PIL import Image
img = Image.new('RGB', (224, 224), color = 'red')
img.save('uploads/debug_image.jpg')

print("Starting debug prediction...")
result, confidence = predict_image_pytorch('uploads/debug_image.jpg')
print(f"Result: {result}")
print(f"Confidence: {confidence}")
