import torch
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import densenet121
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as tfms
import torchvision.transforms.functional as T
import numpy as np

class CFG:
    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    # BASE_PATH = Path("/kaggle/input/nih-chest-x-ray-14-224x224-resized")
    # BEST_MODEL_PATH = "models/best_model.pt"
    EPOCHS = 20
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    INTERVAL = 10
    
class DenseNet121(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.2):
        super(DenseNet121, self).__init__()
        
       
        self.densenet121 = densenet121(pretrained=True)
        
       
        n_features = self.densenet121.classifier.in_features
        
        
        
        
       
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_features),  
            nn.Dropout(dropout_rate),  
            nn.Linear(n_features, n_classes),  
            nn.Sigmoid()  
        )
        
        
        self.densenet121.classifier = self.classifier

    def forward(self, x):
        return self.densenet121(x)

model = DenseNet121(n_classes=14)  # Assuming 14 output classes in your case
model = model.to(CFG.DEVICE)

model_path = './cheXNET.pth'
checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint) 


import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

def load_image(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        # If it's a URL, download the image
        try:
            response = requests.get(image_path)
            # Check for a valid response (status code 200) and Content-Type
            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                image = Image.open(BytesIO(response.content))
            else:
                raise ValueError("Invalid image content or URL response")
        except Exception as e:
            raise ValueError(f"Error fetching the image from the URL: {e}")
    elif os.path.isfile(image_path):
        # If it's a local file path, open the image
        image = Image.open(image_path)
    else:
        raise ValueError("Invalid image path provided.")
    
    return np.asarray(image.convert("RGB"))




def map_predictions_to_classes(predictions, class_names):
    mapped_predictions = []
    for i, pred in enumerate(predictions):
        classes = [class_names[j] for j in range(len(pred)) if pred[j] == 1]
        mapped_predictions.append({
            "sample_index": i,
            "diagnosed with": classes
        })
    return mapped_predictions


def make_image_prediction(IMAGE_PATH):
    # Load the image
    x = load_image(IMAGE_PATH)

    # Define the transforms with a resize step to match DenseNet121 input (224x224)
    transforms = A.Compose([
                A.Resize(224, 224),  # Resize image to 224x224 (standard for DenseNet121)
                A.RandomRotate90(), 
                A.Rotate(limit=10, p=0.5), 
                A.HorizontalFlip(p=0.5),  
                A.VerticalFlip(p=0.1),    
                A.RandomBrightnessContrast(p=0.2),  
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),  
                A.CLAHE(clip_limit=2.0, p=0.3),  
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                ToTensorV2(), 
            ])

    # Apply the transforms
    transformed = transforms(image=np.asarray(x))

    # Get the transformed tensor
    torch_tensor_image = transformed['image']

    # Ensure the tensor is in the correct shape [batch_size, 3, height, width]
    torch_tensor_image = torch_tensor_image.unsqueeze(0)  # Add batch dimension

    # Pass the image through the model
    x = model(torch_tensor_image)

    # Apply thresholding to get binary predictions
    x_b = (x == x.max(dim=1, keepdim=True)[0]).float()

    # Map predictions to class names
    return map_predictions_to_classes(x_b, CFG.CLASS_NAMES)[0]['diagnosed with'][0]


img_url="./img.png"

result = make_image_prediction(img_url)
print(result)