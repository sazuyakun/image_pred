import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, lr_scheduler

from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights
import torchvision.transforms as tfms
import torchvision.transforms.functional as T

from PIL import Image
import requests
from io import BytesIO
import os

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
    x = load_image(IMAGE_PATH)

    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transforms(image=np.asarray(x))

    torch_tensor_image = transformed['image']

    torch_tensor_image = torch_tensor_image.unsqueeze(0).to(CFG.DEVICE) 
    model.eval() 
    with torch.no_grad():
        x = model(torch_tensor_image)
        
    x_b = (x == x.max(dim=1, keepdim=True)[0]).float()
    return map_predictions_to_classes(x_b, CFG.CLASS_NAMES)[0]['diagnosed with'][0]



# img_url="./imag.png"

# result = make_image_prediction(img_url)
# print(result)