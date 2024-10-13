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

def make_image_prediction(IMAGE_PATH):
    x = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))

    transforms = A.Compose([
                A.RandomRotate90(), 
                A.Rotate(limit=10, p=0.5), 
                A.HorizontalFlip(p=0.5),  
                A.VerticalFlip(p=0.1),    
                A.RandomBrightnessContrast(p=0.2),  
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),  
                A.CLAHE(clip_limit=2.0, p=0.3),  
                A.Normalize(mean=[0.485], std=[0.229]), 
                ToTensorV2(), 
            ])

    transformed = transforms(image=np.asarray(x))

    torch_tensor_image = transformed['image']

    class_names = [
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


    def map_predictions_to_classes(predictions, class_names):
        mapped_predictions = []
        for i, pred in enumerate(predictions):
            classes = [class_names[j] for j in range(len(pred)) if pred[j] == 1]
            mapped_predictions.append({
                "sample_index": i,
                "diagnosed with": classes
            })
        return mapped_predictions

    x=model(torch_tensor_image.unsqueeze(0))
    x_b = (x == x.max(dim=1, keepdim=True)[0]).float()
    return map_predictions_to_classes(x_b,class_names)[0]['diagnosed with'][0]