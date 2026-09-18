import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=37):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
