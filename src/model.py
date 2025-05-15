import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class EnhancedResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(EnhancedResNet18, self).__init__()

        # Use the new `weights` API
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Remove the final classification layer
        self.resnet = torch.nn.Sequential(*list(backbone.children())[:-1])

        # Fully connected layers
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)

        x = self.fc1(features)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.fc4(x)
        return x


class ManhattanLossV2(nn.Module):
    def __init__(self, r=None, m=None, beta_l=1.0, beta_s=1.0):
        super(ManhattanLossV2, self).__init__()
        self.r = r  
        self.m = m  
        self.beta_l = beta_l  
        self.beta_s = beta_s  

    def forward(self, features, labels):
        mask_normal = (labels == 1)  
        mask_abnormal = (labels == 0)  

        normal_features = features[mask_normal]
        abnormal_features = features[mask_abnormal]

        N_l = normal_features.size(0)
        N_s = abnormal_features.size(0)

        loss_l = torch.tensor(0.0, device=features.device)
        loss_s = torch.tensor(0.0, device=features.device)

        if N_l > 0:
            # Calculate the Manhattan distance for live features
            manhattan_l = torch.sum(torch.abs(normal_features), dim=1)  # ||f_j^n||_1
            loss_l = torch.mean(torch.max(manhattan_l - self.r ** 2, torch.tensor(0.0, device=features.device)))
            loss_l *= self.beta_l / N_l  

        if N_s > 0:
            # Calculate the Manhattan distance for spoof features
            manhattan_s = torch.sum(torch.abs(abnormal_features), dim=1)  # ||f_j^a||_1
            loss_s = torch.mean(torch.max((self.m **2) - manhattan_s, torch.tensor(0.0, device=features.device)))
            loss_s *= self.beta_s / N_s 

        total_loss = loss_l + loss_s
        return total_loss
