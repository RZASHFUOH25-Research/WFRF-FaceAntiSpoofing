import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from dataset_loader import CustomImageDataset, extract_labels
from model import EnhancedResNet18


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedResNet18(pretrained=True).to(device)
model_path = " "   ## Path to trained model
model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare test data
test_folder = r" "
test_real = [os.path.join(test_folder, "Real", f) for f in os.listdir(os.path.join(test_folder, "Real")) if f.endswith(('.jpg', '.png'))]
test_spoof = [os.path.join(test_folder, "Spoof", f) for f in os.listdir(os.path.join(test_folder, "Spoof")) if f.endswith(('.jpg', '.png'))]
test_paths = test_real + test_spoof
test_labels = [1] * len(test_real) + [0] * len(test_spoof)

test_dataset = CustomImageDataset(test_paths, test_labels, transform=None)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Evaluation function

def test_model(model, test_loader, threshold=None):
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    false_positives = []
    false_negatives = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Fixed extra parenthesis
            norms = torch.norm(outputs, p=2, dim=1)
            predictions = (norms < threshold).long()

            # Handle false positives and false negatives
            for i in range(len(predictions)):
                global_idx = batch_idx * test_loader.batch_size + i
                img_path = test_loader.dataset.image_paths[global_idx]
                if predictions[i] == 0 and labels[i] == 1:  # False Positive
                    false_positives.append(img_path)
                elif predictions[i] == 1 and labels[i] == 0:  # False Negative
                    false_negatives.append(img_path)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(norms.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN = FP = FN = TP = 0

    apcer = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    bpcer = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    acer = (apcer + bpcer) / 2

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    print(f"\nThreshold:          {threshold:.4f}")
    print(f"APCER (FP Rate):      {apcer:.4f}")
    print(f"BPCER (FN Rate):      {bpcer:.4f}")
    print(f"ACER:                 {acer:.4f}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return all_predictions, all_scores, false_positives, false_negatives

#Threshold from train stage
predictions, scores = test_model(model, test_loader, threshold = 20)

