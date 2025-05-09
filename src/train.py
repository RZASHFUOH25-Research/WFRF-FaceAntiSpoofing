import os
import torch
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np
from model import EnhancedResNet18
from model import ManhattanLossV2
from dataset_loader import CustomImageDataset, extract_labels
from torchvision import transforms



def plot_features(epoch_features, epoch_labels, epoch, r=None):
    # Flatten the feature tensors and convert them to NumPy arrays
    features_tensor = torch.cat(epoch_features, dim=0).cpu().detach().numpy()
    labels_tensor = torch.cat(epoch_labels, dim=0).cpu().detach().numpy()

    r = 2
    # Plotting 2D t-SNE visualization (first two dimensions of the features)
    plt.figure(figsize=(8, 8))
    
    # Scatter plot: Real samples in blue, Spoof samples in red
    plt.scatter(features_tensor[labels_tensor == 1, 0], features_tensor[labels_tensor == 1, 1],
                c='blue', label='Real', alpha=0.5)
    plt.scatter(features_tensor[labels_tensor == 0, 0], features_tensor[labels_tensor == 0, 1],
                c='red', label='Spoof', alpha=0.5)

    # Draw a circle with radius r at (0,0) (you can adjust position if needed)
    circle = plt.Circle((0, 0), r, color='green', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

    plt.title(f'Epoch {epoch + 1} - Feature Scatter')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    
# Initialize device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedResNet18(pretrained=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = ManhattanLossV2(r=2, m=13, beta_n=1.0, beta_a=1.0).to(device)

# Define the transformation for image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Real / Spoof samples
train_folder_real = r" "
train_folder_spoof = r" "

train_paths_real = [os.path.join(train_folder_real, f) for f in os.listdir(train_folder_real) if f.endswith(('.jpg', '.png'))]
train_paths_spoof = [os.path.join(train_folder_spoof, f) for f in os.listdir(train_folder_spoof) if f.endswith(('.jpg', '.png'))]

train_paths = train_paths_real + train_paths_spoof
train_labels = [1] * len(train_paths_real) + [0] * len(train_paths_spoof)

# Create dataset and dataloader
train_dataset = CustomImageDataset(train_paths, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)

# Directory for saving models
save_dir = r" "
os.makedirs(save_dir, exist_ok=True)

# List to store loss values for plotting
losses = []

# Best threshold variables
best_threshold = 24
best_auc = 0.0

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    epoch_features = []
    epoch_labels = []
    all_scores = []
    all_true_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        epoch_features.append(outputs.cpu().detach())
        epoch_labels.append(labels.cpu().detach())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Store output and true labels for ROC calculation
        all_scores.append(outputs.cpu().detach().numpy())
        all_true_labels.append(labels.cpu().detach().numpy())

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)  # Append loss to the list

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    scheduler.step()

    # Flatten all_scores and all_true_labels
    all_scores = np.concatenate(all_scores)
    all_true_labels = np.concatenate(all_true_labels)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_scores[:, 1])  # Use the second column (spoof probability)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold (where the sum of TPR and (1-FPR) is maximized)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    best_auc = roc_auc

    print(f"Epoch [{epoch+1}/{num_epochs}], Best AUC: {best_auc:.4f}, Best Threshold: {best_threshold:.4f}")

    # Save model checkpoint
    model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Show scatter plot every 5 epochs
    if (epoch + 1) % 10 == 0:
        plot_features(epoch_features, epoch_labels, epoch)

# Final best threshold after training
print(f"Best Threshold after training: {best_threshold:.4f}")
