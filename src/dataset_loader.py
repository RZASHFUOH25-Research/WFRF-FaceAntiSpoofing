import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    """
    Dataset for loading and transforming face anti-spoofing images.
    Loads RGB images, resizes them, applies transforms, and returns label.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load and validate image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (224, 224))
        image_rgb = Image.fromarray(image_rgb)

        if self.transform:
            image_rgb = self.transform(image_rgb)

        label = self.labels[idx] if self.labels is not None else -1
        return image_rgb, label


def extract_labels(image_paths):
    """
    Extracts labels based on the filename pattern.
    Assumes that the label is the last underscore-separated token in the filename:
    e.g., 'video_frame_1.jpg' → label 1, 'video_frame_0.jpg' → label 0
    """
    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        try:
            last_number = parts[-1].split('.')[0]
            labels.append(1 if last_number == '1' else 0)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid filename format: {filename}")
    return labels


def load_dataset(real_dir, spoof_dir, batch_size=30, shuffle=True):
    """
    Loads real and spoof images, creates DataLoader for training.

    Args:
        real_dir (str): Path to real images.
        spoof_dir (str): Path to spoof images.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Combined DataLoader for real and spoof samples.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    spoof_paths = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) if f.endswith(('.jpg', '.png'))]

    labels_real = [1] * len(real_paths)
    labels_spoof = [0] * len(spoof_paths)

    all_paths = real_paths + spoof_paths
    all_labels = labels_real + labels_spoof

    dataset = CustomImageDataset(all_paths, all_labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f" Loaded {len(real_paths)} real and {len(spoof_paths)} spoof images.")
    return loader