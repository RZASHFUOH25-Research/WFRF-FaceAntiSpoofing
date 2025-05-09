import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm  # For displaying progress bar

# Input image directory
input_dir = r''

# Output directory for saving processed images
output_dir = r''
os.makedirs(output_dir, exist_ok=True)

# Function to create radial frequency weight matrix
def create_weight_matrix(height, width, low_freq_weight=None, high_freq_weight=None):
    center_y, center_x = height // 2, width // 2
    weights = np.ones((height, width))

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if distance > 0:
                max_distance = np.sqrt(center_y**2 + center_x**2)
                normalized_distance = distance / max_distance
                weights[y, x] = low_freq_weight + (high_freq_weight - low_freq_weight) * normalized_distance

    # Assign 1 to the center (DC component)
    weights[center_y, center_x] = 1
    return weights

# Frequency filtering function for a single image channel
def process_frequency(image_channel, weights):
    f = np.fft.fft2(image_channel)
    fshift = np.fft.fftshift(f)
    fshift_weighted = fshift * weights
    f_ishift = np.fft.ifftshift(fshift_weighted)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

# Get list of JPG image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

# Process each image
for filename in tqdm(image_files, desc="Processing..."):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        continue  # Skip corrupted images

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape

    # Create frequency weight matrix
    weights = create_weight_matrix(height, width, low_freq_weight=0.5, high_freq_weight=5)

    # Process each channel separately
    output_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(channels):
        output_image[:, :, channel] = process_frequency(image[:, :, channel], weights)

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Save the processed image
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

print(f"\n Processing complete. Output saved to: {output_dir}")
